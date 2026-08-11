//! Durable, least-privilege credentials for remote daemon clients.
//!
//! Owner/bootstrap bearers remain process configuration in `brazierd`. Tokens
//! issued here belong to one named client, carry explicit scopes, can be
//! revoked without restarting the daemon, and are stored only as SHA-256
//! digests. Pairing codes have the same at-rest property and are short-lived,
//! single-use, and attempt-bounded.

use std::{
    collections::BTreeSet,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, Row as _};
use subtle::ConstantTimeEq as _;
use uuid::Uuid;

use brazier_protocol::execution_location::ExecutionLocation;

use crate::db::Database;

pub const DEFAULT_PAIRING_TTL_SECONDS: u64 = 5 * 60;
pub const MAX_PAIRING_TTL_SECONDS: u64 = 15 * 60;
pub const MAX_PAIRING_ATTEMPTS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientScope {
    Inference,
    Management,
    Agent,
}

impl ClientScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inference => "inference",
            Self::Management => "management",
            Self::Agent => "agent",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiClient {
    pub id: String,
    pub name: String,
    pub scopes: Vec<ClientScope>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<String>,
}

impl ApiClient {
    pub fn has_scope(&self, scope: ClientScope) -> bool {
        self.scopes.contains(&scope)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairingRequest {
    pub id: String,
    pub client_name: String,
    pub scopes: Vec<ClientScope>,
    pub expires_at: u64,
    pub attempts: u32,
    pub max_attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consumed_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct CreatedPairing {
    pub request: PairingRequest,
    /// Returned exactly once to the operator who starts pairing.
    pub code: String,
}

#[derive(Debug, Clone)]
pub struct IssuedClientCredential {
    pub client: ApiClient,
    /// Returned exactly once to the client claiming the pairing request.
    pub api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DaemonIdentity {
    pub instance_id: String,
    pub display_name: String,
    pub created_at: String,
}

impl DaemonIdentity {
    pub fn execution_location(&self) -> ExecutionLocation {
        ExecutionLocation::daemon(&self.instance_id, &self.display_name)
    }
}

#[derive(FromRow)]
struct ApiClientRow {
    id: String,
    name: String,
    scopes_json: String,
    created_at: String,
    last_used_at: Option<String>,
    revoked_at: Option<String>,
}

#[derive(FromRow)]
struct PairingRow {
    id: String,
    client_name: String,
    code_hash: String,
    scopes_json: String,
    expires_at: i64,
    attempts: i64,
    max_attempts: i64,
    consumed_at: Option<String>,
    created_at: String,
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn digest(secret: &str) -> String {
    hex::encode(Sha256::digest(secret.as_bytes()))
}

fn normalized_scopes(scopes: &[ClientScope]) -> anyhow::Result<Vec<ClientScope>> {
    let scopes: Vec<_> = scopes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    anyhow::ensure!(
        !scopes.is_empty(),
        "at least one client scope must be granted"
    );
    Ok(scopes)
}

fn valid_client_name(name: &str) -> anyhow::Result<String> {
    let name = name.trim();
    anyhow::ensure!(!name.is_empty(), "client name must not be empty");
    anyhow::ensure!(name.chars().count() <= 80, "client name is too long");
    anyhow::ensure!(
        !name.chars().any(char::is_control),
        "client name contains control characters"
    );
    Ok(name.to_owned())
}

fn client_from_row(row: ApiClientRow) -> anyhow::Result<ApiClient> {
    Ok(ApiClient {
        id: row.id,
        name: row.name,
        scopes: serde_json::from_str(&row.scopes_json).context("decode client scopes")?,
        created_at: row.created_at,
        last_used_at: row.last_used_at,
        revoked_at: row.revoked_at,
    })
}

fn pairing_from_row(row: &PairingRow) -> anyhow::Result<PairingRequest> {
    Ok(PairingRequest {
        id: row.id.clone(),
        client_name: row.client_name.clone(),
        scopes: serde_json::from_str(&row.scopes_json).context("decode pairing scopes")?,
        expires_at: row.expires_at.max(0) as u64,
        attempts: row.attempts.max(0) as u32,
        max_attempts: row.max_attempts.max(0) as u32,
        consumed_at: row.consumed_at.clone(),
        created_at: row.created_at.clone(),
    })
}

fn pairing_code() -> String {
    let raw = Uuid::new_v4().simple().to_string().to_uppercase();
    raw.as_bytes()
        .chunks(4)
        .map(|chunk| std::str::from_utf8(chunk).expect("UUID is ASCII"))
        .collect::<Vec<_>>()
        .join("-")
}

fn client_token(client_id: &str) -> String {
    format!(
        "brazier_client_{client_id}_{}{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    )
}

impl Database {
    /// Current daemon identity rendered as a durable execution snapshot.
    pub async fn execution_location(&self) -> anyhow::Result<ExecutionLocation> {
        Ok(self.daemon_identity(None).await?.execution_location())
    }

    /// Return the stable identity of this data directory, creating it on first
    /// use. An explicitly configured display name updates only the human label;
    /// the instance UUID never changes.
    pub async fn daemon_identity(
        &self,
        configured_name: Option<&str>,
    ) -> anyhow::Result<DaemonIdentity> {
        let display_name = match configured_name {
            Some(name) => valid_client_name(name)?,
            None => "Brazier daemon".to_owned(),
        };
        sqlx::query(
            r#"INSERT OR IGNORE INTO daemon_identity(singleton, instance_id, display_name)
               VALUES(1, ?, ?)"#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(&display_name)
        .execute(&self.pool)
        .await?;
        if configured_name.is_some() {
            sqlx::query("UPDATE daemon_identity SET display_name = ? WHERE singleton = 1")
                .bind(&display_name)
                .execute(&self.pool)
                .await?;
        }
        let row = sqlx::query(
            "SELECT instance_id, display_name, created_at FROM daemon_identity WHERE singleton = 1",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(DaemonIdentity {
            instance_id: row.try_get("instance_id")?,
            display_name: row.try_get("display_name")?,
            created_at: row.try_get("created_at")?,
        })
    }

    pub async fn create_pairing_request(
        &self,
        client_name: &str,
        scopes: &[ClientScope],
        ttl_seconds: u64,
    ) -> anyhow::Result<CreatedPairing> {
        let client_name = valid_client_name(client_name)?;
        let scopes = normalized_scopes(scopes)?;
        let ttl_seconds = ttl_seconds.clamp(1, MAX_PAIRING_TTL_SECONDS);
        let id = Uuid::new_v4().to_string();
        let code = pairing_code();
        let expires_at = now_epoch_seconds().saturating_add(ttl_seconds);
        sqlx::query(
            r#"INSERT INTO api_pairing_requests(
                   id, client_name, code_hash, scopes_json, expires_at, max_attempts)
               VALUES(?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(&client_name)
        .bind(digest(&code))
        .bind(serde_json::to_string(&scopes)?)
        .bind(expires_at as i64)
        .bind(i64::from(MAX_PAIRING_ATTEMPTS))
        .execute(&self.pool)
        .await?;
        let request = self
            .pairing_request(&id)
            .await?
            .context("created pairing request disappeared")?;
        Ok(CreatedPairing { request, code })
    }

    pub async fn pairing_request(&self, id: &str) -> anyhow::Result<Option<PairingRequest>> {
        let row = sqlx::query_as::<_, PairingRow>(
            r#"SELECT id, client_name, code_hash, scopes_json, expires_at, attempts,
                      max_attempts, consumed_at, created_at
               FROM api_pairing_requests WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(pairing_from_row).transpose()
    }

    pub async fn list_pairing_requests(&self) -> anyhow::Result<Vec<PairingRequest>> {
        let rows = sqlx::query_as::<_, PairingRow>(
            r#"SELECT id, client_name, code_hash, scopes_json, expires_at, attempts,
                      max_attempts, consumed_at, created_at
               FROM api_pairing_requests
               ORDER BY created_at DESC"#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(pairing_from_row).collect()
    }

    pub async fn cancel_pairing_request(&self, id: &str) -> anyhow::Result<bool> {
        Ok(
            sqlx::query("DELETE FROM api_pairing_requests WHERE id = ? AND consumed_at IS NULL")
                .bind(id)
                .execute(&self.pool)
                .await?
                .rows_affected()
                == 1,
        )
    }

    pub async fn claim_pairing_request(
        &self,
        id: &str,
        code: &str,
    ) -> anyhow::Result<IssuedClientCredential> {
        const UNAVAILABLE: &str = "pairing request is invalid or unavailable";
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query_as::<_, PairingRow>(
            r#"SELECT id, client_name, code_hash, scopes_json, expires_at, attempts,
                      max_attempts, consumed_at, created_at
               FROM api_pairing_requests WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&mut *tx)
        .await?
        .context(UNAVAILABLE)?;
        let now = now_epoch_seconds();
        anyhow::ensure!(
            row.consumed_at.is_none()
                && row.expires_at >= now as i64
                && row.attempts < row.max_attempts,
            UNAVAILABLE
        );

        let supplied_hash = digest(code.trim());
        if !bool::from(row.code_hash.as_bytes().ct_eq(supplied_hash.as_bytes())) {
            sqlx::query("UPDATE api_pairing_requests SET attempts = attempts + 1 WHERE id = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            anyhow::bail!(UNAVAILABLE);
        }

        let claimed = sqlx::query(
            r#"UPDATE api_pairing_requests
               SET consumed_at = datetime('now')
               WHERE id = ? AND consumed_at IS NULL AND expires_at >= ?
                     AND attempts < max_attempts"#,
        )
        .bind(id)
        .bind(now as i64)
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(claimed.rows_affected() == 1, UNAVAILABLE);

        let client_id = Uuid::new_v4().to_string();
        let api_key = client_token(&client_id);
        sqlx::query(
            r#"INSERT INTO api_clients(id, name, token_hash, scopes_json)
               VALUES(?, ?, ?, ?)"#,
        )
        .bind(&client_id)
        .bind(&row.client_name)
        .bind(digest(&api_key))
        .bind(&row.scopes_json)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;

        let client = self
            .api_client(&client_id)
            .await?
            .context("issued client credential disappeared")?;
        Ok(IssuedClientCredential { client, api_key })
    }

    pub async fn api_client(&self, id: &str) -> anyhow::Result<Option<ApiClient>> {
        let row = sqlx::query_as::<_, ApiClientRow>(
            r#"SELECT id, name, scopes_json, created_at, last_used_at, revoked_at
               FROM api_clients WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.map(client_from_row).transpose()
    }

    pub async fn list_api_clients(&self) -> anyhow::Result<Vec<ApiClient>> {
        let rows = sqlx::query_as::<_, ApiClientRow>(
            r#"SELECT id, name, scopes_json, created_at, last_used_at, revoked_at
               FROM api_clients ORDER BY created_at DESC"#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(client_from_row).collect()
    }

    /// Authenticate against the current durable state on every request. This
    /// makes revocation effective immediately, without a daemon restart or an
    /// in-memory invalidation race.
    pub async fn authenticate_api_client(
        &self,
        api_key: &str,
    ) -> anyhow::Result<Option<ApiClient>> {
        let token_hash = digest(api_key);
        let row = sqlx::query_as::<_, ApiClientRow>(
            r#"SELECT id, name, scopes_json, created_at, last_used_at, revoked_at
               FROM api_clients WHERE token_hash = ? AND revoked_at IS NULL"#,
        )
        .bind(token_hash)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        sqlx::query("UPDATE api_clients SET last_used_at = datetime('now') WHERE id = ?")
            .bind(&row.id)
            .execute(&self.pool)
            .await?;
        client_from_row(row).map(Some)
    }

    pub async fn revoke_api_client(&self, id: &str) -> anyhow::Result<bool> {
        Ok(sqlx::query(
            "UPDATE api_clients SET revoked_at = datetime('now')
             WHERE id = ? AND revoked_at IS NULL",
        )
        .bind(id)
        .execute(&self.pool)
        .await?
        .rows_affected()
            == 1)
    }

    #[cfg(test)]
    async fn stored_client_hash(&self, id: &str) -> anyhow::Result<String> {
        Ok(
            sqlx::query("SELECT token_hash FROM api_clients WHERE id = ?")
                .bind(id)
                .fetch_one(&self.pool)
                .await?
                .try_get("token_hash")?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn database() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("test.sqlite"))
            .await
            .unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn pairing_issues_a_hashed_scoped_credential_once() {
        let (_dir, db) = database().await;
        let created = db
            .create_pairing_request(
                "Dylan's laptop",
                &[
                    ClientScope::Agent,
                    ClientScope::Inference,
                    ClientScope::Agent,
                ],
                60,
            )
            .await
            .unwrap();
        assert_eq!(
            created.request.scopes,
            vec![ClientScope::Inference, ClientScope::Agent]
        );

        let issued = db
            .claim_pairing_request(&created.request.id, &created.code)
            .await
            .unwrap();
        assert_ne!(
            db.stored_client_hash(&issued.client.id).await.unwrap(),
            issued.api_key
        );
        assert!(
            db.authenticate_api_client(&issued.api_key)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            db.claim_pairing_request(&created.request.id, &created.code)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn wrong_pairing_codes_exhaust_the_attempt_budget() {
        let (_dir, db) = database().await;
        let created = db
            .create_pairing_request("Tablet", &[ClientScope::Inference], 60)
            .await
            .unwrap();
        for _ in 0..MAX_PAIRING_ATTEMPTS {
            assert!(
                db.claim_pairing_request(&created.request.id, "wrong")
                    .await
                    .is_err()
            );
        }
        assert!(
            db.claim_pairing_request(&created.request.id, &created.code)
                .await
                .is_err()
        );
        assert_eq!(
            db.pairing_request(&created.request.id)
                .await
                .unwrap()
                .unwrap()
                .attempts,
            MAX_PAIRING_ATTEMPTS
        );
    }

    #[tokio::test]
    async fn concurrent_claims_issue_exactly_one_credential() {
        let (_dir, db) = database().await;
        let created = db
            .create_pairing_request("One client", &[ClientScope::Inference], 60)
            .await
            .unwrap();
        let first_db = db.clone();
        let first_id = created.request.id.clone();
        let first_code = created.code.clone();
        let second_db = db.clone();
        let second_id = created.request.id.clone();
        let second_code = created.code.clone();
        let (first, second) = tokio::join!(
            async move { first_db.claim_pairing_request(&first_id, &first_code).await },
            async move {
                second_db
                    .claim_pairing_request(&second_id, &second_code)
                    .await
            }
        );
        assert_ne!(
            first.is_ok(),
            second.is_ok(),
            "exactly one concurrent claimant must receive a credential"
        );
        assert_eq!(db.list_api_clients().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn expired_and_cancelled_pairings_cannot_be_claimed() {
        let (_dir, db) = database().await;
        let expired = db
            .create_pairing_request("Old client", &[ClientScope::Inference], 1)
            .await
            .unwrap();
        sqlx::query("UPDATE api_pairing_requests SET expires_at = 0 WHERE id = ?")
            .bind(&expired.request.id)
            .execute(&db.pool)
            .await
            .unwrap();
        assert!(
            db.claim_pairing_request(&expired.request.id, &expired.code)
                .await
                .is_err()
        );

        let cancelled = db
            .create_pairing_request("Cancelled", &[ClientScope::Inference], 60)
            .await
            .unwrap();
        assert!(
            db.cancel_pairing_request(&cancelled.request.id)
                .await
                .unwrap()
        );
        assert!(
            db.claim_pairing_request(&cancelled.request.id, &cancelled.code)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn revocation_is_immediate_and_survives_reopen() {
        let (dir, db) = database().await;
        let created = db
            .create_pairing_request("CLI", &[ClientScope::Management], 60)
            .await
            .unwrap();
        let issued = db
            .claim_pairing_request(&created.request.id, &created.code)
            .await
            .unwrap();
        assert!(db.revoke_api_client(&issued.client.id).await.unwrap());
        assert!(
            db.authenticate_api_client(&issued.api_key)
                .await
                .unwrap()
                .is_none()
        );
        drop(db);
        let reopened = Database::open(&dir.path().join("test.sqlite"))
            .await
            .unwrap();
        let client = reopened
            .api_client(&issued.client.id)
            .await
            .unwrap()
            .unwrap();
        assert!(client.revoked_at.is_some());
        assert!(
            reopened
                .authenticate_api_client(&issued.api_key)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn daemon_identity_is_stable_but_its_display_name_is_configurable() {
        let (dir, db) = database().await;
        let first = db.daemon_identity(Some("Studio Mac")).await.unwrap();
        drop(db);
        let reopened = Database::open(&dir.path().join("test.sqlite"))
            .await
            .unwrap();
        let same = reopened.daemon_identity(None).await.unwrap();
        assert_eq!(same.instance_id, first.instance_id);
        assert_eq!(same.display_name, "Studio Mac");
        let renamed = reopened
            .daemon_identity(Some("Basement GPU"))
            .await
            .unwrap();
        assert_eq!(renamed.instance_id, first.instance_id);
        assert_eq!(renamed.display_name, "Basement GPU");
    }
}
