//! Persisted acknowledgements of model license agreements.
//!
//! Some curated models are released under terms that bind the person running
//! them — MiniMax-H3, for example, is licensed worldwide except the EU, the
//! UK, the Republic of Korea, and the United States unless MiniMax grants a
//! separate license. The agreement is accepted in the interface before the
//! model can be installed, and this store keeps that acceptance durable so a
//! restart cannot silently undo the choice. Each record also remembers which
//! version of the terms the person actually saw.

use anyhow::Context;
use sqlx::{FromRow, Row};

use crate::db::Database;

/// One recorded acceptance of a license agreement.
#[derive(Debug, Clone, FromRow)]
pub struct LicenseConsent {
    pub license_id: String,
    pub license_version: String,
    pub accepted_at: String,
}

/// The recorded consent for a license, when one exists.
pub async fn consent(db: &Database, license_id: &str) -> anyhow::Result<Option<LicenseConsent>> {
    let row = sqlx::query(
        r#"SELECT license_id, license_version, accepted_at
           FROM license_consents WHERE license_id = ?"#,
    )
    .bind(license_id)
    .fetch_optional(&db.pool)
    .await
    .context("read license consent")?;
    Ok(row.map(|row| LicenseConsent {
        license_id: row.get("license_id"),
        license_version: row.get("license_version"),
        accepted_at: row.get("accepted_at"),
    }))
}

/// Whether a consent exists for the given terms version.
///
/// An earlier version does not satisfy a newer requirement: the person must
/// have seen the version currently in effect, because the terms can change.
pub async fn has_consent(
    db: &Database,
    license_id: &str,
    license_version: &str,
) -> anyhow::Result<bool> {
    Ok(consent(db, license_id)
        .await?
        .is_some_and(|record| record.license_version == license_version))
}

/// Record that the person accepted the given version of the license.
///
/// Re-accepting a newer version replaces the older record; the timestamp of
/// the latest acceptance is what is kept.
pub async fn record_consent(
    db: &Database,
    license_id: &str,
    license_version: &str,
) -> anyhow::Result<()> {
    sqlx::query(
        r#"INSERT INTO license_consents(license_id, license_version)
           VALUES(?, ?)
           ON CONFLICT(license_id) DO UPDATE SET
               license_version = excluded.license_version,
               accepted_at = datetime('now')"#,
    )
    .bind(license_id)
    .bind(license_version)
    .execute(&db.pool)
    .await
    .context("record license consent")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn database() -> (tempfile::TempDir, Database) {
        let dir = tempdir().expect("temp dir");
        let db = Database::open(&dir.path().join("brazier.sqlite"))
            .await
            .expect("open database");
        (dir, db)
    }

    #[tokio::test]
    async fn consent_is_durable_and_versioned() {
        let (_dir, db) = database().await;
        assert!(!has_consent(&db, "minimax-h3", "2026-08-02").await.unwrap());

        record_consent(&db, "minimax-h3", "2026-08-02")
            .await
            .unwrap();
        assert!(has_consent(&db, "minimax-h3", "2026-08-02").await.unwrap());
        assert!(!has_consent(&db, "minimax-h3", "2026-09-01").await.unwrap());

        let saved = consent(&db, "minimax-h3").await.unwrap().expect("consent");
        assert_eq!(saved.license_version, "2026-08-02");

        // Re-accepting a newer version replaces the earlier record, so a
        // re-termed agreement cannot be silently grandfathered.
        record_consent(&db, "minimax-h3", "2026-09-01")
            .await
            .unwrap();
        assert!(has_consent(&db, "minimax-h3", "2026-09-01").await.unwrap());
        assert!(!has_consent(&db, "minimax-h3", "2026-08-02").await.unwrap());

        // Different licenses are independent.
        assert!(!has_consent(&db, "some-other-license", "1").await.unwrap());
    }
}
