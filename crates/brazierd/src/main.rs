use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};

use anyhow::Context;
use brazierd::{
    AppState, active_downloads::ActiveDownloads, api, builds, db::Database,
    download_queue::DownloadQueue, engine::Runtime,
};
use clap::Parser;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "brazierd",
    about = "Brazier local model and conversation daemon"
)]
struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    /// TCP port. Service mode defaults to 7614; the desktop-oriented default
    /// remains an ephemeral loopback port.
    #[arg(long)]
    port: Option<u16>,
    /// Run as a persistent daemon: retain a generated bearer credential and
    /// publish a stable, non-secret readiness descriptor.
    #[arg(long)]
    service: bool,
    /// Override where service readiness metadata is written. It contains no
    /// credential and is owner-only on Unix.
    #[arg(long, requires = "service")]
    ready_file: Option<PathBuf>,
    /// Bearer credential accepted by the API. Repeatable: any listed key
    /// authenticates, so distinct clients can each be revoked independently.
    /// A single key is still the common case.
    #[arg(long, env = "BRAZIER_API_KEY")]
    api_key: Vec<String>,
    /// Whether API requests may load a non-resident local model on demand.
    #[arg(long)]
    jit_loading: Option<bool>,
    #[arg(long, conflicts_with = "api_key")]
    no_auth: bool,
    #[arg(long, requires = "no_auth")]
    allow_insecure_remote: bool,
    #[arg(long, env = "BRAZIER_DATA_DIR")]
    data_dir: Option<PathBuf>,
    /// Extra browser origin allowed to call the API. Repeatable, or
    /// comma-separated in `BRAZIER_ALLOWED_ORIGINS`. The packaged UI and the dev
    /// server are always allowed; a wildcard is not accepted.
    #[arg(long = "allowed-origin", env = "BRAZIER_ALLOWED_ORIGINS")]
    allowed_origins: Vec<String>,
}

fn default_data_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("BRAZIER_DATA_DIR") {
        return PathBuf::from(path);
    }
    #[cfg(target_os = "windows")]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(path).join("Brazier");
    }
    #[cfg(target_os = "macos")]
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join("Library/Application Support/Brazier");
    }
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(path).join("brazier");
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join(".local/share/brazier");
    }
    PathBuf::from(".brazier")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("brazierd=info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    anyhow::ensure!(
        args.host.is_loopback() || !args.no_auth,
        "keyless access (--no-auth) is not permitted on a non-loopback interface; bind to loopback or supply an API key"
    );

    // Parsed before anything is opened: a typo in an origin should fail at the
    // command line, not after a database and a model cache are warm.
    let allowed_origins = api::parse_origins(&args.allowed_origins)?;
    if !args.allowed_origins.is_empty() {
        tracing::info!(
            origins = ?args.allowed_origins,
            "allowing extra browser origins to call the API"
        );
    }

    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);
    tokio::fs::create_dir_all(&data_dir)
        .await
        .context("create data directory")?;
    // Release lookups persist across restarts so Manage → Runtimes never waits
    // on GitHub just to show what is installed.
    brazierd::github_releases::set_cache_dir(data_dir.join("state"));
    let db = Database::open(&data_dir.join("brazier.sqlite")).await?;
    if let Some(jit_loading) = args.jit_loading {
        let mut settings = brazierd::runtime_settings::load(&data_dir);
        settings.jit_loading = jit_loading;
        brazierd::runtime_settings::save(&data_dir, &settings).await?;
    }
    // Downloads do not survive a restart, but their partial files do; mark any
    // that were mid-flight as paused so they can be resumed rather than
    // appearing to still be running.
    db.interrupt_running_download_jobs().await?;
    let api_keys = if args.no_auth {
        Vec::new()
    } else if args.service {
        // Service mode keeps a single durable credential keyed to the data
        // directory; an explicitly supplied key remains its source of truth.
        vec![brazierd::service::service_api_key(
            &data_dir,
            args.api_key.first().cloned(),
        )?]
    } else {
        let mut keys = args.api_key.clone();
        if keys.is_empty() {
            keys.push(format!("brazier_{}", Uuid::new_v4().simple()));
        }
        keys
    };
    let http = reqwest::Client::builder()
        .user_agent(format!("brazier/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let runtime = Runtime::new(data_dir.clone(), http.clone());
    let active_downloads = Arc::new(ActiveDownloads::new());
    let download_queue = DownloadQueue::spawn(
        http.clone(),
        data_dir.clone(),
        db.clone(),
        Arc::clone(&active_downloads),
        Arc::clone(&runtime),
    );
    let computer_broker =
        brazierd::computer_exec::ComputerBroker::open(data_dir.join("computer_sessions.json"))
            .await?;
    let computer_preference = db
        .application_preference(brazierd::api::COMPUTER_PREFERENCE_KEY)
        .await?;
    let action_settle_delay_ms = computer_preference
        .as_ref()
        .and_then(|value| value["action_settle_delay_ms"].as_u64())
        .unwrap_or(brazierd::computer_exec::DEFAULT_ACTION_SETTLE_DELAY_MS);
    computer_broker.set_action_settle_delay_ms(action_settle_delay_ms);
    let state = AppState {
        db,
        runtime: Arc::clone(&runtime),
        api_keys: api_keys.clone(),
        http,
        data_dir: data_dir.clone(),
        active_builds: Arc::new(builds::ActiveBuilds::new()),
        build_slots: Arc::new(tokio::sync::Semaphore::new(1)),
        active_downloads,
        download_queue,
        runtimes_cache: Arc::new(Mutex::new(None)),
        agent_broker: Arc::new(brazierd::agent_exec::AgentBroker::new()),
        computer_broker: Arc::new(computer_broker),
    };
    // Gated-model access is watched on a daemon timer (every five minutes, up
    // to twenty-four checks) rather than on the settings page poll, so a grant
    // is noticed even when nobody is looking at the queue.
    let hf_access_shutdown = Arc::new(tokio::sync::Notify::new());
    let hf_access_checker =
        brazierd::api::spawn_hf_access_checker(state.clone(), Arc::clone(&hf_access_shutdown));

    let port = args.port.unwrap_or(if args.service {
        brazierd::service::DEFAULT_SERVICE_PORT
    } else {
        0
    });
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, port))
        .await
        .context("bind daemon listener")?;
    let address = listener.local_addr()?;
    let address_url = format!("http://{address}");
    if args.service {
        let ready_path = args
            .ready_file
            .unwrap_or_else(|| brazierd::service::ready_path(&data_dir));
        brazierd::service::write_ready_descriptor(&ready_path, &address_url)?;
        tracing::info!(ready_file = %ready_path.display(), "wrote service readiness descriptor");
    }
    if args.service {
        // Service mode never emits the bearer to stdout: the readiness
        // descriptor on disk deliberately omits it, and the daemon's stdout is
        // often captured by a journal or pipe that should not see credentials.
        eprintln!(
            "BRAZIER_READY {}",
            serde_json::to_string(&serde_json::json!({ "address": address_url }))?
        );
    } else {
        // Non-service mode runs bound to loopback; the desktop launcher reads
        // the first key from stdout to authenticate its loopback connection.
        println!(
            "BRAZIER_READY {}",
            serde_json::to_string(&serde_json::json!({
                "address": address_url,
                // The desktop's internal connection uses the first key; extra keys
                // are passed for its configured clients, not reported here.
                "api_key": api_keys.first()
            }))?
        );
    }
    tracing::info!(%address, data_dir = %data_dir.display(), "brazier daemon ready");
    axum::serve(
        listener,
        api::router_with_origins(state, allowed_origins)
            .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(
        runtime,
        hf_access_shutdown,
        hf_access_checker,
    ))
    .await
    .context("serve daemon")
}

async fn shutdown_signal(
    runtime: Arc<Runtime>,
    hf_access_shutdown: Arc<tokio::sync::Notify>,
    hf_access_checker: tokio::task::JoinHandle<()>,
) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    hf_access_shutdown.notify_one();
    hf_access_checker.abort();
    if let Err(error) = hf_access_checker.await
        && error.is_panic()
    {
        tracing::warn!(%error, "HF access checker task panicked during shutdown");
    }
    runtime.shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_mode_has_a_stable_default_but_normal_launches_do_not() {
        let service = Args::try_parse_from(["brazierd", "--service"]).unwrap();
        assert!(service.service);
        assert_eq!(service.port, None);
        let normal = Args::try_parse_from(["brazierd"]).unwrap();
        assert!(!normal.service);
        assert_eq!(normal.port, None);
    }

    #[test]
    fn custom_ready_file_requires_service_mode() {
        assert!(Args::try_parse_from(["brazierd", "--ready-file", "/tmp/ready.json"]).is_err());
        let service =
            Args::try_parse_from(["brazierd", "--service", "--ready-file", "/tmp/ready.json"])
                .unwrap();
        assert_eq!(
            service.ready_file.unwrap(),
            PathBuf::from("/tmp/ready.json")
        );
    }
}
