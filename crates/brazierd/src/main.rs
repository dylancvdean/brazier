use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};

use anyhow::Context;
use brazierd::{AppState, api, db::Database, engine::MockEngine};
use clap::Parser;
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
    #[arg(long, default_value_t = 0)]
    port: u16,
    #[arg(long, env = "BRAZIER_API_KEY")]
    api_key: Option<String>,
    #[arg(long, conflicts_with = "api_key")]
    no_auth: bool,
    #[arg(long, requires = "no_auth")]
    allow_insecure_remote: bool,
    #[arg(long, env = "BRAZIER_DATA_DIR")]
    data_dir: Option<PathBuf>,
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
        args.host.is_loopback() || !args.no_auth || args.allow_insecure_remote,
        "keyless access on a non-loopback interface requires --allow-insecure-remote"
    );
    if !args.host.is_loopback() && args.no_auth {
        tracing::warn!("daemon is exposed beyond loopback without authentication");
    }

    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);
    let db = Database::open(&data_dir.join("brazier.sqlite")).await?;
    let api_key = if args.no_auth {
        None
    } else {
        Some(
            args.api_key
                .unwrap_or_else(|| format!("brazier_{}", Uuid::new_v4().simple())),
        )
    };
    let state = AppState {
        db,
        engine: Arc::new(MockEngine),
        api_key: api_key.clone(),
        http: reqwest::Client::builder()
            .user_agent(format!("brazier/{}", env!("CARGO_PKG_VERSION")))
            .build()?,
    };

    let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port))
        .await
        .context("bind daemon listener")?;
    let address = listener.local_addr()?;
    println!(
        "BRAZIER_READY {}",
        serde_json::to_string(&serde_json::json!({
            "address": format!("http://{address}"),
            "api_key": api_key
        }))?
    );
    tracing::info!(%address, data_dir = %data_dir.display(), "brazier daemon ready");
    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serve daemon")
}

async fn shutdown_signal() {
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
}
