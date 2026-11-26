//! HTTPS for non-loopback listeners.
//!
//! The daemon keeps one self-signed certificate per data directory. Clients
//! never rely on a CA for it: pairing invites carry the certificate's SHA-256
//! fingerprint, and paired clients pin that exact certificate. Each accepted
//! connection is sniffed so the same port also serves plaintext HTTP to
//! loopback callers (the desktop app) and, only when explicitly allowed, to
//! other devices.

use std::{
    io,
    net::SocketAddr,
    path::Path,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use anyhow::Context as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncRead, AsyncWrite, AsyncWriteExt as _, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{Semaphore, mpsc},
};
use tokio_rustls::{TlsAcceptor, server::TlsStream};

const CERTIFICATE_FILE: &str = "daemon-certificate.der";
const KEY_FILE: &str = "daemon-key.der";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const TLS_HANDSHAKE_RECORD: u8 = 0x16;
const MAX_PENDING_HANDSHAKES: usize = 256;

pub struct DaemonCertificate {
    pub sha256: String,
    pub acceptor: TlsAcceptor,
}

/// Load the data directory's certificate, creating it on first use.
pub fn load_or_create(data_dir: &Path) -> anyhow::Result<DaemonCertificate> {
    let directory = data_dir.join("tls");
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create {}", directory.display()))?;
    let certificate_path = directory.join(CERTIFICATE_FILE);
    let key_path = directory.join(KEY_FILE);
    let (certificate, key) = match (std::fs::read(&certificate_path), std::fs::read(&key_path)) {
        (Ok(certificate), Ok(key)) => (certificate, key),
        _ => {
            let _ = std::fs::remove_file(&certificate_path);
            let _ = std::fs::remove_file(&key_path);
            let (certificate, key) = generate()?;
            crate::service::write_new_private(&key_path, &key)?;
            std::fs::write(&certificate_path, &certificate)
                .with_context(|| format!("write {}", certificate_path.display()))?;
            (certificate, key)
        }
    };
    let sha256 = hex::encode(Sha256::digest(&certificate));
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("configure TLS protocol versions")?
        .with_no_client_auth()
        .with_single_cert(
            vec![CertificateDer::from(certificate)],
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key)),
        )
        .context("load daemon TLS certificate")?;
    Ok(DaemonCertificate {
        sha256,
        acceptor: TlsAcceptor::from(Arc::new(config)),
    })
}

fn generate() -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let key = rcgen::KeyPair::generate().context("generate daemon TLS key")?;
    let mut params = rcgen::CertificateParams::new(vec!["brazier.local".to_owned()])
        .context("build daemon TLS certificate")?;
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "Brazier daemon");
    params.not_before = rcgen::date_time_ymd(2024, 1, 1);
    params.not_after = rcgen::date_time_ymd(2124, 1, 1);
    let certificate = params
        .self_signed(&key)
        .context("sign daemon TLS certificate")?;
    Ok((certificate.der().to_vec(), key.serialize_der()))
}

pub enum DaemonStream {
    Plain(TcpStream),
    Tls(Box<TlsStream<TcpStream>>),
}

impl AsyncRead for DaemonStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_read(cx, buf),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for DaemonStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write(cx, buf),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_flush(cx),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(cx),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_shutdown(cx),
        }
    }
}

/// Accepts TCP connections and completes TLS handshakes off the accept path,
/// so one slow or silent client cannot stall other connections.
pub struct DaemonListener {
    local_addr: SocketAddr,
    accepted: mpsc::Receiver<(DaemonStream, SocketAddr)>,
}

impl DaemonListener {
    /// Without an acceptor every connection is plaintext; callers only do that
    /// on a loopback listener. With one, plaintext is limited to loopback peers
    /// unless `allow_remote_plaintext` is set.
    pub fn new(
        listener: TcpListener,
        acceptor: Option<TlsAcceptor>,
        allow_remote_plaintext: bool,
    ) -> io::Result<Self> {
        let local_addr = listener.local_addr()?;
        let (sender, accepted) = mpsc::channel(64);
        let pending = Arc::new(Semaphore::new(MAX_PENDING_HANDSHAKES));
        tokio::spawn(async move {
            loop {
                // Bound unauthenticated connections that are still being
                // classified so silent peers cannot exhaust tasks and sockets.
                let Ok(permit) = pending.clone().acquire_owned().await else {
                    return;
                };
                let (stream, peer) = match listener.accept().await {
                    Ok(connection) => connection,
                    Err(error) => {
                        tracing::debug!(%error, "accept failed");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let Some(acceptor) = acceptor.clone() else {
                    if sender
                        .send((DaemonStream::Plain(stream), peer))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    continue;
                };
                let sender = sender.clone();
                tokio::spawn(async move {
                    let plaintext_allowed =
                        allow_remote_plaintext || peer.ip().to_canonical().is_loopback();
                    let classified = classify(stream, acceptor, plaintext_allowed).await;
                    drop(permit);
                    if let Some(stream) = classified {
                        let _ = sender.send((stream, peer)).await;
                    }
                });
            }
        });
        Ok(Self {
            local_addr,
            accepted,
        })
    }
}

async fn classify(
    stream: TcpStream,
    acceptor: TlsAcceptor,
    plaintext_allowed: bool,
) -> Option<DaemonStream> {
    let mut first = [0_u8; 1];
    let peeked = tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.peek(&mut first)).await;
    if !matches!(peeked, Ok(Ok(1))) {
        return None;
    }
    if first[0] == TLS_HANDSHAKE_RECORD {
        return match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
            Ok(Ok(stream)) => Some(DaemonStream::Tls(Box::new(stream))),
            Ok(Err(error)) => {
                tracing::debug!(%error, "TLS handshake failed");
                None
            }
            Err(_) => None,
        };
    }
    if plaintext_allowed {
        return Some(DaemonStream::Plain(stream));
    }
    let mut stream = stream;
    let body = "This Brazier daemon requires HTTPS from other devices.\n";
    let response = format!(
        "HTTP/1.1 426 Upgrade Required\r\nUpgrade: TLS/1.2, HTTP/1.1\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.write_all(response.as_bytes())).await;
    let _ = stream.shutdown().await;
    None
}

impl axum::serve::Listener for DaemonListener {
    type Io = DaemonStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.accepted.recv().await {
            Some(connection) => connection,
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_is_created_once_and_reused() {
        let directory = tempfile::tempdir().unwrap();
        let first = load_or_create(directory.path()).unwrap();
        let second = load_or_create(directory.path()).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.sha256.len(), 64);
        assert!(
            first
                .sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn a_damaged_certificate_is_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let first = load_or_create(directory.path()).unwrap();
        std::fs::remove_file(directory.path().join("tls").join(KEY_FILE)).unwrap();
        let second = load_or_create(directory.path()).unwrap();
        assert_ne!(first.sha256, second.sha256);
    }
}
