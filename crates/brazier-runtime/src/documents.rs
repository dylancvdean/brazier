//! Text and image extraction from documents (PDF, RTF, DOC, DOCX).
//!
//! One core backs both the chat `doc_read` tool and the agent's document
//! reading. PDFs go through Poppler (`pdftotext`, `pdfinfo`, `pdftoppm`) so a
//! model can read a page range instead of a whole book; RTF and DOCX are
//! decoded in-process; legacy `.doc` needs an external converter. Scanned PDFs
//! have no text layer at all, so pages can be rendered to images with
//! `pdftoppm` for a vision-capable model to read.

use std::{path::Path, process::Stdio, time::Duration};

use anyhow::Context;
use serde_json::Value;
use tokio::process::Command;

use crate::{blob_store, toolchain_hints::resolve_command};

/// Longest text a `doc_read` call returns. Larger output tells the caller to
/// narrow its page or line range.
pub const MAX_EXTRACTION_CHARS: usize = 24_000;
/// Character budget for documents inlined into a chat message, matching the
/// plain-text attachment limit.
pub const MAX_INLINE_CHARS: usize = 1_000_000;
/// Default PDF page window when the caller names no range.
pub const DEFAULT_PAGE_COUNT: u32 = 3;
/// Widest PDF window a text call accepts, so a slip does not flood the context.
pub const MAX_TEXT_PAGES: u32 = 25;
/// Most pages one render call produces; page images are far dearer than text.
pub const MAX_RENDER_PAGES: u32 = 4;
const RENDER_DPI: u32 = 150;

const EXTRACT_TIMEOUT: Duration = Duration::from_secs(120);
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
const POPPLER_TOOLS: [&str; 3] = ["pdftotext", "pdfinfo", "pdftoppm"];
const POPPLER_INSTALL_HINT: &str = "Install Poppler (for example `brew install poppler`, `apt install poppler-utils`, or \
     `winget install oschwartz10612.Poppler`) and restart Brazier.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentKind {
    Pdf,
    Rtf,
    Doc,
    Docx,
}

impl DocumentKind {
    /// Short name used in attachment notices and tool output.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pdf => "PDF",
            Self::Rtf => "RTF",
            Self::Doc => "DOC",
            Self::Docx => "DOCX",
        }
    }
}

/// Kind from an attachment's declared mime type, falling back to its name.
pub fn kind_for_mime(mime_type: &str, name: &str) -> Option<DocumentKind> {
    match mime_type {
        "application/pdf" => Some(DocumentKind::Pdf),
        "application/rtf" | "text/rtf" => Some(DocumentKind::Rtf),
        "application/msword" => Some(DocumentKind::Doc),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(DocumentKind::Docx)
        }
        _ => kind_for_name(name),
    }
}

/// Kind from a file name's extension.
pub fn kind_for_name(name: &str) -> Option<DocumentKind> {
    let extension = name.rsplit('.').next()?.to_ascii_lowercase();
    match extension.as_str() {
        "pdf" => Some(DocumentKind::Pdf),
        "rtf" => Some(DocumentKind::Rtf),
        "doc" => Some(DocumentKind::Doc),
        "docx" => Some(DocumentKind::Docx),
        _ => None,
    }
}

/// Whether the mime type or name names one of the formats `doc_read` handles
/// (as opposed to plain text, which is inlined directly).
pub fn is_supported_document(mime_type: &str, name: &str) -> bool {
    kind_for_mime(mime_type, name).is_some()
}

/// Return the Poppler utilities that this PDF pipeline cannot currently find.
pub fn missing_poppler_tools() -> Vec<&'static str> {
    POPPLER_TOOLS
        .iter()
        .copied()
        .filter(|name| resolve_command(name).is_none())
        .collect()
}

/// Explain a missing Poppler installation in terms a desktop user can act on.
pub fn poppler_missing_message() -> Option<String> {
    poppler_missing_message_for(&missing_poppler_tools())
}

fn poppler_missing_message_for(missing: &[&str]) -> Option<String> {
    if missing.is_empty() {
        return None;
    }
    let utilities = missing
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let subject = if missing.len() == 1 {
        "Poppler utility"
    } else {
        "Poppler utilities"
    };
    Some(format!(
        "PDF support is unavailable because {subject} {utilities} are missing. \
         {POPPLER_INSTALL_HINT}"
    ))
}

/// Fail before a PDF attachment reaches model generation when its runtime is
/// unavailable, instead of deferring the error to a hidden `doc_read` call.
pub fn ensure_poppler_available() -> anyhow::Result<()> {
    if let Some(message) = poppler_missing_message() {
        anyhow::bail!("{message}");
    }
    Ok(())
}

/// What came back from a text extraction.
#[derive(Debug, Clone)]
pub struct Extraction {
    pub text: String,
    /// PDF page count when Poppler could report it.
    pub page_count: Option<u32>,
    /// 1-based inclusive page window the text covers (PDFs only).
    pub pages: Option<(u32, u32)>,
    /// Total extracted lines before windowing (non-PDF formats).
    pub total_lines: Option<usize>,
    /// 1-based inclusive line window the text covers (non-PDF formats).
    pub lines: Option<(usize, usize)>,
    /// True when the text had to be cut to the caller's character budget.
    pub truncated: bool,
}

impl Extraction {
    /// Render the extraction as tool output, telling the caller exactly where
    /// it is in the document and how to read further.
    pub fn describe(self) -> String {
        let mut header = String::new();
        if let Some((start, end)) = self.pages {
            header = match self.page_count {
                Some(total) => format!("[pages {start}–{end} of {total}]"),
                None => format!("[pages {start}–{end}]"),
            };
        } else if let Some((start, end)) = self.lines {
            header = format!(
                "[lines {start}–{end} of {total}]",
                total = self.total_lines.unwrap_or(end)
            );
        }
        if self.truncated {
            if !header.is_empty() {
                header.push(' ');
            }
            header.push_str("[output truncated — narrow the range to read the rest]");
        }
        if header.is_empty() {
            self.text
        } else {
            format!("{header}\n{text}", text = self.text)
        }
    }
}

/// A page rendered to an image, stored as a blob for chat transcripts, with
/// its bytes kept for transports that inline base64 (the agent worker).
#[derive(Debug, Clone)]
pub struct RenderedPage {
    pub page: u32,
    pub sha256: String,
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

/// A stable display/save name for a page rendered from a document.
pub fn rendered_page_name(document_name: &str, page: u32) -> String {
    let stem = Path::new(document_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("document");
    let safe_stem: String = stem
        .chars()
        .map(|character| {
            if character == '/' || character == '\\' || character.is_control() {
                '_'
            } else {
                character
            }
        })
        .collect();
    let stem = if safe_stem.is_empty() {
        "document"
    } else {
        safe_stem.as_str()
    };
    format!("{stem}-page-{page}.png")
}

impl RenderedPage {
    /// Base64 of the page image, without a data-URL prefix.
    pub fn base64_data(&self) -> String {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(&self.bytes)
    }
}

fn poppler_tool(name: &str) -> anyhow::Result<std::path::PathBuf> {
    resolve_command(name).with_context(|| {
        format!("Reading PDFs requires the `{name}` utility. {POPPLER_INSTALL_HINT}")
    })
}

async fn run_with_timeout(
    command: &mut Command,
    timeout: Duration,
    what: &str,
) -> anyhow::Result<std::process::Output> {
    command.kill_on_drop(true);
    let output = tokio::time::timeout(timeout, command.output())
        .await
        .with_context(|| format!("{what} timed out after {}s", timeout.as_secs()))?
        .with_context(|| format!("run {what}"))?;
    anyhow::ensure!(
        output.status.success(),
        "{what} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output)
}

/// PDF page count via `pdfinfo`. `Ok(None)` means `pdfinfo` was unavailable or
/// did not report a parseable count; callers treat the count as unknown.
pub async fn page_count(path: &Path) -> anyhow::Result<Option<u32>> {
    let Some(pdfinfo) = resolve_command("pdfinfo") else {
        return Ok(None);
    };
    let mut command = Command::new(pdfinfo);
    command.arg(path.display().to_string()).stdin(Stdio::null());
    let output = run_with_timeout(&mut command, PROBE_TIMEOUT, "pdfinfo").await?;
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text
        .lines()
        .find_map(|line| line.strip_prefix("Pages:")?.trim().parse::<u32>().ok()))
}

/// Extract text from a document, bounded to `max_chars`. PDFs honour a
/// 1-based inclusive `pages` window; other formats honour a 1-based inclusive
/// `lines` window after extraction. No window reads from the start.
pub async fn extract_text(
    path: &Path,
    kind: DocumentKind,
    pages: Option<(u32, u32)>,
    lines: Option<(usize, usize)>,
    max_chars: usize,
) -> anyhow::Result<Extraction> {
    match kind {
        DocumentKind::Pdf => extract_pdf_text(path, pages, max_chars).await,
        other => {
            anyhow::ensure!(
                pages.is_none(),
                "page ranges only apply to PDFs; use line ranges for {} documents",
                other.label()
            );
            let raw = match other {
                DocumentKind::Rtf => {
                    rtf_to_text(&tokio::fs::read(path).await.context("read RTF file")?)
                        .context("could not decode the RTF document")?
                }
                DocumentKind::Docx => {
                    docx_to_text(&tokio::fs::read(path).await.context("read DOCX file")?)
                        .context("could not decode the DOCX document")?
                }
                DocumentKind::Doc => extract_doc_text(path).await?,
                DocumentKind::Pdf => unreachable!(),
            };
            Ok(window_lines(&raw, lines, max_chars))
        }
    }
}

async fn extract_pdf_text(
    path: &Path,
    pages: Option<(u32, u32)>,
    max_chars: usize,
) -> anyhow::Result<Extraction> {
    let pdftotext = poppler_tool("pdftotext")?;
    let path_string = path.display().to_string();
    let mut command = Command::new(pdftotext);
    command.arg("-layout").stdin(Stdio::null());
    let window = pages.map(|(start, end)| (start.max(1), end.max(start.max(1))));
    if let Some((start, end)) = window {
        command
            .arg("-f")
            .arg(start.to_string())
            .arg("-l")
            .arg(end.to_string());
    }
    command.arg(&path_string).arg("-");
    let output = run_with_timeout(&mut command, EXTRACT_TIMEOUT, "pdftotext").await?;
    let text = String::from_utf8(output.stdout).context("PDF text was not UTF-8")?;
    let total = page_count(path).await.unwrap_or(None);

    match window {
        Some((start, _)) => {
            // pdftotext separates pages with a form feed; turn those into
            // explicit markers so the model can cite what it is reading.
            let segments: Vec<String> = text
                .split('\u{000C}')
                .map(str::trim)
                .filter(|segment| !segment.is_empty())
                .map(ToOwned::to_owned)
                .collect();
            let mut body = String::new();
            for (offset, segment) in segments.iter().enumerate() {
                if !body.is_empty() {
                    body.push_str("\n\n");
                }
                body.push_str(&format!("[Page {}]\n{segment}", start + offset as u32));
            }
            if body.is_empty() {
                body = "(no extractable text in that page range — the pages may be scanned \
                        images; call doc_read again with render_pages to see them)"
                    .to_owned();
            }
            let (body, truncated) = bound_chars(body, max_chars);
            let end = start + segments.len().saturating_sub(1) as u32;
            Ok(Extraction {
                text: body,
                page_count: total,
                pages: Some((start, end)),
                total_lines: None,
                lines: None,
                truncated,
            })
        }
        None => {
            let collapsed = text.replace('\u{000C}', "\n\n");
            let (body, truncated) = bound_chars(collapsed.trim().to_owned(), max_chars);
            Ok(Extraction {
                text: body,
                page_count: total,
                pages: None,
                total_lines: None,
                lines: None,
                truncated,
            })
        }
    }
}

/// Slice an extracted text by 1-based inclusive line numbers.
fn window_lines(raw: &str, lines: Option<(usize, usize)>, max_chars: usize) -> Extraction {
    let all: Vec<&str> = raw.lines().collect();
    let total = all.len();
    let start = lines.map(|(start, _)| start.max(1)).unwrap_or(1);
    let end = lines
        .map(|(_, end)| end.max(start))
        .unwrap_or(total)
        .min(total);
    let window = if start > total || start > end {
        String::new()
    } else {
        all[(start - 1)..end].join("\n")
    };
    let (text, truncated) = bound_chars(window, max_chars);
    Extraction {
        text,
        page_count: None,
        pages: None,
        total_lines: Some(total),
        lines: Some((start.min(total.max(1)), end)),
        truncated,
    }
}

/// Cut to a character budget on a char boundary, reporting whether it cut.
fn bound_chars(text: String, max: usize) -> (String, bool) {
    if text.len() <= max {
        return (text, false);
    }
    let mut cut = max;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    (text[..cut].trim_end().to_owned(), true)
}

/// Render a 1-based page window of a PDF to PNG blobs via pdftoppm.
pub async fn render_pages(
    data_dir: &Path,
    path: &Path,
    start_page: u32,
    count: u32,
) -> anyhow::Result<Vec<RenderedPage>> {
    let pdftoppm = poppler_tool("pdftoppm")?;
    let start = start_page.max(1);
    let end = start + count.clamp(1, MAX_RENDER_PAGES) - 1;
    let out_dir = data_dir.join("tmp").join("documents");
    tokio::fs::create_dir_all(&out_dir)
        .await
        .context("create document render directory")?;
    let prefix = out_dir.join(format!("render-{}", uuid::Uuid::new_v4()));
    let mut command = Command::new(pdftoppm);
    command
        .arg("-png")
        .arg("-r")
        .arg(RENDER_DPI.to_string())
        .arg("-f")
        .arg(start.to_string())
        .arg("-l")
        .arg(end.to_string())
        .arg(path.display().to_string())
        .arg(prefix.display().to_string())
        .stdin(Stdio::null());
    run_with_timeout(&mut command, EXTRACT_TIMEOUT, "pdftoppm").await?;

    let prefix_name = prefix
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned();
    let mut files = Vec::new();
    let mut entries = tokio::fs::read_dir(&out_dir)
        .await
        .context("read rendered pages")?;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix_name) && name.ends_with(".png") {
            files.push(entry.path());
        }
    }
    files.sort();

    let mut rendered = Vec::new();
    for (offset, file) in files.iter().enumerate() {
        let bytes = tokio::fs::read(file).await.context("read rendered page")?;
        let blob = blob_store::store_bytes(data_dir, &bytes, "image/png", Some("page.png"))
            .await
            .context("store rendered page")?;
        rendered.push(RenderedPage {
            page: start + offset as u32,
            sha256: blob.sha256,
            mime_type: "image/png".to_owned(),
            bytes,
        });
        let _ = tokio::fs::remove_file(file).await;
    }
    anyhow::ensure!(
        !rendered.is_empty(),
        "pdftoppm produced no pages for that range"
    );
    Ok(rendered)
}

/// Legacy `.doc` needs an external converter; try the usual ones in order.
async fn extract_doc_text(path: &Path) -> anyhow::Result<String> {
    let converters: &[(&str, &[&str])] = &[
        ("antiword", &[]),
        ("catdoc", &[]),
        ("textutil", &["-convert", "txt", "-stdout"]),
    ];
    for (name, extra) in converters {
        let Some(binary) = resolve_command(name) else {
            continue;
        };
        let mut command = Command::new(binary);
        command
            .args(extra.iter().map(|arg| arg.to_string()))
            .arg(path.display().to_string())
            .stdin(Stdio::null());
        let output = run_with_timeout(&mut command, PROBE_TIMEOUT, name).await?;
        return String::from_utf8(output.stdout).context("DOC text was not UTF-8");
    }
    anyhow::bail!(
        "Reading legacy .doc files requires a converter. Install antiword or catdoc (for \
         example `brew install antiword` or `apt install antiword`) and restart Brazier."
    )
}

// ---------------------------------------------------------------------------
// RTF
// ---------------------------------------------------------------------------

/// Decode RTF to plain text. Handles the constructs real documents use —
/// groups, control words with numeric arguments, `\'hh` bytes, `\uN` codepoints
/// (with `uc` fallback skipping), and the common symbol controls — and skips
/// destinations that carry no prose (font tables, stylesheets, pictures, …).
pub fn rtf_to_text(bytes: &[u8]) -> anyhow::Result<String> {
    anyhow::ensure!(bytes.starts_with(b"{\\rtf"), "file does not look like RTF");
    const IGNORABLE: &[&str] = &[
        "fonttbl",
        "colortbl",
        "stylesheet",
        "info",
        "pict",
        "object",
        "nonshppict",
        "xmlnstbl",
        "datastore",
        "themedata",
        "listtable",
        "listoverridetable",
        "revtbl",
        "rsidtbl",
        "generator",
        "datafield",
        "filetbl",
        "shp",
        "xe",
        "tc",
    ];
    #[derive(Clone, Copy)]
    struct GroupState {
        uc: usize,
        ignorable: bool,
    }
    let mut output = String::with_capacity(bytes.len() / 2);
    let mut stack: Vec<GroupState> = vec![GroupState {
        uc: 1,
        ignorable: false,
    }];
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'{' => {
                let current = *stack.last().expect("root group");
                stack.push(current);
                index += 1;
            }
            b'}' => {
                if stack.len() > 1 {
                    stack.pop();
                }
                index += 1;
            }
            b'\\' => {
                index += 1;
                if index >= bytes.len() {
                    break;
                }
                let next = bytes[index];
                // Escaped literal characters.
                if matches!(next, b'{' | b'}' | b'\\') {
                    if !stack.last().expect("group").ignorable {
                        output.push(next as char);
                    }
                    index += 1;
                    continue;
                }
                // Hex-escaped byte in the document's code page.
                if next == b'\'' {
                    let hex = bytes.get(index + 1..index + 3);
                    if let Some(hex) = hex
                        && let Ok(value) =
                            u8::from_str_radix(std::str::from_utf8(hex).unwrap_or(""), 16)
                    {
                        if !stack.last().expect("group").ignorable {
                            output.push(cp1252_char(value));
                        }
                        index += 3;
                    } else {
                        index += 1;
                    }
                    continue;
                }
                // Control symbol (non-alphabetic).
                if !next.is_ascii_alphabetic() {
                    match next {
                        b'*' => {
                            // Ignorable destination marker.
                            if let Some(group) = stack.last_mut() {
                                group.ignorable = true;
                            }
                        }
                        b'~' if !stack.last().expect("group").ignorable => output.push(' '),
                        b'_' if !stack.last().expect("group").ignorable => output.push('\u{2011}'),
                        _ => {}
                    }
                    index += 1;
                    continue;
                }
                // Control word: letters, optional signed number, optional
                // single-space delimiter.
                let word_start = index;
                while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
                    index += 1;
                }
                let word = std::str::from_utf8(&bytes[word_start..index]).unwrap_or("");
                let negative = index < bytes.len() && bytes[index] == b'-';
                let mut number: Option<i64> = None;
                if negative || (index < bytes.len() && bytes[index].is_ascii_digit()) {
                    let number_start = index;
                    if negative {
                        index += 1;
                    }
                    while index < bytes.len() && bytes[index].is_ascii_digit() {
                        index += 1;
                    }
                    number = std::str::from_utf8(&bytes[number_start..index])
                        .unwrap_or("")
                        .parse::<i64>()
                        .ok();
                }
                if index < bytes.len() && bytes[index] == b' ' {
                    index += 1;
                }
                let state = *stack.last().expect("group");
                match word {
                    "uc" => {
                        if let Some(group) = stack.last_mut() {
                            group.uc = number.unwrap_or(1).clamp(0, 8) as usize;
                        }
                    }
                    "u" => {
                        if let Some(code) = number {
                            let code = if code < 0 { code + 65_536 } else { code };
                            if !state.ignorable
                                && let Some(character) =
                                    char::from_u32(code.clamp(0, 0x10_FFFF) as u32)
                            {
                                output.push(character);
                            }
                            // Skip the \uc fallback bytes following \uN.
                            index = (index + state.uc).min(bytes.len());
                        }
                    }
                    _ => {
                        if IGNORABLE.contains(&word) {
                            if let Some(group) = stack.last_mut() {
                                group.ignorable = true;
                            }
                        } else if !state.ignorable {
                            match word {
                                "par" | "line" | "row" | "sect" | "page" => output.push('\n'),
                                "tab" | "emspace" | "enspace" => output.push('\t'),
                                "emdash" => output.push('\u{2014}'),
                                "endash" => output.push('\u{2013}'),
                                "bullet" => output.push('\u{2022}'),
                                "lquote" => output.push('\u{2018}'),
                                "rquote" => output.push('\u{2019}'),
                                "ldblquote" => output.push('\u{201C}'),
                                "rdblquote" => output.push('\u{201D}'),
                                _ => {}
                            }
                        }
                    }
                }
            }
            b'\r' | b'\n' => {
                index += 1;
            }
            _ => {
                if !stack.last().expect("group").ignorable {
                    output.push(byte as char);
                }
                index += 1;
            }
        }
    }
    Ok(collapse_blank_lines(&output))
}

/// Map a raw RTF hex byte to a character. Bytes below 0x80 are ASCII; the
/// 0x80–0x9F range follows Windows-1252, the de-facto RTF code page.
fn cp1252_char(byte: u8) -> char {
    const HIGH: [char; 32] = [
        '\u{20AC}', '\u{0081}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}',
        '\u{2021}', '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{008D}',
        '\u{017D}', '\u{008F}', '\u{0090}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}',
        '\u{2022}', '\u{2013}', '\u{2014}', '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}',
        '\u{0153}', '\u{009D}', '\u{017E}', '\u{0178}',
    ];
    match byte {
        0x80..=0x9F => HIGH[(byte - 0x80) as usize],
        _ => byte as char,
    }
}

// ---------------------------------------------------------------------------
// DOCX
// ---------------------------------------------------------------------------

/// Extract the prose of a DOCX document: the text runs of `word/document.xml`,
/// with paragraph, table, tab, and break structure preserved.
pub fn docx_to_text(bytes: &[u8]) -> anyhow::Result<String> {
    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader).context("DOCX is not a readable zip")?;
    let mut document = archive
        .by_name("word/document.xml")
        .context("DOCX has no word/document.xml")?;
    let mut xml = String::new();
    std::io::Read::read_to_string(&mut document, &mut xml).context("read word/document.xml")?;

    // Sequential scan: at every tag, either capture a text run or map a
    // structural marker, so everything stays in document order.
    let mut output = String::with_capacity(xml.len() / 4);
    let mut rest = xml.as_str();
    while let Some(open) = rest.find('<') {
        let tag = &rest[open..];
        if tag.starts_with("<w:t>") || tag.starts_with("<w:t ") {
            let content_start = tag.find('>').map(|at| at + 1).unwrap_or(1);
            let close = tag.find("</w:t>").unwrap_or(tag.len());
            output.push_str(&xml_unescape(
                &tag[content_start..close.max(content_start).min(tag.len())],
            ));
            rest = &tag[close + 1..];
            continue;
        }
        for (marker, replacement) in [
            ("<w:tab/>", "\t"),
            ("<w:br/>", "\n"),
            ("</w:p>", "\n"),
            ("</w:tr>", "\n"),
            ("</w:tc>", "\t"),
        ] {
            if tag.starts_with(marker) {
                output.push_str(replacement);
                break;
            }
        }
        let tag_end = tag.find('>').map(|at| at + 1).unwrap_or(tag.len());
        rest = &tag[tag_end..];
    }
    Ok(collapse_blank_lines(&output))
}

/// One left-to-right entity pass: `&amp;lt;` becomes `&lt;`, never `<`.
fn xml_unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let after = &rest[at..];
        if let Some(decoded) = after
            .strip_prefix("&lt;")
            .map(|_| '<')
            .or_else(|| after.strip_prefix("&gt;").map(|_| '>'))
            .or_else(|| after.strip_prefix("&quot;").map(|_| '"'))
            .or_else(|| after.strip_prefix("&apos;").map(|_| '\''))
            .or_else(|| after.strip_prefix("&amp;").map(|_| '&'))
        {
            out.push(decoded);
            rest = &after[after.find(';').map(|end| end + 1).unwrap_or(after.len())..];
            continue;
        }
        if after.starts_with("&#")
            && let Some(end) = after.find(';')
        {
            let digits = &after[2..end];
            let code = digits
                .strip_prefix('x')
                .or_else(|| digits.strip_prefix('X'))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| digits.parse::<u32>().ok());
            if let Some(character) = code.and_then(char::from_u32) {
                out.push(character);
                rest = &after[end + 1..];
                continue;
            }
        }
        out.push('&');
        rest = &after[1..];
    }
    out.push_str(rest);
    out
}

/// Normalise the blank-line runs that paragraph markers tend to produce.
fn collapse_blank_lines(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut blank = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if !blank && !collapsed.is_empty() {
                collapsed.push('\n');
            }
            blank = true;
        } else {
            collapsed.push_str(trimmed);
            collapsed.push('\n');
            blank = false;
        }
    }
    collapsed.trim_end().to_owned()
}

/// How many hex characters of a blob id the model is shown / asked to echo.
///
/// Full SHA-256 digests are 64 characters; models truncate or invent them.
/// A short unique prefix from the attachment notice is enough for `doc_read`
/// to resolve against the conversation's document list.
pub const DOCUMENT_ID_PREFIX_LEN: usize = 12;

/// Short id shown in attachment notices for `doc_read`.
pub fn short_document_id(sha256: &str) -> &str {
    let end = DOCUMENT_ID_PREFIX_LEN.min(sha256.len());
    &sha256[..end]
}

/// Attachment notice: what the model needs to call `doc_read` for this
/// document — its name, format, page count when known, and a short blob id.
pub fn attachment_notice(
    name: &str,
    mime_type: &str,
    sha256: &str,
    page_count: Option<u32>,
) -> Value {
    let kind = kind_for_mime(mime_type, name);
    let format = kind.map(DocumentKind::label).unwrap_or("document");
    let length = match (kind, page_count) {
        (Some(DocumentKind::Pdf), Some(count)) => format!(", {count} pages"),
        _ => String::new(),
    };
    let how = match kind {
        Some(DocumentKind::Pdf) => {
            "pick a page range, or set render_pages when the document is a scan \
             or its layout matters"
        }
        _ => "pick a line range with start_line and end_line",
    };
    let document_id = short_document_id(sha256);
    serde_json::json!({
        "type": "text",
        "text": format!(
            "[Attached {format} document: {name}{length}. Its contents are not included here. \
             Use the doc_read tool with document \"{document_id}\" to read it — {how}.]"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_detected_by_mime_then_extension() {
        assert_eq!(
            kind_for_mime("application/pdf", "x"),
            Some(DocumentKind::Pdf)
        );
        assert_eq!(
            kind_for_mime("application/octet-stream", "report.docx"),
            Some(DocumentKind::Docx)
        );
        assert_eq!(kind_for_name("notes.rtf"), Some(DocumentKind::Rtf));
        assert_eq!(kind_for_name("legacy.doc"), Some(DocumentKind::Doc));
        assert_eq!(kind_for_name("plain.txt"), None);
        assert!(is_supported_document("application/octet-stream", "a.pdf"));
        assert!(!is_supported_document("text/plain", "a.txt"));
    }

    #[test]
    fn rtf_extracts_text_and_skips_metadata_groups() {
        let rtf = br"{\rtf1\ansi\deff0{\fonttbl{\f0 Arial;}}{\colortbl;\red255\green0\blue0;}\pard Hello {\b world}.\par Second line\'e9\u238?}
";
        let text = rtf_to_text(rtf).unwrap();
        assert!(text.contains("Hello world."), "{text}");
        assert!(text.contains("Second lineéî"), "{text}");
        assert!(!text.contains("Arial"), "{text}");
        assert!(!text.contains("red255"), "{text}");
    }

    #[test]
    fn rtf_handles_symbols_paragraphs_and_ignorable_destinations() {
        let rtf = br"{\rtf1 First\par\par Second\tab column\emdash end{\pict\pngblip DEADBEEF}\line Third\'80}";
        let text = rtf_to_text(rtf).unwrap();
        assert!(text.contains("First"), "{text}");
        assert!(text.contains("Second\tcolumn—end"), "{text}");
        assert!(text.contains("Third€"), "{text}");
        assert!(!text.contains("DEADBEEF"), "{text}");
    }

    #[test]
    fn rtf_rejects_non_rtf_bytes() {
        assert!(rtf_to_text(b"plain text").is_err());
    }

    fn docx_with_document_xml(document: &str) -> Vec<u8> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("word/document.xml", options).unwrap();
        std::io::Write::write_all(&mut writer, document.as_bytes()).unwrap();
        writer.finish().unwrap();
        buffer.into_inner()
    }

    #[test]
    fn docx_extracts_runs_paragraphs_and_entities() {
        let bytes = docx_with_document_xml(
            r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>Hello &amp; welcome</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve"> spaced </w:t></w:r><w:r><w:tab/><w:t>tabs</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>cell one</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>cell two</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body>
</w:document>"#,
        );
        let text = docx_to_text(&bytes).unwrap();
        assert!(text.contains("Hello & welcome"), "{text}");
        assert!(text.contains(" spaced "), "{text}");
        assert!(text.contains("tabs"), "{text}");
        assert!(text.contains("cell one"), "{text}");
        assert!(text.contains("cell two"), "{text}");
        // Paragraph and table structure survives in order.
        let (hello, _) = text.split_once("Hello & welcome").unwrap();
        assert!(hello.is_empty(), "{text}");
        assert!(text.find("cell one").unwrap() < text.find("cell two").unwrap());
    }

    #[test]
    fn docx_rejects_non_zip_bytes() {
        assert!(docx_to_text(b"not a zip").is_err());
    }

    #[test]
    fn xml_entities_decode_in_a_single_pass() {
        assert_eq!(xml_unescape("&amp;lt;"), "&lt;");
        assert_eq!(xml_unescape("a &lt; b &#38; c"), "a < b & c");
        assert_eq!(xml_unescape("&#x2014;"), "—");
        assert_eq!(xml_unescape("not an &entity;"), "not an &entity;");
    }

    #[test]
    fn line_windows_slice_and_report_totals() {
        let raw = (1..=100)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let extraction = window_lines(&raw, Some((10, 12)), MAX_EXTRACTION_CHARS);
        assert_eq!(extraction.text, "line 10\nline 11\nline 12");
        assert_eq!(extraction.total_lines, Some(100));
        assert_eq!(extraction.lines, Some((10, 12)));
        assert!(!extraction.truncated);

        let described = window_lines(&raw, Some((95, 200)), MAX_EXTRACTION_CHARS).describe();
        assert!(described.contains("line 95"), "{described}");
        assert!(described.contains("lines 95–100 of 100"), "{described}");
    }

    #[test]
    fn bounding_cuts_on_char_boundaries() {
        let long = "é".repeat(MAX_EXTRACTION_CHARS * 2);
        let (cut, truncated) = bound_chars(long, MAX_EXTRACTION_CHARS);
        assert!(truncated);
        assert!(cut.len() <= MAX_EXTRACTION_CHARS);
    }

    #[test]
    fn notices_name_the_format_pages_and_short_blob_id() {
        let digest = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let notice = attachment_notice("scan.pdf", "application/pdf", digest, Some(12));
        let text = notice["text"].as_str().unwrap();
        assert!(text.contains("scan.pdf"), "{text}");
        assert!(text.contains("12 pages"), "{text}");
        assert!(text.contains(short_document_id(digest)), "{text}");
        assert!(
            !text.contains(digest),
            "full digest should not be shown: {text}"
        );
        assert!(text.contains("PDF"), "{text}");

        let docx = attachment_notice(
            "letter.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "def456",
            None,
        );
        let text = docx["text"].as_str().unwrap();
        assert!(text.contains("DOCX"), "{text}");
        assert!(!text.contains("pages"), "{text}");
    }

    #[test]
    fn short_document_id_takes_a_stable_prefix() {
        let digest = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert_eq!(short_document_id(digest), &digest[..DOCUMENT_ID_PREFIX_LEN]);
        assert_eq!(short_document_id("abc"), "abc");
    }

    #[test]
    fn rendered_page_names_keep_the_document_name() {
        assert_eq!(
            rendered_page_name("Quarterly report.pdf", 3),
            "Quarterly report-page-3.png"
        );
        assert_eq!(
            rendered_page_name("archive/report.pdf", 1),
            "report-page-1.png"
        );
    }

    #[test]
    fn missing_poppler_message_includes_install_guidance() {
        let message = poppler_missing_message_for(&["pdftotext"]).expect("missing utility");
        assert!(message.contains("pdftotext"), "{message}");
        assert!(message.contains("brew install poppler"), "{message}");
        assert!(message.contains("restart Brazier"), "{message}");
        assert!(poppler_missing_message_for(&[]).is_none());
    }
}
