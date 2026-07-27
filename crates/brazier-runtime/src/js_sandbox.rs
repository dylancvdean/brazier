//! Bounded QuickJS sandbox for the bundled `run_javascript` tool.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Context;
use rquickjs::{Array, CatchResultExt, Context as JsContext, Ctx, FromJs, Runtime, Value};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Default code-size cap (also used by the catalog when no settings are loaded).
pub(crate) const DEFAULT_MAX_CODE_BYTES: usize = 16_384;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 8_000;
const DEFAULT_MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_MAX_STACK_BYTES: usize = 256 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);

const ROOMY_MAX_CODE_BYTES: usize = 64 * 1024;
const ROOMY_MAX_OUTPUT_CHARS: usize = 32_000;
const ROOMY_MEMORY_LIMIT_BYTES: usize = 32 * 1024 * 1024;
const ROOMY_MAX_STACK_BYTES: usize = 512 * 1024;
const ROOMY_TIMEOUT: Duration = Duration::from_secs(10);

const MAX_LOG_LINES: usize = 500;

/// Named presets for the chat JavaScript sandbox.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum JsSandboxProfile {
    /// Current resource limits; no `console` binding.
    Strict,
    /// Console capture on, default resource limits.
    #[default]
    Default,
    /// Higher time/memory/code caps with console capture.
    Roomy,
    /// Start from default limits; apply explicit overrides from settings.
    Custom,
}

/// Persisted knobs for the JavaScript sandbox (under runtime settings).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct JavascriptSandboxSettings {
    pub profile: JsSandboxProfile,
    /// When set, overrides the profile's console-capture default.
    pub capture_console: Option<bool>,
    pub timeout_ms: Option<u32>,
    pub memory_mb: Option<u32>,
    pub max_code_bytes: Option<u32>,
    pub max_output_chars: Option<u32>,
    pub max_stack_kb: Option<u32>,
}

impl JavascriptSandboxSettings {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(ms) = self.timeout_ms {
            anyhow::ensure!(
                (50..=60_000).contains(&ms),
                "javascript_sandbox.timeout_ms must be between 50 and 60000"
            );
        }
        if let Some(mb) = self.memory_mb {
            anyhow::ensure!(
                (1..=256).contains(&mb),
                "javascript_sandbox.memory_mb must be between 1 and 256"
            );
        }
        if let Some(bytes) = self.max_code_bytes {
            anyhow::ensure!(
                (256..=256 * 1024).contains(&bytes),
                "javascript_sandbox.max_code_bytes must be between 256 and 262144"
            );
        }
        if let Some(chars) = self.max_output_chars {
            anyhow::ensure!(
                (256..=100_000).contains(&chars),
                "javascript_sandbox.max_output_chars must be between 256 and 100000"
            );
        }
        if let Some(kb) = self.max_stack_kb {
            anyhow::ensure!(
                (64..=2048).contains(&kb),
                "javascript_sandbox.max_stack_kb must be between 64 and 2048"
            );
        }
        Ok(())
    }
}

/// Resolved sandbox limits and features used for one `run_javascript` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsSandboxConfig {
    pub profile: JsSandboxProfile,
    pub capture_console: bool,
    pub timeout: Duration,
    pub memory_limit_bytes: usize,
    pub max_stack_bytes: usize,
    pub max_code_bytes: usize,
    pub max_output_chars: usize,
}

impl Default for JsSandboxConfig {
    fn default() -> Self {
        Self::for_profile(JsSandboxProfile::Default)
    }
}

impl JsSandboxConfig {
    pub fn for_profile(profile: JsSandboxProfile) -> Self {
        match profile {
            JsSandboxProfile::Strict => Self {
                profile,
                capture_console: false,
                timeout: DEFAULT_TIMEOUT,
                memory_limit_bytes: DEFAULT_MEMORY_LIMIT_BYTES,
                max_stack_bytes: DEFAULT_MAX_STACK_BYTES,
                max_code_bytes: DEFAULT_MAX_CODE_BYTES,
                max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            },
            JsSandboxProfile::Default | JsSandboxProfile::Custom => Self {
                profile,
                capture_console: true,
                timeout: DEFAULT_TIMEOUT,
                memory_limit_bytes: DEFAULT_MEMORY_LIMIT_BYTES,
                max_stack_bytes: DEFAULT_MAX_STACK_BYTES,
                max_code_bytes: DEFAULT_MAX_CODE_BYTES,
                max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            },
            JsSandboxProfile::Roomy => Self {
                profile,
                capture_console: true,
                timeout: ROOMY_TIMEOUT,
                memory_limit_bytes: ROOMY_MEMORY_LIMIT_BYTES,
                max_stack_bytes: ROOMY_MAX_STACK_BYTES,
                max_code_bytes: ROOMY_MAX_CODE_BYTES,
                max_output_chars: ROOMY_MAX_OUTPUT_CHARS,
            },
        }
    }

    pub fn resolve(settings: &JavascriptSandboxSettings) -> Self {
        let mut config = Self::for_profile(settings.profile);
        if let Some(capture) = settings.capture_console {
            config.capture_console = capture;
        }
        if let Some(ms) = settings.timeout_ms {
            config.timeout = Duration::from_millis(u64::from(ms));
        }
        if let Some(mb) = settings.memory_mb {
            config.memory_limit_bytes = (mb as usize).saturating_mul(1024 * 1024);
        }
        if let Some(bytes) = settings.max_code_bytes {
            config.max_code_bytes = bytes as usize;
        }
        if let Some(chars) = settings.max_output_chars {
            config.max_output_chars = chars as usize;
        }
        if let Some(kb) = settings.max_stack_kb {
            config.max_stack_bytes = (kb as usize).saturating_mul(1024);
        }
        config
    }

    pub fn from_runtime_settings(settings: &crate::runtime_settings::RuntimeSettings) -> Self {
        Self::resolve(&settings.javascript_sandbox)
    }

    /// Model-facing note appended to the `run_javascript` tool description.
    pub fn describe_for_model(&self) -> String {
        let timeout_secs = self.timeout.as_secs().max(1);
        let memory_mb = (self.memory_limit_bytes / (1024 * 1024)).max(1);
        let code_kb = (self.max_code_bytes / 1024).max(1);
        let console = if self.capture_console {
            "console.log/info/warn/error are captured into the `logs` array of the JSON result \
             `{\"return\": <value|null>, \"logs\": [\"...\"]}`. Prefer `return` for the primary \
             value; log-only scripts still return that envelope."
        } else {
            "There is no `console` object; use `return` with a JSON-serializable value."
        };
        format!(
            "Not Node or Python: no modules, require/import, network, filesystem, npm, or \
             symbolic-math libraries. {console} Limits: {code_kb} KB code, {timeout_secs}s, \
             {memory_mb} MiB memory."
        )
    }

    pub fn describe_for_catalog(&self) -> String {
        let timeout_secs = self.timeout.as_secs().max(1);
        let code_kb = (self.max_code_bytes / 1024).max(1);
        let console = if self.capture_console {
            "console captured"
        } else {
            "no console"
        };
        format!("QuickJS sandbox ({code_kb} KB code, {timeout_secs}s, {console}, no I/O).")
    }
}

/// Run user JavaScript with the default (console-on) profile.
pub fn run_javascript(code: &str, timeout: Duration) -> anyhow::Result<String> {
    let config = JsSandboxConfig {
        timeout,
        ..JsSandboxConfig::default()
    };
    run_javascript_with_config(code, &config)
}

/// Run user JavaScript synchronously inside a fresh QuickJS runtime.
pub fn run_javascript_with_config(code: &str, config: &JsSandboxConfig) -> anyhow::Result<String> {
    anyhow::ensure!(!code.is_empty(), "run_javascript requires non-empty `code`");
    anyhow::ensure!(
        code.len() <= config.max_code_bytes,
        "code exceeds the {} byte limit",
        config.max_code_bytes
    );
    block_dangerous_tokens(code)?;

    let runtime = Runtime::new().context("create QuickJS runtime")?;
    runtime.set_memory_limit(config.memory_limit_bytes);
    runtime.set_max_stack_size(config.max_stack_bytes);

    let deadline = Instant::now() + config.timeout;
    let timed_out = Arc::new(AtomicBool::new(false));
    let timed_out_interrupt = timed_out.clone();
    runtime.set_interrupt_handler(Some(Box::new(move || {
        if Instant::now() >= deadline {
            timed_out_interrupt.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    })));

    let context = JsContext::full(&runtime).context("create QuickJS context")?;
    let capture_console = config.capture_console;
    let output = context.with(|ctx| {
        eval_user_code(&ctx, code, capture_console)
            .catch(&ctx)
            .map_err(|error| {
                if timed_out.load(Ordering::Relaxed) {
                    anyhow::anyhow!(
                        "execution timed out after {}s",
                        config.timeout.as_secs().max(1)
                    )
                } else {
                    anyhow::anyhow!("{error}")
                }
            })
    })?;

    truncate_output(output, config.max_output_chars)
}

fn eval_user_code<'js>(
    ctx: &Ctx<'js>,
    code: &str,
    capture_console: bool,
) -> rquickjs::Result<String> {
    if capture_console {
        install_console(ctx)?;
    }

    let globals = ctx.globals();
    let json: rquickjs::Object = globals.get("JSON")?;
    let stringify: rquickjs::Function = json.get("stringify")?;

    let script = format!("(function() {{\n'use strict';\n{code}\n}})()");
    let value: Value = ctx.eval(script.as_bytes())?;

    if !capture_console {
        if value.is_undefined() {
            return Ok(String::new());
        }
        let serialized: String = stringify.call((value,))?;
        return Ok(serialized);
    }

    let logs = read_logs(ctx)?;
    let return_json = if value.is_undefined() {
        serde_json::Value::Null
    } else {
        let serialized: String = stringify.call((value,))?;
        serde_json::from_str(&serialized).unwrap_or(serde_json::Value::String(serialized))
    };

    Ok(json!({ "return": return_json, "logs": logs }).to_string())
}

fn install_console<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<()> {
    // Bind on globalThis so the strict-mode IIFE can see `console`.
    let _: () = ctx.eval(
        r#"
globalThis.__brazier_logs = [];
globalThis.console = {
  log: function () { globalThis.__brazier_log(arguments); },
  info: function () { globalThis.__brazier_log(arguments); },
  warn: function () { globalThis.__brazier_log(arguments); },
  error: function () { globalThis.__brazier_log(arguments); },
  debug: function () { globalThis.__brazier_log(arguments); }
};
globalThis.__brazier_log = function (args) {
  var parts = [];
  for (var i = 0; i < args.length; i++) {
    var a = args[i];
    if (typeof a === 'string') {
      parts.push(a);
    } else {
      try { parts.push(JSON.stringify(a)); }
      catch (e) { parts.push(String(a)); }
    }
  }
  globalThis.__brazier_logs.push(parts.join(' '));
};
"#,
    )?;
    Ok(())
}

fn read_logs<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<Vec<String>> {
    let globals = ctx.globals();
    let logs_value: Value = globals.get("__brazier_logs")?;
    let Ok(array) = Array::from_js(ctx, logs_value) else {
        return Ok(Vec::new());
    };
    let len = array.len();
    let mut logs = Vec::with_capacity(len.min(MAX_LOG_LINES));
    for i in 0..len.min(MAX_LOG_LINES) {
        let entry: Value = array.get(i)?;
        match String::from_js(ctx, entry) {
            Ok(text) => logs.push(text),
            Err(_) => logs.push(String::new()),
        }
    }
    if len > MAX_LOG_LINES {
        logs.push(format!(
            "… [{} more log lines truncated]",
            len - MAX_LOG_LINES
        ));
    }
    Ok(logs)
}

fn block_dangerous_tokens(code: &str) -> anyhow::Result<()> {
    let lower = code.to_ascii_lowercase();
    const BLOCKED: &[&str] = &[
        "import ",
        "require(",
        "eval(",
        "Function(",
        "WebAssembly",
        "Atomics",
        "SharedArrayBuffer",
        "process.",
        "Deno.",
        "Bun.",
    ];
    for token in BLOCKED {
        if lower.contains(token) {
            anyhow::bail!("code contains disallowed token `{token}`");
        }
    }
    Ok(())
}

fn truncate_output(mut output: String, max_output_chars: usize) -> anyhow::Result<String> {
    if output.len() > max_output_chars {
        let mut cut = max_output_chars;
        while !output.is_char_boundary(cut) {
            cut -= 1;
        }
        output.truncate(cut);
        output.push_str("… [truncated]");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_evaluates_expressions_with_envelope_by_default() {
        let out = run_javascript("return 6 * 7;", Duration::from_secs(1)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["return"], 42);
        assert_eq!(parsed["logs"], json!([]));
    }

    #[test]
    fn sandbox_captures_console_logs_without_return() {
        let out = run_javascript(
            r#"console.log("hello", 1 + 1); console.warn({a: 1});"#,
            Duration::from_secs(1),
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(parsed["return"].is_null());
        assert_eq!(parsed["logs"][0], "hello 2");
        assert_eq!(parsed["logs"][1], r#"{"a":1}"#);
    }

    #[test]
    fn sandbox_strict_profile_returns_plain_json_without_console() {
        let config = JsSandboxConfig::for_profile(JsSandboxProfile::Strict);
        let out = run_javascript_with_config("return 6 * 7;", &config).unwrap();
        assert_eq!(out, "42");

        let err = run_javascript_with_config("console.log('x'); return 1;", &config);
        assert!(err.is_err(), "strict profile has no console: {err:?}");
    }

    #[test]
    fn sandbox_blocks_require() {
        assert!(run_javascript("require('fs')", Duration::from_secs(1)).is_err());
    }

    #[test]
    fn sandbox_respects_timeout() {
        let result = run_javascript("while (true) {}", Duration::from_millis(50));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[test]
    fn roomy_profile_allows_larger_code() {
        let config = JsSandboxConfig::for_profile(JsSandboxProfile::Roomy);
        assert!(config.max_code_bytes > DEFAULT_MAX_CODE_BYTES);
        assert!(config.capture_console);
        assert_eq!(config.timeout, ROOMY_TIMEOUT);
    }

    #[test]
    fn settings_overlays_override_profile() {
        let settings = JavascriptSandboxSettings {
            profile: JsSandboxProfile::Default,
            capture_console: Some(false),
            timeout_ms: Some(5_000),
            memory_mb: Some(16),
            max_code_bytes: Some(32_000),
            max_output_chars: Some(12_000),
            max_stack_kb: Some(400),
        };
        let config = JsSandboxConfig::resolve(&settings);
        assert!(!config.capture_console);
        assert_eq!(config.timeout, Duration::from_millis(5_000));
        assert_eq!(config.memory_limit_bytes, 16 * 1024 * 1024);
        assert_eq!(config.max_code_bytes, 32_000);
        assert_eq!(config.max_output_chars, 12_000);
        assert_eq!(config.max_stack_bytes, 400 * 1024);
    }

    #[test]
    fn model_description_mentions_limits_and_not_node() {
        let text = JsSandboxConfig::default().describe_for_model();
        assert!(text.contains("Not Node"));
        assert!(text.contains("console.log"));
        assert!(text.contains("no modules"));
    }
}
