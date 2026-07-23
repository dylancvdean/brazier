//! Bounded QuickJS sandbox for the bundled `run_javascript` tool.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Context;
use rquickjs::{CatchResultExt, Context as JsContext, Ctx, Runtime, Value};

pub(crate) const MAX_CODE_BYTES: usize = 16_384;
const MAX_OUTPUT_CHARS: usize = 8_000;
const MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const MAX_STACK_BYTES: usize = 256 * 1024;

/// Run user JavaScript synchronously inside a fresh QuickJS runtime.
pub fn run_javascript(code: &str, timeout: Duration) -> anyhow::Result<String> {
    anyhow::ensure!(!code.is_empty(), "run_javascript requires non-empty `code`");
    anyhow::ensure!(
        code.len() <= MAX_CODE_BYTES,
        "code exceeds the {MAX_CODE_BYTES} byte limit"
    );
    block_dangerous_tokens(code)?;

    let runtime = Runtime::new().context("create QuickJS runtime")?;
    runtime.set_memory_limit(MEMORY_LIMIT_BYTES);
    runtime.set_max_stack_size(MAX_STACK_BYTES);

    let deadline = Instant::now() + timeout;
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
    let output = context.with(|ctx| {
        eval_user_code(&ctx, code).catch(&ctx).map_err(|error| {
            if timed_out.load(Ordering::Relaxed) {
                anyhow::anyhow!("execution timed out after {}s", timeout.as_secs())
            } else {
                anyhow::anyhow!("{error}")
            }
        })
    })?;

    truncate_output(output)
}

fn eval_user_code<'js>(ctx: &Ctx<'js>, code: &str) -> rquickjs::Result<String> {
    let globals = ctx.globals();
    let json: rquickjs::Object = globals.get("JSON")?;
    let stringify: rquickjs::Function = json.get("stringify")?;

    let script = format!("(function() {{\n'use strict';\n{code}\n}})()");
    let value: Value = ctx.eval(script.as_bytes())?;
    if value.is_undefined() {
        return Ok(String::new());
    }
    let serialized: String = stringify.call((value,))?;
    Ok(serialized)
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

fn truncate_output(mut output: String) -> anyhow::Result<String> {
    if output.len() > MAX_OUTPUT_CHARS {
        let mut cut = MAX_OUTPUT_CHARS;
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
    fn sandbox_evaluates_expressions() {
        let out = run_javascript("return 6 * 7;", Duration::from_secs(1)).unwrap();
        assert_eq!(out, "42");
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
}
