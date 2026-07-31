//! Fara1.5 action dialect: XML `<tool_call>` blocks calling `computer_use`.

use anyhow::{Context, Result, bail};
use brazier_protocol::computer_types::ComputerAction;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct FaraParseResult {
    pub thought: Option<String>,
    pub actions: Vec<ComputerAction>,
    pub raw_tool_calls: Vec<String>,
}

/// Parse model output that may contain a thought block plus Fara tool calls.
pub fn parse_fara_output(text: &str) -> Result<FaraParseResult> {
    let thought = extract_thought(text);
    let mut actions = Vec::new();
    let mut raw_tool_calls = Vec::new();

    for block in extract_tool_call_blocks(text) {
        raw_tool_calls.push(block.clone());
        actions.push(parse_tool_call_json(&block)?);
    }

    // Some servers wrap a single JSON object without XML.
    if actions.is_empty() {
        let trimmed = text.trim();
        if trimmed.starts_with('{')
            && let Ok(action) = parse_tool_call_json(trimmed)
        {
            actions.push(action);
            raw_tool_calls.push(trimmed.to_owned());
        }
    }

    Ok(FaraParseResult {
        thought,
        actions,
        raw_tool_calls,
    })
}

fn extract_thought(text: &str) -> Option<String> {
    for (open, close) in [
        ("<think>", "</think>"),
        ("<thought>", "</thought>"),
        ("```thought", "```"),
    ] {
        if let Some(start) = text.find(open) {
            let after = start + open.len();
            if let Some(end) = text[after..].find(close) {
                let thought = text[after..after + end].trim();
                if !thought.is_empty() {
                    return Some(thought.to_owned());
                }
            }
        }
    }
    // Text before the first tool call is treated as thought.
    if let Some(idx) = text.find("<tool_call>") {
        let prefix = text[..idx].trim();
        if !prefix.is_empty() {
            return Some(prefix.to_owned());
        }
    }
    None
}

fn extract_tool_call_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = start + "<tool_call>".len();
        let Some(end) = rest[after..].find("</tool_call>") else {
            break;
        };
        blocks.push(rest[after..after + end].trim().to_owned());
        rest = &rest[after + end + "</tool_call>".len()..];
    }
    blocks
}

fn parse_tool_call_json(block: &str) -> Result<ComputerAction> {
    let value: Value = serde_json::from_str(block).context("Fara tool_call JSON")?;
    // Accept either {"name":"computer_use","arguments":{...}} or bare action object.
    let args = if value.get("name").and_then(|v| v.as_str()) == Some("computer_use") {
        value
            .get("arguments")
            .cloned()
            .or_else(|| value.get("parameters").cloned())
            .unwrap_or(Value::Null)
    } else if value.get("action").is_some() || value.get("type").is_some() {
        value.clone()
    } else if let Some(function) = value.get("function") {
        function.get("arguments").cloned().unwrap_or(Value::Null)
    } else {
        value.clone()
    };

    let args = if let Value::String(text) = args {
        serde_json::from_str(&text).context("nested arguments JSON string")?
    } else {
        args
    };

    action_from_args(&args)
}

fn action_from_args(args: &Value) -> Result<ComputerAction> {
    let action = args
        .get("action")
        .or_else(|| args.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("screenshot");

    let coord = |key_x: &str, key_y: &str| -> Result<(f64, f64)> {
        if let Some(arr) = args.get("coordinate").and_then(|v| v.as_array())
            && arr.len() >= 2
        {
            return Ok((
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
            ));
        }
        let x = args
            .get(key_x)
            .and_then(|v| v.as_f64())
            .or_else(|| args.get("x").and_then(|v| v.as_f64()))
            .unwrap_or(0.0);
        let y = args
            .get(key_y)
            .and_then(|v| v.as_f64())
            .or_else(|| args.get("y").and_then(|v| v.as_f64()))
            .unwrap_or(0.0);
        Ok((x, y))
    };

    Ok(match action {
        "screenshot" => ComputerAction::Screenshot,
        "left_click" | "click" => {
            let (x, y) = coord("x", "y")?;
            ComputerAction::LeftClick { x, y }
        }
        "right_click" => {
            let (x, y) = coord("x", "y")?;
            ComputerAction::RightClick { x, y }
        }
        "double_click" => {
            let (x, y) = coord("x", "y")?;
            ComputerAction::DoubleClick { x, y }
        }
        "triple_click" => {
            let (x, y) = coord("x", "y")?;
            ComputerAction::TripleClick { x, y }
        }
        "mouse_move" => {
            let (x, y) = coord("x", "y")?;
            ComputerAction::MouseMove { x, y }
        }
        "left_click_drag" | "drag" => {
            let (start_x, start_y) =
                if let Some(arr) = args.get("start_coordinate").and_then(|v| v.as_array()) {
                    (
                        arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0),
                        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
                    )
                } else {
                    coord("start_x", "start_y")?
                };
            let (end_x, end_y) =
                if let Some(arr) = args.get("coordinate").and_then(|v| v.as_array()) {
                    (
                        arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0),
                        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
                    )
                } else {
                    coord("end_x", "end_y")?
                };
            ComputerAction::LeftClickDrag {
                start_x,
                start_y,
                end_x,
                end_y,
            }
        }
        "type" | "type_text" => ComputerAction::Type {
            text: args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        },
        "keypress" | "key" | "hotkey" => {
            let keys = if let Some(arr) = args.get("keys").and_then(|v| v.as_array()) {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            } else if let Some(key) = args.get("key").and_then(|v| v.as_str()) {
                vec![key.to_owned()]
            } else {
                Vec::new()
            };
            ComputerAction::Keypress { keys }
        }
        "scroll" => {
            let (x, y) = coord("x", "y").unwrap_or((0.0, 0.0));
            let delta_x = args
                .get("delta_x")
                .or_else(|| args.get("scroll_x"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let delta_y = args
                .get("delta_y")
                .or_else(|| args.get("scroll_y"))
                .or_else(|| args.get("pixels"))
                .and_then(|v| v.as_f64())
                .unwrap_or(-400.0);
            ComputerAction::Scroll {
                x,
                y,
                delta_x,
                delta_y,
            }
        }
        "wait" => ComputerAction::Wait {
            milliseconds: args
                .get("milliseconds")
                .or_else(|| args.get("ms"))
                .or_else(|| args.get("time"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1000),
        },
        "visit_url" | "goto" | "navigate" => ComputerAction::VisitUrl {
            url: args
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("about:blank")
                .to_owned(),
        },
        "web_search" | "search" => ComputerAction::WebSearch {
            query: args
                .get("query")
                .or_else(|| args.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        },
        "pause_and_memorize_fact" | "memorize" => ComputerAction::Memorize {
            fact: args
                .get("fact")
                .or_else(|| args.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        },
        "ask_user_question" | "ask_user" => ComputerAction::AskUser {
            question: args
                .get("question")
                .or_else(|| args.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("Need your input.")
                .to_owned(),
        },
        "terminate" | "done" | "finish" => ComputerAction::Terminate {
            response: args
                .get("response")
                .or_else(|| args.get("text"))
                .and_then(|v| v.as_str())
                .map(str::to_owned),
        },
        other => bail!("unsupported Fara action: {other}"),
    })
}

/// Whether a model id/name looks like a Fara computer-use model.
pub fn looks_like_fara_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("fara") || lower.contains("fara1.5") || lower.contains("fara-1.5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xml_tool_call() {
        let text = r#"I should open the site.
<tool_call>
{"name":"computer_use","arguments":{"action":"visit_url","url":"https://example.com"}}
</tool_call>"#;
        let parsed = parse_fara_output(text).unwrap();
        assert!(parsed.thought.unwrap().contains("open the site"));
        assert_eq!(
            parsed.actions,
            vec![ComputerAction::VisitUrl {
                url: "https://example.com".into()
            }]
        );
    }

    #[test]
    fn parses_click_coordinates() {
        let text = r#"<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[120,240]}}
</tool_call>"#;
        let parsed = parse_fara_output(text).unwrap();
        assert_eq!(
            parsed.actions[0],
            ComputerAction::LeftClick { x: 120.0, y: 240.0 }
        );
    }
}
