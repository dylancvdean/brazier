use async_trait::async_trait;

use crate::types::{ChatCompletionRequest, ModelCapabilities, ModelDescriptor, text_from_content};

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub reasoning: Option<String>,
}

#[async_trait]
pub trait Engine: Send + Sync {
    fn id(&self) -> &'static str;
    async fn models(&self) -> anyhow::Result<Vec<ModelDescriptor>>;
    async fn generate(&self, request: &ChatCompletionRequest) -> anyhow::Result<Generation>;
}

#[derive(Default)]
pub struct MockEngine;

#[async_trait]
impl Engine for MockEngine {
    fn id(&self) -> &'static str {
        "mock"
    }

    async fn models(&self) -> anyhow::Result<Vec<ModelDescriptor>> {
        Ok(vec![ModelDescriptor {
            id: "brazier/mock".to_owned(),
            name: "Brazier Development Model".to_owned(),
            engine: self.id().to_owned(),
            capabilities: ModelCapabilities {
                input_modalities: vec![
                    "text".into(),
                    "image".into(),
                    "audio".into(),
                    "video".into(),
                ],
                output_modalities: vec!["text".into()],
                streaming: true,
                tools: true,
                reasoning: true,
            },
        }])
    }

    async fn generate(&self, request: &ChatCompletionRequest) -> anyhow::Result<Generation> {
        let prompt = request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map(|message| text_from_content(&message.content))
            .unwrap_or_else(|| "Hello".to_owned());
        let tools = request.tools.as_ref().map_or(
            "",
            |_| " Tool definitions were received but were not executed by the development engine.",
        );
        Ok(Generation {
            text: format!("Local development response: {prompt}.{tools}"),
            reasoning: Some(
                "The deterministic engine echoes the newest user content to test the complete chat pipeline."
                    .to_owned(),
            ),
        })
    }
}
