pub use utoipa_gen::{path, OpenApi, ToSchema};

pub trait OpenApi {
    fn openapi() -> openapi::OpenApi;
}

pub trait ToSchema {
    fn name() -> std::borrow::Cow<'static, str>;
}

pub mod openapi {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct OpenApi {
        pub openapi: String,
        pub info: Info,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct Info {
        pub title: String,
        pub version: String,
        pub description: Option<String>,
    }

    impl OpenApi {
        pub fn new(title: impl Into<String>, version: impl Into<String>) -> Self {
            Self {
                openapi: "3.1.0".to_string(),
                info: Info {
                    title: title.into(),
                    version: version.into(),
                    description: Some("GFlowNet-Prover Autonomous Neuro-Symbolic Verification Service".to_string()),
                },
            }
        }

        pub fn to_json(&self) -> Result<String, serde_json::Error> {
            serde_json::to_string(self)
        }

        pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
            serde_json::to_string_pretty(self)
        }
    }
}
