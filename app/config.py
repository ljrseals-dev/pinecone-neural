from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required
    pinecone_api_key: str
    openai_api_key: str

    # Optional with defaults
    pinecone_index: str = "agent-harness-v2"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"

    max_context_tokens: int = 3500
    summary_token_limit: int = 200
    prune_age_days: int = 7
    prune_utility_floor: float = 0.15
    soft_delete_retention_days: int = 14

    max_steps: int = 6
    max_run_tokens: int = 50_000
    max_run_usd: float = 0.25
    tool_timeout_s: float = 2.0
    tool_output_max_bytes: int = 4096

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"  # comma-separated in production


settings = Settings()
