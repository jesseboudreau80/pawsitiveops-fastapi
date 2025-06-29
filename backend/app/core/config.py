from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI fallback
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # OpenRouter support
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_PROVIDER: str = "openai"

    class Config:
        env_file = ".env"

settings = Settings()
