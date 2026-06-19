from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    
    index_cache_dir: str = "datasets/cache/index"
    embeddings_cache_dir: str = "datasets/cache/embeddings"
    synonyms_path: str = "datasets/miracl/ar/synonyms.json"

    class Config:
        env_file = ".env"

settings = Settings()
