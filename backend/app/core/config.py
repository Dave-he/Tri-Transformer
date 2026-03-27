from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./tritransformer.db"
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    mock_inference: bool = True
    model_path: Optional[str] = None
    embedding_model_path: str = "BAAI/bge-large-zh-v1.5"
    reranker_model_path: str = "BAAI/bge-reranker-large"

    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    train_d_model: int = 512
    train_num_heads: int = 8
    train_num_layers: int = 6
    train_vocab_size: int = 32000
    train_max_seq_len: int = 512
    train_epochs_default: int = 3
    train_lr_default: float = 1e-4
    train_device: str = "cpu"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
