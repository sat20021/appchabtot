import os
from pathlib import Path

class Config:
    APP_NAME = "Enterprise FAQ Engine"
    VERSION = "1.0.0"
    
    # Model Config
    MODEL_NAME = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    CONFIDENCE_THRESHOLD = 0.8
    
    # Paths
    # Use centralized LOCAL_STORAGE_PATH if available
    local_storage = os.getenv("LOCAL_STORAGE_PATH")
    if local_storage:
         local_path = Path(local_storage)
         DATA_DIR = str(local_path / "csv_files")
         ARTIFACTS_DIR = str(local_path / "faq_artifacts")
    else:
         BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         DATA_DIR = os.path.join(BASE_DIR, "data")
         ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
         
    INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
    METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.pkl")

    # Create dirs if not exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    @classmethod
    def get_domain_data_dir(cls, domain: str) -> str:
        path = os.path.join(cls.DATA_DIR, domain)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_domain_artifacts_dir(cls, domain: str) -> str:
        path = os.path.join(cls.ARTIFACTS_DIR, domain)
        os.makedirs(path, exist_ok=True)
        return path
        
    @classmethod
    def get_index_path(cls, domain: str) -> str:
        return os.path.join(cls.get_domain_artifacts_dir(domain), "faiss_index.bin")
        
    @classmethod
    def get_metadata_path(cls, domain: str) -> str:
        return os.path.join(cls.get_domain_artifacts_dir(domain), "metadata.pkl")
