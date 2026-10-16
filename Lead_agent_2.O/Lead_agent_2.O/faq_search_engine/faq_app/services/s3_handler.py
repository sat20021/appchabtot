try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = None

import os
import logging
import shutil
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class S3Handler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Handler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.use_s3 = os.getenv("USE_S3", "false").lower() == "true"
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.local_storage_root = Path(os.getenv("LOCAL_STORAGE_PATH", "./storage"))
        
        if self.use_s3:
            if not self.bucket_name:
                logger.warning("S3_BUCKET_NAME not set. S3 functionality will be disabled.")
                self.s3_client = None
            else:
                try:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                        region_name=self.region
                    )
                    logger.info(f"S3 Handler initialized for bucket: {self.bucket_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize S3 client: {e}")
                    self.s3_client = None
        else:
            self.s3_client = None
            self.local_storage_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local Storage Handler initialized at: {self.local_storage_root}")
        
        self._initialized = True

    def _get_local_storage_path(self, key: str) -> Path:
        """Get the full path in local storage for a given key, creating parent dirs."""
        clean_key = key.lstrip("/").replace("..", "")
        full_path = self.local_storage_root / clean_key
        return full_path

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 (or copy from local) with caching based on ETag.
        """
        local_dest = Path(local_path)
        etag_file = local_dest.with_suffix(local_dest.suffix + ".etag")
        
        # --- Local Storage Mode ---
        if not self.use_s3:
            source_path = self._get_local_storage_path(s3_key)
            if not source_path.exists():
                logger.warning(f"File not found in local storage: {source_path}")
                return False
            try:
                # Prevent copying to self
                if source_path.resolve() == local_dest.resolve():
                    logger.info(f"Source and destination are same, skipping copy: {local_path}")
                    return True

                local_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, local_dest)
                logger.info(f"Copied from local storage: {s3_key} -> {local_path}")
                return True
            except Exception as e:
                logger.error(f"Error copying local file {s3_key}: {e}")
                return False

        # --- S3 Mode ---
        if not self.s3_client:
            logger.warning("S3 client not initialized. Skipping download.")
            return False

        try:
            # Check if file exists in S3
            try:
                head = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.warning(f"File not found in S3: {s3_key}")
                    return False
                raise

            s3_etag = head['ETag'].strip('"')
            
            # Check local file and ETag
            if local_dest.exists() and etag_file.exists():
                try:
                    with open(etag_file, 'r') as f:
                        local_etag = f.read().strip()
                    
                    if local_etag == s3_etag:
                        logger.info(f"Using cached file for {s3_key} (ETag match)")
                        return True
                except Exception:
                    logger.warning(f"Could not read ETag file for {local_path}, forcing download.")

            # Download
            logger.info(f"Downloading {s3_key} from S3...")
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            
            # Save ETag
            with open(etag_file, 'w') as f:
                f.write(s3_etag)
                
            logger.info(f"Downloaded {s3_key} successfully.")
            return True

        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3 or local storage."""
        if not os.path.exists(local_path):
             logger.error(f"Source file does not exist: {local_path}")
             return False

        # --- Local Storage Mode ---
        if not self.use_s3:
            try:
                dest_path = self._get_local_storage_path(s3_key)

                # Prevent copying to self
                if Path(local_path).resolve() == dest_path.resolve():
                    logger.info(f"Source and destination are same, skipping upload copy: {local_path}")
                    return True

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest_path)
                logger.info(f"Copied to local storage: {local_path} -> {s3_key}")
                return True
            except Exception as e:
                logger.error(f"Error uploading to local storage {s3_key}: {e}")
                return False

        # --- S3 Mode ---
        if not self.s3_client:
            return False

        try:
            logger.info(f"Uploading {local_path} to S3 as {s3_key}...")
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(f"Uploaded {s3_key} successfully.")
            return True
        except Exception as e:
            logger.error(f"Error uploading {s3_key}: {e}")
            return False
