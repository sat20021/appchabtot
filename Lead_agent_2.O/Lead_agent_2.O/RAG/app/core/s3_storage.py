import os
import logging
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = None
    
import shutil
from typing import Optional, BinaryIO
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class S3StorageService:
    """
    Service for interacting with AWS S3 or Local Storage (Hybrid).
    """
    def __init__(self):
        self.use_s3 = os.getenv("USE_S3", "false").lower() == "true"
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.local_storage_root = Path(os.getenv("LOCAL_STORAGE_PATH", "./storage"))
        
        self._s3_client = None
        
        if self.use_s3:
            # Log S3 configuration status
            if all([self.bucket_name, self.aws_access_key_id, self.aws_secret_access_key]):
                logger.info(f"📦 S3 Storage configured: bucket={self.bucket_name}, region={self.aws_region}")
            else:
                logger.warning("⚠️ S3 Storage NOT configured - missing credentials")
        else:
            self.local_storage_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Local Storage configured at: {self.local_storage_root}")

    @property
    def s3_client(self):
        if not self.use_s3:
            return None

        if self._s3_client is None:
            if not all([self.bucket_name, self.aws_access_key_id, self.aws_secret_access_key]):
                logger.warning("⚠️ S3 credentials not fully configured.")
                return None
            
            try:
                logger.info(f"🔌 Initializing S3 client for region: {self.aws_region}")
                self._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region
                )
                logger.info(f"✅ S3 client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create S3 client: {e}")
                return None
        return self._s3_client

    def is_s3_enabled(self) -> bool:
        """Check if S3 (or storage) is configured and accessible."""
        if self.use_s3:
            return self.s3_client is not None
        return True # Local storage is always enabled if fallback is on

    def _get_local_path(self, key: str) -> Path:
        """Resolve key to local storage path."""
        # Clean key
        clean_key = key.lstrip("/").replace("..", "")
        return self.local_storage_root / clean_key

    def upload_file(self, file_obj: BinaryIO, key: str, content_type: str = "application/octet-stream") -> bool:
        """
        Upload a file-like object to S3 or local storage.
        """
        # --- Local Storage Mode ---
        if not self.use_s3:
            try:
                dest_path = self._get_local_path(key)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get file size logic for logging
                try:
                     current_pos = file_obj.tell()
                     file_obj.seek(0, 2)
                     file_size = file_obj.tell()
                     file_obj.seek(current_pos)
                except:
                     file_size = "unknown"

                logger.info(f"📤 LOCAL UPLOAD START: key={key}, size={file_size}")
                
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(file_obj, f)
                
                logger.info(f"✅ LOCAL UPLOAD SUCCESS: {dest_path}")
                return True
            except Exception as e:
                logger.error(f"❌ LOCAL UPLOAD FAILED: key={key}, error={e}")
                return False

        # --- S3 Mode ---
        client = self.s3_client
        if not client:
            logger.error("❌ S3 Upload failed: S3 client not available")
            return False
        
        try:
            # Get file size for logging
            current_pos = file_obj.tell()
            file_obj.seek(0, 2)  # Seek to end
            file_size = file_obj.tell()
            file_obj.seek(current_pos)  # Reset to original position
            
            logger.info(f"📤 S3 UPLOAD START: key={key}, size={file_size:,} bytes, type={content_type}")
            
            client.upload_fileobj(
                file_obj, 
                self.bucket_name, 
                key,
                ExtraArgs={"ContentType": content_type}
            )
            
            logger.info(f"✅ S3 UPLOAD SUCCESS: s3://{self.bucket_name}/{key} ({file_size:,} bytes)")
            return True
        except ClientError as e:
            logger.error(f"❌ S3 UPLOAD FAILED: key={key}, error={e}")
            return False
        except Exception as e:
            logger.error(f"❌ S3 UPLOAD FAILED (unexpected): key={key}, error={e}")
            return False

    def download_to_file(self, key: str, destination_path: Path) -> bool:
        """
        Download a file from S3 (or copy from local) to a local path.
        """
        # --- Local Storage Mode ---
        if not self.use_s3:
            try:
                source_path = self._get_local_path(key)
                if not source_path.exists():
                     logger.error(f"❌ LOCAL DOWNLOAD FAILED: File not found {source_path}")
                     return False
                
                logger.info(f"📥 LOCAL DOWNLOAD START: {source_path} -> {destination_path}")
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                
                file_size = destination_path.stat().st_size
                logger.info(f"✅ LOCAL DOWNLOAD SUCCESS: ({file_size:,} bytes)")
                return True
            except Exception as e:
                logger.error(f"❌ LOCAL DOWNLOAD FAILED: key={key}, error={e}")
                return False

        # --- S3 Mode ---
        client = self.s3_client
        if not client:
            logger.error("❌ S3 Download failed: S3 client not available")
            return False
            
        try:
            logger.info(f"📥 S3 DOWNLOAD START: key={key} -> {destination_path}")
            
            # Ensure parent directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            client.download_file(self.bucket_name, key, str(destination_path))
            
            # Get downloaded file size
            file_size = destination_path.stat().st_size if destination_path.exists() else 0
            logger.info(f"✅ S3 DOWNLOAD SUCCESS: s3://{self.bucket_name}/{key} -> {destination_path} ({file_size:,} bytes)")
            return True
        except ClientError as e:
            logger.error(f"❌ S3 DOWNLOAD FAILED: key={key}, error={e}")
            return False
        except Exception as e:
            logger.error(f"❌ S3 DOWNLOAD FAILED (unexpected): key={key}, error={e}")
            return False

    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3 or local storage.
        """
        # --- Local Storage Mode ---
        if not self.use_s3:
            try:
                target_path = self._get_local_path(key)
                if target_path.exists():
                    target_path.unlink()
                    logger.info(f"✅ LOCAL DELETE SUCCESS: {target_path}")
                    return True
                else:
                    logger.warning(f"⚠️ LOCAL DELETE SKIPPED: File not found {target_path}")
                    return False
            except Exception as e:
                logger.error(f"❌ LOCAL DELETE FAILED: key={key}, error={e}")
                return False

        # --- S3 Mode ---
        client = self.s3_client
        if not client:
            logger.error("❌ S3 Delete failed: S3 client not available")
            return False
            
        try:
            logger.info(f"🗑️ S3 DELETE START: key={key}")
            client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"✅ S3 DELETE SUCCESS: s3://{self.bucket_name}/{key}")
            return True
        except ClientError as e:
            logger.error(f"❌ S3 DELETE FAILED: key={key}, error={e}")
            return False

_s3_service = S3StorageService()

def get_s3_storage_service() -> S3StorageService:
    return _s3_service

