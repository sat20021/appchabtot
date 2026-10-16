from datetime import datetime
from typing import List, Optional, Dict, Any
import mongoengine as me

class DocumentStatus:
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class KnowledgeDocument(me.Document):
    """
    MongoDB model for a knowledge document.
    """
    meta = {
        'collection': 'knowledge_documents',
        'indexes': [
            'doc_id',
            'domain',
            'status',
            ('domain', 'status'),
            'uploaded_at'
        ]
    }

    doc_id = me.StringField(required=True, unique=True)
    original_name = me.StringField(required=True)
    file_name = me.StringField(required=True)  # Stored filename (e.g. uuid_original.pdf)
    domain = me.StringField(required=True)
    
    # Storage details
    storage_path = me.StringField(required=True)  # s3://... or /local/path/...
    size_bytes = me.IntField(default=0)
    content_type = me.StringField(default="application/octet-stream")
    checksum = me.StringField()
    
    # Processing status
    embedding_status = me.StringField(default="pending") # pending, processing, completed, failed
    chunk_count = me.IntField(default=0)
    
    # Metadata
    tags = me.ListField(me.StringField())
    description = me.StringField()
    uploaded_by = me.StringField()
    metadata = me.DictField()
    
    # Lifecycle
    status = me.StringField(default=DocumentStatus.ACTIVE)
    uploaded_at = me.DateTimeField(default=datetime.utcnow)
    deleted_at = me.DateTimeField()

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "doc_id": self.doc_id,
            "original_name": self.original_name,
            "domain": self.domain,
            "status": self.status,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "embedding_status": self.embedding_status,
            "chunk_count": self.chunk_count,
            "tags": self.tags,
            "metadata": self.metadata
        }
