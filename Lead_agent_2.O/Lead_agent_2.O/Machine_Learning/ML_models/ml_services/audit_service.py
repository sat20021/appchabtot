"""
Audit Logging Service

Tracks security-relevant events and user actions for compliance and security monitoring.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum

from pydantic import BaseModel, Field
from pymongo import MongoClient

from core.state import mongo_db

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Enumeration of auditable actions"""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    REGISTER = "register"
    TOKEN_REFRESH = "token_refresh"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    
    # Data Operations
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_DOWNLOAD = "document_download"
    FAQ_UPLOAD = "faq_upload"
    FAQ_DELETE = "faq_delete"
    
    # Queries
    RAG_QUERY = "rag_query"
    FAQ_SEARCH = "faq_search"
    CONVERSATION = "conversation"
    
    # Admin Actions
    USER_ROLE_UPDATE = "user_role_update"
    USER_DEACTIVATE = "user_deactivate"
    
    # Security Events
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"


class AuditStatus(str, Enum):
    """Status of the audited action"""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class AuditLog(BaseModel):
    """Audit log entry"""
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    action: AuditAction
    resource: Optional[str] = None  # Endpoint or resource ID
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: AuditStatus
    metadata: Dict = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "audit_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-11-21T18:00:00Z",
                "user_id": "user_123",
                "action": "document_upload",
                "resource": "domain:finance",
                "ip_address": "192.168.1.1",
                "status": "success",
                "metadata": {"file_count": 3}
            }
        }


class AuditService:
    """Service for audit logging operations"""
    
    COLLECTION_NAME = "audit_logs"
    
    def __init__(self):
        self.collection = mongo_db[self.COLLECTION_NAME]
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary database indexes"""
        try:
            # Index on user_id and timestamp for user activity queries
            self.collection.create_index([("user_id", 1), ("timestamp", -1)])
            
            # Index on action and timestamp for action-based queries
            self.collection.create_index([("action", 1), ("timestamp", -1)])
            
            # Index on timestamp for time-based queries
            self.collection.create_index([("timestamp", -1)])
            
            # Index on status for filtering
            self.collection.create_index("status")
            
            logger.info("Audit log indexes created successfully")
        except Exception as e:
            logger.warning(f"Failed to create audit log indexes: {e}")
    
    def log(
        self,
        action: AuditAction,
        status: AuditStatus,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log an audit event
        
        Args:
            action: The action being audited
            status: Success or failure status
            user_id: Optional user identifier
            resource: Optional resource identifier (endpoint, document ID, etc.)
            ip_address: Optional client IP address
            user_agent: Optional client user agent
            metadata: Optional additional context
            error_message: Optional error message for failures
            
        Returns:
            Audit log ID
        """
        audit_log = AuditLog(
            action=action,
            status=status,
            user_id=user_id,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
            error_message=error_message
        )
        
        try:
            self.collection.insert_one(audit_log.dict())
            
            # Log to application logger as well
            log_level = logging.INFO if status == AuditStatus.SUCCESS else logging.WARNING
            logger.log(
                log_level,
                f"Audit: {action.value} - {status.value} - User: {user_id or 'anonymous'} - Resource: {resource or 'N/A'}"
            )
            
            return audit_log.audit_id
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # Don't fail the request if audit logging fails
            return ""
    
    def get_user_activity(
        self,
        user_id: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[AuditLog]:
        """
        Get audit logs for a specific user
        
        Args:
            user_id: User identifier
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            
        Returns:
            List of audit logs
        """
        logs = self.collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit).skip(skip)
        
        return [AuditLog(**log) for log in logs]
    
    def get_action_logs(
        self,
        action: AuditAction,
        limit: int = 100,
        skip: int = 0
    ) -> List[AuditLog]:
        """
        Get audit logs for a specific action type
        
        Args:
            action: Action type to filter by
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            
        Returns:
            List of audit logs
        """
        logs = self.collection.find(
            {"action": action.value}
        ).sort("timestamp", -1).limit(limit).skip(skip)
        
        return [AuditLog(**log) for log in logs]
    
    def get_security_events(
        self,
        limit: int = 100,
        skip: int = 0
    ) -> List[AuditLog]:
        """
        Get security-related audit logs (failures, rate limits, etc.)
        
        Args:
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            
        Returns:
            List of audit logs
        """
        security_actions = [
            AuditAction.AUTH_FAILURE.value,
            AuditAction.RATE_LIMIT_EXCEEDED.value,
            AuditAction.INVALID_TOKEN.value
        ]
        
        logs = self.collection.find(
            {"action": {"$in": security_actions}}
        ).sort("timestamp", -1).limit(limit).skip(skip)
        
        return [AuditLog(**log) for log in logs]
    
    def get_recent_logs(
        self,
        limit: int = 100,
        skip: int = 0,
        status: Optional[AuditStatus] = None
    ) -> List[AuditLog]:
        """
        Get recent audit logs
        
        Args:
            limit: Maximum number of logs to return
            skip: Number of logs to skip (for pagination)
            status: Optional status filter
            
        Returns:
            List of audit logs
        """
        query = {}
        if status:
            query["status"] = status.value
        
        logs = self.collection.find(query).sort("timestamp", -1).limit(limit).skip(skip)
        
        return [AuditLog(**log) for log in logs]


# Singleton instance
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create the audit service singleton"""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


def log_audit(
    action: AuditAction,
    status: AuditStatus = AuditStatus.SUCCESS,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict] = None,
    error_message: Optional[str] = None
) -> str:
    """
    Convenience function to log an audit event
    
    Usage:
        from ml_services.audit_service import log_audit, AuditAction, AuditStatus
        
        log_audit(
            action=AuditAction.DOCUMENT_UPLOAD,
            status=AuditStatus.SUCCESS,
            user_id=user.user_id,
            resource=f"domain:{domain}",
            metadata={"file_count": 3}
        )
    """
    service = get_audit_service()
    return service.log(
        action=action,
        status=status,
        user_id=user_id,
        resource=resource,
        ip_address=ip_address,
        user_agent=user_agent,
        metadata=metadata,
        error_message=error_message
    )
