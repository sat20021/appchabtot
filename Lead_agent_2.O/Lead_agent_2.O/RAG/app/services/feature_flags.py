"""
Feature Flags Service for Deterministic First, Generative Last Implementation
Phase 1: Foundation & Intent Registry

This module implements feature flag system for safe deployment and rollback.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
import os

logger = logging.getLogger(__name__)

class FeatureFlags:
    """
    Feature flag manager for controlling system behavior
    """
    
    def __init__(self, mongo_client: MongoClient, database_name: str):
        self.client = mongo_client
        self.db = mongo_client[database_name]
        self.flags_collection = self.db.feature_flags
        self._flags_cache = {}
        self._last_cache_update = None
        self.cache_ttl = 60  # 1 minute cache TTL
        
        # Initialize default flags
        self._initialize_default_flags()
    
    def _initialize_default_flags(self):
        """Initialize default feature flags"""
        default_flags = {
            "intent_classification_enabled": {
                "value": os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true",
                "description": "Enable intent classification system",
                "environment": "all"
            },
            "template_responses_enabled": {
                "value": os.getenv("TEMPLATE_RESPONSES_ENABLED", "true").lower() == "true",
                "description": "Enable template-based responses",
                "environment": "all"
            },
            "deterministic_routing_enabled": {
                "value": os.getenv("DETERMINISTIC_ROUTING_ENABLED", "true").lower() == "true",
                "description": "Enable deterministic routing",
                "environment": "all"
            },
            "fallback_to_rag_enabled": {
                "value": os.getenv("FALLBACK_TO_RAG_ENABLED", "true").lower() == "true",
                "description": "Enable fallback to RAG system",
                "environment": "all"
            },
            "performance_monitoring_enabled": {
                "value": os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true",
                "description": "Enable performance monitoring",
                "environment": "all"
            },
            "debug_mode": {
                "value": os.getenv("DEBUG_MODE", "false").lower() == "true",
                "description": "Enable debug mode",
                "environment": "development"
            }
        }
        
        # Store default flags in cache
        self._flags_cache = default_flags.copy()
        self._last_cache_update = datetime.utcnow()
    
    def is_enabled(self, flag_name: str, environment: str = "production") -> bool:
        """
        Check if a feature flag is enabled
        
        Args:
            flag_name: Name of the feature flag
            environment: Environment context (development, staging, production)
            
        Returns:
            True if flag is enabled, False otherwise
        """
        try:
            # Check cache first
            if self._is_cache_valid() and flag_name in self._flags_cache:
                flag_data = self._flags_cache[flag_name]
                
                # Check environment restriction
                flag_env = flag_data.get("environment", "all")
                if flag_env != "all" and flag_env != environment:
                    return False
                
                return flag_data.get("value", False)
            
            # Fetch from database if not in cache
            flag_data = self.flags_collection.find_one({
                "flag_name": flag_name,
                "is_active": True
            })
            
            if flag_data:
                # Cache the result
                self._flags_cache[flag_name] = {
                    "value": flag_data.get("value", False),
                    "description": flag_data.get("description", ""),
                    "environment": flag_data.get("environment", "all")
                }
                self._last_cache_update = datetime.utcnow()
                
                # Check environment restriction
                flag_env = flag_data.get("environment", "all")
                if flag_env != "all" and flag_env != environment:
                    return False
                
                return flag_data.get("value", False)
            
            # Return default value if not found
            return self._flags_cache.get(flag_name, {}).get("value", False)
            
        except Exception as e:
            logger.error(f"Error checking feature flag {flag_name}: {e}")
            # Return safe default (disabled) on error
            return False
    
    def get_flag_value(self, flag_name: str, default_value: Any = None) -> Any:
        """
        Get the value of a feature flag
        
        Args:
            flag_name: Name of the feature flag
            default_value: Default value if flag not found
            
        Returns:
            Flag value or default value
        """
        try:
            if self._is_cache_valid() and flag_name in self._flags_cache:
                return self._flags_cache[flag_name].get("value", default_value)
            
            # Fetch from database
            flag_data = self.flags_collection.find_one({
                "flag_name": flag_name,
                "is_active": True
            })
            
            if flag_data:
                return flag_data.get("value", default_value)
            
            return default_value
            
        except Exception as e:
            logger.error(f"Error getting feature flag value {flag_name}: {e}")
            return default_value
    
    def set_flag(self, flag_name: str, value: Any, description: str = "", environment: str = "all") -> bool:
        """
        Set a feature flag value
        
        Args:
            flag_name: Name of the feature flag
            value: Flag value
            description: Flag description
            environment: Environment restriction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            flag_data = {
                "flag_name": flag_name,
                "value": value,
                "description": description,
                "environment": environment,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = self.flags_collection.update_one(
                {"flag_name": flag_name},
                {"$set": flag_data},
                upsert=True
            )
            
            # Update cache
            self._flags_cache[flag_name] = {
                "value": value,
                "description": description,
                "environment": environment
            }
            self._last_cache_update = datetime.utcnow()
            
            logger.info(f"Set feature flag {flag_name} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting feature flag {flag_name}: {e}")
            return False
    
    def disable_flag(self, flag_name: str) -> bool:
        """Disable a feature flag"""
        try:
            result = self.flags_collection.update_one(
                {"flag_name": flag_name},
                {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
            )
            
            # Remove from cache
            if flag_name in self._flags_cache:
                del self._flags_cache[flag_name]
            
            logger.info(f"Disabled feature flag {flag_name}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error disabling feature flag {flag_name}: {e}")
            return False
    
    def get_all_flags(self, environment: str = "all") -> Dict[str, Any]:
        """Get all feature flags for an environment"""
        try:
            flags = {}
            
            # Get from cache first
            if self._is_cache_valid():
                for flag_name, flag_data in self._flags_cache.items():
                    flag_env = flag_data.get("environment", "all")
                    if flag_env == "all" or flag_env == environment:
                        flags[flag_name] = flag_data
            
            # Get from database
            db_flags = self.flags_collection.find({
                "is_active": True
            })
            
            for flag_data in db_flags:
                flag_name = flag_data["flag_name"]
                flag_env = flag_data.get("environment", "all")
                
                if flag_env == "all" or flag_env == environment:
                    flags[flag_name] = {
                        "value": flag_data.get("value"),
                        "description": flag_data.get("description", ""),
                        "environment": flag_env
                    }
            
            return flags
            
        except Exception as e:
            logger.error(f"Error getting all feature flags: {e}")
            return {}
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._last_cache_update:
            return False
        
        current_time = datetime.utcnow()
        return (current_time - self._last_cache_update).total_seconds() < self.cache_ttl
    
    def clear_cache(self):
        """Clear feature flag cache"""
        self._flags_cache.clear()
        self._last_cache_update = None
        logger.info("Feature flag cache cleared")

# Global feature flags instance
_feature_flags_instance = None

def get_feature_flags(mongo_client: MongoClient = None, database_name: str = None) -> FeatureFlags:
    """Get or create global feature flags instance"""
    global _feature_flags_instance
    
    if _feature_flags_instance is None:
        if mongo_client is None or database_name is None:
            raise ValueError("MongoDB client and database name are required for first initialization")
        
        _feature_flags_instance = FeatureFlags(mongo_client, database_name)
    
    return _feature_flags_instance

# Convenience functions
def is_intent_classification_enabled(environment: str = "production") -> bool:
    """Check if intent classification is enabled"""
    try:
        flags = get_feature_flags()
        return flags.is_enabled("intent_classification_enabled", environment)
    except:
        return False

def is_template_responses_enabled(environment: str = "production") -> bool:
    """Check if template responses are enabled"""
    try:
        flags = get_feature_flags()
        return flags.is_enabled("template_responses_enabled", environment)
    except:
        return False

def is_deterministic_routing_enabled(environment: str = "production") -> bool:
    """Check if deterministic routing is enabled"""
    try:
        flags = get_feature_flags()
        return flags.is_enabled("deterministic_routing_enabled", environment)
    except:
        return False

def is_fallback_to_rag_enabled(environment: str = "production") -> bool:
    """Check if fallback to RAG is enabled"""
    try:
        flags = get_feature_flags()
        return flags.is_enabled("fallback_to_rag_enabled", environment)
    except:
        return True  # Default to enabled for safety
