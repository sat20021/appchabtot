"""
Template Manager Service for Deterministic First, Generative Last Implementation
Phase 1: Foundation & Intent Registry

This module implements template-based response generation with placeholder filling
and multi-language support.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pymongo import MongoClient
import asyncio
from jinja2 import Template, Environment, BaseLoader, TemplateNotFound

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Manages response templates with placeholder filling and multi-language support
    """
    
    def __init__(self, mongo_client: MongoClient, database_name: str):
        self.client = mongo_client
        self.db = mongo_client[database_name]
        self.templates_collection = self.db.response_templates
        self._template_cache = {}
        self._last_cache_update = None
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    async def get_response_template(
        self, 
        intent_id: str, 
        language: str = "en", 
        domain: str = "domain_2",
        version: str = "1.0"
    ) -> Optional[Dict[str, Any]]:
        """
        Get response template for a specific intent and language
        
        Args:
            intent_id: Intent identifier
            language: Language code (e.g., 'en', 'es', 'fr')
            domain: Domain context
            version: Template version
            
        Returns:
            Template data or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{intent_id}_{language}_{domain}_{version}"
            if self._is_cache_valid() and cache_key in self._template_cache:
                return self._template_cache[cache_key]
            
            # Fetch from database
            template = self.templates_collection.find_one({
                "intent_id": intent_id,
                "language": language,
                "domain": domain,
                "version": version,
                "is_active": True
            })
            
            if template:
                # Cache the result
                self._template_cache[cache_key] = template
                self._last_cache_update = datetime.utcnow()
                
                logger.debug(f"Retrieved template for intent: {intent_id}, language: {language}")
                return template
            else:
                logger.warning(f"No template found for intent: {intent_id}, language: {language}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving template for {intent_id}: {e}")
            return None
    
    async def render_response(
        self, 
        intent_id: str, 
        placeholders: Dict[str, Any], 
        language: str = "en",
        domain: str = "domain_2",
        version: str = "1.0"
    ) -> Dict[str, Any]:
        """
        Render a response using template and placeholders
        
        Args:
            intent_id: Intent identifier
            placeholders: Dictionary of placeholder values
            language: Language code
            domain: Domain context
            version: Template version
            
        Returns:
            Rendered response with metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Get template
            template_data = await self.get_response_template(intent_id, language, domain, version)
            
            if not template_data:
                return {
                    "success": False,
                    "error": f"No template found for intent: {intent_id}",
                    "rendered_text": None,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            # Render template
            template_text = template_data.get("template_text", "")
            rendered_text = self._render_template_text(template_text, placeholders)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "rendered_text": rendered_text,
                "template_id": template_data.get("template_id"),
                "intent_id": intent_id,
                "language": language,
                "domain": domain,
                "version": version,
                "placeholders_used": list(placeholders.keys()),
                "processing_time_ms": processing_time,
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"Rendered response for intent {intent_id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error rendering response for {intent_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "rendered_text": None,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    def _render_template_text(self, template_text: str, placeholders: Dict[str, Any]) -> str:
        """Render template text with placeholders using Jinja2"""
        try:
            # Create Jinja2 template
            template = self.jinja_env.from_string(template_text)
            
            # Render with placeholders
            rendered = template.render(**placeholders)
            
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template text: {e}")
            # Fallback to simple string replacement
            return self._simple_placeholder_replacement(template_text, placeholders)
    
    def _simple_placeholder_replacement(self, template_text: str, placeholders: Dict[str, Any]) -> str:
        """Fallback simple placeholder replacement"""
        try:
            result = template_text
            
            # Replace placeholders in format {placeholder_name}
            for key, value in placeholders.items():
                placeholder = "{" + key + "}"
                result = result.replace(placeholder, str(value))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simple placeholder replacement: {e}")
            return template_text
    
    async def add_template(self, template_data: Dict[str, Any]) -> bool:
        """Add a new response template"""
        try:
            # Validate required fields
            required_fields = ["template_id", "intent_id", "language", "template_text"]
            for field in required_fields:
                if field not in template_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Add metadata
            template_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            })
            
            # Insert template
            result = self.templates_collection.insert_one(template_data)
            
            # Clear cache
            self._clear_cache()
            
            logger.info(f"Added new template: {template_data.get('template_id')}")
            return result.inserted_id is not None
            
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
    
    async def update_template(
        self, 
        template_id: str, 
        update_data: Dict[str, Any]
    ) -> bool:
        """Update an existing template"""
        try:
            update_data["updated_at"] = datetime.utcnow()
            
            result = self.templates_collection.update_one(
                {"template_id": template_id},
                {"$set": update_data}
            )
            
            # Clear cache
            self._clear_cache()
            
            logger.info(f"Updated template: {template_id}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating template {template_id}: {e}")
            return False
    
    async def get_templates_for_intent(
        self, 
        intent_id: str, 
        domain: str = "domain_2"
    ) -> List[Dict[str, Any]]:
        """Get all templates for a specific intent"""
        try:
            templates = list(self.templates_collection.find({
                "intent_id": intent_id,
                "domain": domain,
                "is_active": True
            }).sort("language", 1))
            
            return templates
            
        except Exception as e:
            logger.error(f"Error fetching templates for intent {intent_id}: {e}")
            return []
    
    async def get_available_languages(self, intent_id: str, domain: str = "domain_2") -> List[str]:
        """Get available languages for a specific intent"""
        try:
            templates = await self.get_templates_for_intent(intent_id, domain)
            languages = [template["language"] for template in templates]
            return list(set(languages))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error fetching available languages for {intent_id}: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._last_cache_update:
            return False
        
        current_time = datetime.utcnow()
        return (current_time - self._last_cache_update).total_seconds() < self.cache_ttl
    
    def _clear_cache(self):
        """Clear template cache"""
        self._template_cache.clear()
        self._last_cache_update = None
        logger.info("Template cache cleared")
    
    async def validate_template(self, template_text: str, placeholders: List[str]) -> Dict[str, Any]:
        """Validate template syntax and placeholder usage"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "placeholders_found": [],
                "placeholders_missing": []
            }
            
            # Check for Jinja2 syntax errors
            try:
                template = self.jinja_env.from_string(template_text)
                validation_result["placeholders_found"] = self._extract_placeholders(template_text)
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Template syntax error: {str(e)}")
            
            # Check for missing placeholders
            found_placeholders = set(validation_result["placeholders_found"])
            required_placeholders = set(placeholders)
            
            missing_placeholders = required_placeholders - found_placeholders
            if missing_placeholders:
                validation_result["warnings"].append(
                    f"Missing placeholders: {list(missing_placeholders)}"
                )
                validation_result["placeholders_missing"] = list(missing_placeholders)
            
            # Check for unused placeholders
            unused_placeholders = found_placeholders - required_placeholders
            if unused_placeholders:
                validation_result["warnings"].append(
                    f"Unused placeholders: {list(unused_placeholders)}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
                "placeholders_found": [],
                "placeholders_missing": []
            }
    
    def _extract_placeholders(self, template_text: str) -> List[str]:
        """Extract placeholder names from template text"""
        # Find placeholders in format {placeholder_name}
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template_text)
        return matches

# Factory function for creating template manager
def create_template_manager(mongo_client: MongoClient, database_name: str) -> TemplateManager:
    """Create and return a TemplateManager instance"""
    return TemplateManager(mongo_client, database_name)

# Async wrapper functions
async def render_response_async(
    template_manager: TemplateManager,
    intent_id: str,
    placeholders: Dict[str, Any],
    language: str = "en",
    domain: str = "domain_2",
    version: str = "1.0"
) -> Dict[str, Any]:
    """Async wrapper for response rendering"""
    return await template_manager.render_response(
        intent_id, placeholders, language, domain, version
    )

async def get_template_async(
    template_manager: TemplateManager,
    intent_id: str,
    language: str = "en",
    domain: str = "domain_2",
    version: str = "1.0"
) -> Optional[Dict[str, Any]]:
    """Async wrapper for template retrieval"""
    return await template_manager.get_response_template(
        intent_id, language, domain, version
    )
