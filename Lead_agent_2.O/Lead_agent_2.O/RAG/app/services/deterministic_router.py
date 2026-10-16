"""
Deterministic Router Service for Deterministic First, Generative Last Implementation
Phase 1: Foundation & Intent Registry

This module integrates intent classification and template management to provide
deterministic routing for user queries.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pymongo import MongoClient
import asyncio

from .intent_classifier import IntentClassifier, create_intent_classifier
from .template_manager import TemplateManager, create_template_manager
from .feature_flags import get_feature_flags, is_intent_classification_enabled, is_template_responses_enabled

logger = logging.getLogger(__name__)

class DeterministicRouter:
    """
    Main router that handles deterministic query processing
    """
    
    def __init__(self, mongo_client: MongoClient, database_name: str):
        self.client = mongo_client
        self.db = mongo_client[database_name]
        
        # Initialize services
        self.intent_classifier = create_intent_classifier(mongo_client, database_name)
        self.template_manager = create_template_manager(mongo_client, database_name)
        self.feature_flags = get_feature_flags(mongo_client, database_name)
        
        # Performance metrics collection
        self.metrics_collection = self.db.performance_metrics
    
    async def process_query(
        self, 
        query: str, 
        domain: str, 
        user_email: Optional[str] = None,
        language: str = "en",
        environment: str = "production"
    ) -> Dict[str, Any]:
        """
        Process a user query using deterministic routing
        
        Args:
            query: User's input query
            domain: Domain context (domain_1, domain_2, domain_3)
            user_email: User's email for personalization
            language: Language code
            environment: Environment context
            
        Returns:
            Processing result with response or routing decision
        """
        start_time = datetime.utcnow()
        
        try:
            # Check if deterministic routing is enabled
            if not self.feature_flags.is_enabled("deterministic_routing_enabled", environment):
                return {
                    "should_use_deterministic": False,
                    "reason": "deterministic_routing_disabled",
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            # Step 1: Intent Classification
            intent_result = await self._classify_intent(query, domain, user_email, environment)
            
            if not intent_result["success"]:
                return {
                    "should_use_deterministic": False,
                    "reason": "intent_classification_failed",
                    "error": intent_result.get("error"),
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            intent_data = intent_result["data"]
            confidence = intent_data["confidence"]
            confidence_threshold = intent_data.get("confidence_threshold", 0.8)
            
            # Step 2: Check if confidence meets threshold for deterministic response
            if confidence < confidence_threshold:
                return {
                    "should_use_deterministic": False,
                    "reason": "low_confidence",
                    "confidence": confidence,
                    "threshold": confidence_threshold,
                    "intent_id": intent_data["intent_id"],
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            # Step 3: Check if intent is deterministic
            if not intent_data.get("is_deterministic", False):
                return {
                    "should_use_deterministic": False,
                    "reason": "non_deterministic_intent",
                    "intent_id": intent_data["intent_id"],
                    "confidence": confidence,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            # Step 4: Check for required entities
            required_entities = intent_data.get("required_entities", [])
            if required_entities:
                missing_entities = await self._check_required_entities(query, required_entities)
                if missing_entities:
                    return {
                        "should_use_deterministic": False,
                        "reason": "missing_required_entities",
                        "intent_id": intent_data["intent_id"],
                        "missing_entities": missing_entities,
                        "clarifying_prompts": intent_data.get("clarifying_prompts", {}),
                        "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                    }
            
            # Step 5: Generate deterministic response
            response_result = await self._generate_deterministic_response(
                intent_data, query, domain, user_email, language, environment
            )
            
            # Step 6: Record performance metrics
            await self._record_metrics(
                intent_data["intent_id"], domain, "deterministic", 
                (datetime.utcnow() - start_time).total_seconds() * 1000,
                confidence, user_email
            )
            
            return {
                "should_use_deterministic": True,
                "response": response_result,
                "intent_id": intent_data["intent_id"],
                "confidence": confidence,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            logger.error(f"Error in deterministic routing: {e}")
            return {
                "should_use_deterministic": False,
                "reason": "processing_error",
                "error": str(e),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def _classify_intent(
        self, 
        query: str, 
        domain: str, 
        user_email: Optional[str], 
        environment: str
    ) -> Dict[str, Any]:
        """Classify user intent"""
        try:
            if not is_intent_classification_enabled(environment):
                return {
                    "success": False,
                    "error": "Intent classification disabled"
                }
            
            intent_result = await self.intent_classifier.classify_intent(query, domain, user_email)
            
            return {
                "success": True,
                "data": intent_result
            }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_required_entities(
        self, 
        query: str, 
        required_entities: List[str]
    ) -> List[str]:
        """Check if required entities are present in the query"""
        missing_entities = []
        
        # Simple entity extraction (can be enhanced later)
        query_lower = query.lower()
        
        for entity in required_entities:
            if entity == "order_number":
                # Check for order number patterns
                if not any(pattern in query_lower for pattern in ["ord-", "order-", "#"]):
                    missing_entities.append(entity)
            elif entity == "product_type":
                # Check for product mentions
                product_keywords = ["web", "mobile", "app", "consulting", "development"]
                if not any(keyword in query_lower for keyword in product_keywords):
                    missing_entities.append(entity)
            # Add more entity checks as needed
        
        return missing_entities
    
    async def _generate_deterministic_response(
        self, 
        intent_data: Dict[str, Any], 
        query: str, 
        domain: str, 
        user_email: Optional[str],
        language: str,
        environment: str
    ) -> Dict[str, Any]:
        """Generate deterministic response using templates"""
        try:
            if not is_template_responses_enabled(environment):
                return {
                    "success": False,
                    "error": "Template responses disabled"
                }
            
            intent_id = intent_data["intent_id"]
            
            # Extract placeholders from query (simplified)
            placeholders = await self._extract_placeholders(query, intent_data)
            
            # Render template response
            response_result = await self.template_manager.render_response(
                intent_id, placeholders, language, domain
            )
            
            if response_result["success"]:
                return {
                    "success": True,
                    "response_text": response_result["rendered_text"],
                    "template_id": response_result.get("template_id"),
                    "placeholders_used": response_result.get("placeholders_used", [])
                }
            else:
                return {
                    "success": False,
                    "error": response_result.get("error", "Template rendering failed")
                }
                
        except Exception as e:
            logger.error(f"Error generating deterministic response: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_placeholders(
        self, 
        query: str, 
        intent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract placeholder values from query"""
        placeholders = {}
        
        # Simple placeholder extraction (can be enhanced with NLP)
        query_lower = query.lower()
        
        # Extract order number
        if "order_number" in intent_data.get("required_entities", []):
            import re
            order_match = re.search(r'(ord-|order-|#)(\d+)', query_lower)
            if order_match:
                placeholders["order_number"] = order_match.group(0).upper()
        
        # Extract product type
        if "product_type" in intent_data.get("required_entities", []):
            product_mapping = {
                "web": "web_development",
                "mobile": "mobile_app",
                "app": "mobile_app",
                "consulting": "consulting",
                "maintenance": "maintenance"
            }
            
            for keyword, product_type in product_mapping.items():
                if keyword in query_lower:
                    placeholders["product_type"] = product_type
                    break
        
        # Add default values for common placeholders
        placeholders.update({
            "status": "In Progress",  # Default status
            "additional_info": "Please contact support for more details.",
            "pricing_details": "Please contact our sales team for detailed pricing.",
            "issue_type": "general",
            "ticket_id": f"TKT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "date_range": "this month",
            "total_hours": "40",
            "status": "Active"
        })
        
        return placeholders
    
    async def _record_metrics(
        self, 
        intent_id: str, 
        domain: str, 
        response_type: str, 
        latency_ms: float,
        confidence: float,
        user_email: Optional[str]
    ):
        """Record performance metrics"""
        try:
            if not self.feature_flags.is_enabled("performance_monitoring_enabled"):
                return
            
            metrics_data = {
                "timestamp": datetime.utcnow(),
                "intent_id": intent_id,
                "domain": domain,
                "response_type": response_type,
                "latency_ms": latency_ms,
                "confidence_score": confidence,
                "user_email": user_email
            }
            
            self.metrics_collection.insert_one(metrics_data)
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    async def get_clarifying_question(
        self, 
        intent_id: str, 
        missing_entities: List[str],
        clarifying_prompts: Dict[str, str]
    ) -> Optional[str]:
        """Generate clarifying question for missing entities"""
        try:
            if not missing_entities or not clarifying_prompts:
                return None
            
            # Use the first missing entity's prompt
            first_entity = missing_entities[0]
            prompt_key = f"missing_{first_entity}"
            
            if prompt_key in clarifying_prompts:
                return clarifying_prompts[prompt_key]
            
            # Fallback to generic prompt
            return f"Could you please provide more information about {first_entity}?"
            
        except Exception as e:
            logger.error(f"Error generating clarifying question: {e}")
            return None

# Factory function
def create_deterministic_router(mongo_client: MongoClient, database_name: str) -> DeterministicRouter:
    """Create and return a DeterministicRouter instance"""
    return DeterministicRouter(mongo_client, database_name)

# Async wrapper functions
async def process_query_async(
    router: DeterministicRouter,
    query: str,
    domain: str,
    user_email: Optional[str] = None,
    language: str = "en",
    environment: str = "production"
) -> Dict[str, Any]:
    """Async wrapper for query processing"""
    return await router.process_query(query, domain, user_email, language, environment)
