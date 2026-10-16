"""
Intent Classification Service for Deterministic First, Generative Last Implementation
Phase 1: Foundation & Intent Registry

This module implements rule-based intent classification with keyword matching and regex patterns.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pymongo import MongoClient
import asyncio

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Rule-based intent classifier that uses keyword matching and regex patterns
    to classify user queries into predefined intents.
    """
    
    def __init__(self, mongo_client: MongoClient, database_name: str):
        self.client = mongo_client
        self.db = mongo_client[database_name]
        self.intents_collection = self.db.intents_registry
        self._intent_cache = {}
        self._last_cache_update = None
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    async def classify_intent(self, query: str, domain: str, user_email: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify user query into an intent with confidence score
        
        Args:
            query: User's input query
            domain: Domain context (domain_1, domain_2, domain_3)
            user_email: User's email for personalization
            
        Returns:
            Dict containing intent classification results
        """
        start_time = datetime.utcnow()
        
        try:
            # Normalize query
            normalized_query = self._normalize_query(query)
            
            # Get intents for domain
            intents = await self._get_intents_for_domain(domain)
            
            # Classify using rule-based approach
            classification_result = self._classify_with_rules(normalized_query, intents)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Add metadata
            classification_result.update({
                "processing_time_ms": processing_time,
                "method": "rule_based",
                "domain": domain,
                "timestamp": start_time.isoformat(),
                "user_email": user_email
            })
            
            # Log classification result
            logger.info(f"Intent classified: {classification_result['intent_id']} "
                       f"(confidence: {classification_result['confidence']:.2f}) "
                       f"in {processing_time:.2f}ms")
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                "intent_id": "unknown",
                "intent_name": "Unknown Intent",
                "confidence": 0.0,
                "method": "rule_based",
                "domain": domain,
                "error": str(e),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "timestamp": start_time.isoformat()
            }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        if not query:
            return ""
        
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep alphanumeric and spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        return normalized
    
    async def _get_intents_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get intents for a specific domain with caching"""
        current_time = datetime.utcnow()
        
        # Check cache validity
        if (self._last_cache_update and 
            (current_time - self._last_cache_update).total_seconds() < self.cache_ttl and
            domain in self._intent_cache):
            return self._intent_cache[domain]
        
        try:
            # Fetch intents from database
            intents = list(self.intents_collection.find({
                "domain": domain,
                "is_active": True
            }))
            
            # Cache the results
            self._intent_cache[domain] = intents
            self._last_cache_update = current_time
            
            logger.debug(f"Loaded {len(intents)} intents for domain {domain}")
            return intents
            
        except Exception as e:
            logger.error(f"Error fetching intents for domain {domain}: {e}")
            return []
    
    def _classify_with_rules(self, query: str, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify query using rule-based approach"""
        if not query or not intents:
            return {
                "intent_id": "unknown",
                "intent_name": "Unknown Intent",
                "confidence": 0.0,
                "matched_keywords": [],
                "matched_patterns": []
            }
        
        best_match = None
        best_score = 0.0
        matched_keywords = []
        matched_patterns = []
        
        for intent in intents:
            score = 0.0
            intent_keywords = []
            intent_patterns = []
            
            # Keyword matching
            keywords = intent.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query:
                    score += 1.0
                    intent_keywords.append(keyword)
            
            # Normalize score by number of keywords
            if keywords:
                keyword_score = score / len(keywords)
            else:
                keyword_score = 0.0
            
            # Regex pattern matching (if patterns exist)
            patterns = intent.get("patterns", [])
            pattern_score = 0.0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_score += 1.0
                    intent_patterns.append(pattern)
            
            # Normalize pattern score
            if patterns:
                pattern_score = pattern_score / len(patterns)
            
            # Combined score (weighted: 70% keywords, 30% patterns)
            combined_score = (keyword_score * 0.7) + (pattern_score * 0.3)
            
            # Boost score for exact matches
            if any(keyword.lower() == query for keyword in keywords):
                combined_score += 0.2
            
            # Boost score for multiple keyword matches
            if len(intent_keywords) > 1:
                combined_score += 0.1
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = intent
                matched_keywords = intent_keywords
                matched_patterns = intent_patterns
        
        # Determine confidence and intent
        if best_match and best_score >= 0.3:  # Minimum threshold
            confidence = min(best_score, 1.0)
            return {
                "intent_id": best_match["intent_id"],
                "intent_name": best_match["intent_name"],
                "confidence": confidence,
                "is_deterministic": best_match.get("is_deterministic", False),
                "confidence_threshold": best_match.get("confidence_threshold", 0.8),
                "required_entities": best_match.get("required_entities", []),
                "matched_keywords": matched_keywords,
                "matched_patterns": matched_patterns,
                "clarifying_prompts": best_match.get("clarifying_prompts", {}),
                "allowed_tools": best_match.get("allowed_tools", [])
            }
        else:
            return {
                "intent_id": "unknown",
                "intent_name": "Unknown Intent",
                "confidence": best_score,
                "is_deterministic": False,
                "confidence_threshold": 0.0,
                "required_entities": [],
                "matched_keywords": matched_keywords,
                "matched_patterns": matched_patterns,
                "clarifying_prompts": {},
                "allowed_tools": []
            }
    
    async def get_intent_details(self, intent_id: str, domain: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific intent"""
        try:
            intent = self.intents_collection.find_one({
                "intent_id": intent_id,
                "domain": domain,
                "is_active": True
            })
            return intent
        except Exception as e:
            logger.error(f"Error fetching intent details for {intent_id}: {e}")
            return None
    
    async def add_intent(self, intent_data: Dict[str, Any]) -> bool:
        """Add a new intent to the registry"""
        try:
            intent_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            })
            
            result = self.intents_collection.insert_one(intent_data)
            
            # Clear cache to force refresh
            self._intent_cache.clear()
            self._last_cache_update = None
            
            logger.info(f"Added new intent: {intent_data.get('intent_id')}")
            return result.inserted_id is not None
            
        except Exception as e:
            logger.error(f"Error adding intent: {e}")
            return False
    
    async def update_intent(self, intent_id: str, domain: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing intent"""
        try:
            update_data["updated_at"] = datetime.utcnow()
            
            result = self.intents_collection.update_one(
                {"intent_id": intent_id, "domain": domain},
                {"$set": update_data}
            )
            
            # Clear cache to force refresh
            self._intent_cache.clear()
            self._last_cache_update = None
            
            logger.info(f"Updated intent: {intent_id}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating intent {intent_id}: {e}")
            return False
    
    async def get_all_intents(self, domain: str) -> List[Dict[str, Any]]:
        """Get all intents for a domain"""
        try:
            intents = list(self.intents_collection.find({
                "domain": domain,
                "is_active": True
            }).sort("intent_name", 1))
            return intents
        except Exception as e:
            logger.error(f"Error fetching all intents for domain {domain}: {e}")
            return []
    
    def clear_cache(self):
        """Clear the intent cache"""
        self._intent_cache.clear()
        self._last_cache_update = None
        logger.info("Intent cache cleared")

# Factory function for creating intent classifier
def create_intent_classifier(mongo_client: MongoClient, database_name: str) -> IntentClassifier:
    """Create and return an IntentClassifier instance"""
    return IntentClassifier(mongo_client, database_name)

# Async wrapper for synchronous operations
async def classify_intent_async(
    classifier: IntentClassifier, 
    query: str, 
    domain: str, 
    user_email: Optional[str] = None
) -> Dict[str, Any]:
    """Async wrapper for intent classification"""
    return await classifier.classify_intent(query, domain, user_email)
