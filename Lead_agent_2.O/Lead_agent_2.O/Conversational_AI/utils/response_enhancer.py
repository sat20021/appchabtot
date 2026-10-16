"""
Response Enhancer - Two-LLM Architecture for Rich, Safe Responses

Architecture:
1. Base Question (from question_generation_node) - Validated, safe
2. Enhancer LLM - Adds context and personality WITHOUT changing the core question

This prevents hallucination because:
- The core question is already validated
- The enhancer can only ADD context around it
- The enhancer MUST include the original question verbatim
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

# Add parent paths for imports
# Structure: Lead_agent_2.O/Conversational_AI/utils/response_enhancer.py
_UTILS_DIR = Path(__file__).resolve().parent
_CONVERSATIONAL_AI_DIR = _UTILS_DIR.parent
_LEAD_AGENT_DIR = _CONVERSATIONAL_AI_DIR.parent
_AGENT_DIR = _CONVERSATIONAL_AI_DIR / "agent"

if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))
if str(_LEAD_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_LEAD_AGENT_DIR))

logger = logging.getLogger(__name__)


def query_rag_for_context(
    user_message: str,
    domain: str = "college",
    max_results: int = 2
) -> str:
    """
    DEPRECATED: Proactive RAG queries should be handled by the orchestrator (nodes.py).
    This function is kept as a robust fallback but logs a warning.
    """
    try:
        # ⚠️ ARCHITECTURE WARNING: This path initializes a NEW RAG service instance
        # which may be out of sync with the main application.
        # Prefer injecting 'rag_context' into enhance_response().
        logger.warning("⚠️ [Response Enhancer] DEPRECATED internal RAG query triggered. Context should be injected by orchestrator.")
        
        logger.info(f"📚 Querying RAG for context (domain={domain})...")
        logger.info(f"   Query: {user_message[:100]}...")
        
        # Import RAG service
        import sys
        from pathlib import Path
        
        rag_path = Path(__file__).resolve().parents[2] / "RAG" / "app" / "services"
        if str(rag_path) not in sys.path:
            sys.path.insert(0, str(rag_path))
        
        from document_query import get_document_query_service
        
        # Get RAG service
        query_service = get_document_query_service()
        
        # Query RAG
        results = query_service.search(
            query=user_message,
            domain=domain,
            top_k=max_results,
            threshold=0.5
        )
        
        if not results:
            logger.info("   RAG Results: No results found")
            return ""
        
        # Format results
        context_parts = []
        for result in results:
            # Extract intent from metadata if available
            intent = result.metadata.get("intent", "general") if result.metadata else "general"
            content = result.content[:300]  # Limit content length
            context_parts.append(f"[{intent}] {content}")
        
        formatted_context = "\n".join(context_parts)
        
        logger.info(f"   RAG Results: {len(results)} chunks ({len(formatted_context)} chars)")
        logger.info(f"   RAG Snippet: {formatted_context[:200]}...")
        
        return formatted_context
        
    except Exception as e:
        logger.warning(f"❌ RAG query failed: {e}")
        return ""


def enhance_response(
    base_question: str,
    persona: Dict,
    conversation_history: List[Dict],
    collected_data: Dict,
    current_key: str,
    user_last_message: Optional[str] = None,
    domain: str = "college",
    rag_context: Optional[str] = None,
    original_user_inquiry: Optional[str] = None,
    off_topic_handled: bool = False,
) -> str:
    """
    Enhance a validated base question with context and personality.
    
    Args:
        base_question: The validated question (MUST be included in output)
        persona: Persona configuration with name, instructions
        conversation_history: Last 3 conversation turns
        collected_data: Data already collected from user
        current_key: The field being asked (email, phone, name, etc.)
        user_last_message: The user's last message (to acknowledge)
        domain: Domain/tenant ID for RAG queries
        rag_context: Pre-fetched RAG context (if None, will query RAG)
        original_user_inquiry: User's FIRST inquiry (e.g., "Swift development") for personalization
        off_topic_handled: If True, user asked an off-topic question that needs to be answered first
    
    Returns:
        Enhanced response that includes the base question
    """
    # ========== COMPREHENSIVE LOGGING START ==========
    logger.info("="*80)
    logger.info(f"🎨 RESPONSE ENHANCER TRIGGERED for field: {current_key}")
    logger.info("="*80)
    
    # Query RAG if not provided - use original inquiry as fallback
    if rag_context is None:
        # Priority: user's last message > original inquiry > generic fallback
        query_text = user_last_message or original_user_inquiry or "academic programs admissions campus life"
        if query_text:
            rag_context = query_rag_for_context(query_text, domain)
            if rag_context:
                logger.info(f"✅ RAG QUERIED: Found {len(rag_context)} chars for '{query_text[:50]}...'")
    
    # Log all inputs
    logger.info(f"📝 ENHANCER INPUTS:")
    logger.info(f"   Base Question: {base_question}")
    logger.info(f"   Persona: {persona.get('name', 'Unknown')}")
    logger.info(f"   Domain: {domain}")
    logger.info(f"   User's Last Message: {user_last_message or 'N/A'}")
    logger.info(f"   Conversation History: {len(conversation_history)} messages")
    logger.info(f"   Collected Data: {list(collected_data.keys())}")
    logger.info(f"   RAG Context: {'Yes (%d chars)' % len(rag_context) if rag_context else 'No'}")
    logger.info(f"   Original User Inquiry: {original_user_inquiry or 'None'}")

    
    # Log persona formatting guidelines
    formatting_style = persona.get('formatting_style', {})
    formatting_guidelines = formatting_style.get('guidelines', [])
    logger.info(f"   Persona Formatting Guidelines: {len(formatting_guidelines)} rules")
    if formatting_guidelines:
        logger.info(f"   Sample: {formatting_guidelines[0]}")
    # ========== COMPREHENSIVE LOGGING END ==========
    
    # Dynamic import to handle different execution contexts
    try:
        from agent.llm_gateway import LLMGateway
    except ImportError:
        try:
            from llm_gateway import LLMGateway
        except ImportError:
            import importlib.util
            llm_gateway_path = _AGENT_DIR / "llm_gateway.py"
            spec = importlib.util.spec_from_file_location("llm_gateway", llm_gateway_path)
            llm_gateway_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_gateway_module)
            LLMGateway = llm_gateway_module.LLMGateway
    
    # Build context summary
    context_parts = []
    if collected_data:
        for key, value in collected_data.items():
            context_parts.append(f"- {key}: {value}")
    context_summary = "\n".join(context_parts) if context_parts else "Nothing collected yet"
    
    # Build history summary (last 3 turns max)
    history_parts = []
    for turn in conversation_history[-6:]:  # 3 turns = 6 messages
        role = turn.get("role", turn.get("type", "unknown"))
        content = turn.get("content", turn.get("message", ""))
        if role in ["user", "human"]:
            history_parts.append(f"User: {content}")
        elif role in ["assistant", "ai"]:
            history_parts.append(f"You: {content}")
    history_summary = "\n".join(history_parts) if history_parts else "Starting conversation"
    
    # Persona info
    persona_name = persona.get("name", "Sara")
    persona_instructions = persona.get("instructions", "Be friendly and helpful")
    
    # Construct the enhancement prompt
    # Extract formatting guidelines from persona
    formatting_style = persona.get('formatting_style', {})
    formatting_guidelines = formatting_style.get('guidelines', [])
    
    # 🎯 Extract field name from current_key (more reliable than parsing text)
    field_name = current_key.replace("_", " ") if current_key else "information"
    
    # Override for specific keys if needed for cleaner prompting
    if current_key == "interest":
        field_name = "specific interest or project goal"
    elif current_key == "source":
        field_name = "how you found us"
    
    # Build user topic context
    user_topic = original_user_inquiry or user_last_message or ""
    user_topic_snippet = user_topic[:100] if user_topic else "academic programs"
    
    # 🔴 FIX: Use original_user_inquiry as fallback when user_last_message is empty
    effective_user_message = user_last_message or original_user_inquiry or ""
    
    # 🔴 NEW: Extract actual values from collected_data for acknowledgment
    last_collected_field = None
    last_collected_value = None
    if collected_data:
        # Get the most recently collected field
        for key in ['email', 'phone', 'name', 'company_name']:
            if key in collected_data:
                last_collected_field = key
                last_collected_value = collected_data[key]
    
    # Build acknowledgment hint for the LLM
    ack_hint = ""
    if last_collected_value:
        ack_hint = f"User just provided their {last_collected_field}: {last_collected_value}"
    
    # 🔴 NEW: Build list of already collected fields for guardrail
    already_collected = []
    if collected_data:
        for key, value in collected_data.items():
            already_collected.append(f"- {key}: {value}")
    already_collected_str = "\n".join(already_collected) if already_collected else "None yet"
    
    # Get persona instructions
    persona_instructions = persona.get("instructions", "Be friendly, warm, and helpful.")
    
    system_prompt = f"""You are {persona_name}, a warm and friendly assistant for Campous OS.

## YOUR PERSONALITY
{persona_instructions}
(NOTE: If your personality rules say "ask for email", IGNORE that Rule. Only ask for **{field_name}** as specified below.)

## CURRENT SITUATION
- User just said: "{effective_user_message}"
- {ack_hint if ack_hint else "This is the start of the conversation."}
- What you know about them: {context_summary}
- You need to ask for their: **{field_name}**

## ⚠️ ALREADY COLLECTED (DO NOT ask for these again!):
{already_collected_str}

## COMPANY INFO (use this to answer any questions)
{rag_context or "Campous OS offers a wide range of academic programs, research opportunities, and student services. We are dedicated to academic excellence and student success. Tuition costs vary by program - we can provide specific details after understanding your academic interests."}

## HOW TO RESPOND (2-3 sentences, warm and friendly)

{f'''🚨 **USER ASKED A QUESTION - ANSWER IT FIRST!**
The user asked: "{effective_user_message}"
You MUST answer their question helpfully before asking for {field_name}.

For generic questions like "Can you help me?", respond like:
- "Absolutely! I'm here to help you explore our academic programs and find the right path for you."
- "Of course! I'd love to help - we offer undergraduate, graduate, and research opportunities."
Then transition to ask for {field_name}.
''' if off_topic_handled else ''}
1. **ACKNOWLEDGE** what they just provided:
   - If they gave email: "Thanks! I've got your email as [their email]."
   - If they gave phone: "Perfect, got your number [their number]!"
   - If they gave name: "Nice to meet you, [their name]!"
   - If they just said something: Acknowledge it warmly

2. **ANSWER** any question they asked:
   - If they asked about programs, tuition, campus, etc. - use the university info above
   - Be helpful and specific, not vague
   - For tuition: say we provide specific details based on the program
   - For general "help" requests: Be welcoming and explain you can help with academic inquiries

3. **ASK** for the next piece of info:
   - End with a friendly question for their **{field_name}**
   - Bold the field name like: **phone number**, **email**, **name**
   - Make it conversational, like talking to a friend

## GOOD EXAMPLES

User: "singhtushar@gmail.com"
Response: "Thanks! I've got your email as singhtushar@gmail.com. What's the best **phone number** to reach you at?"

User: "my email is abc@gmail.com, how much is tuition?"
Response: "Thanks for your email! For tuition fees, we provide specific details based on your chosen program - it depends on the degree level and residency status. We'd love to discuss your academic goals in detail.

What's your **phone number** so our admissions team can connect with you?"

User: "I want to study Computer Science"
Response: "That's exciting! We have a top-ranked Computer Science program with state-of-the-art labs and expert faculty. Our graduates are highly sought after in the industry.

To discuss your application, what's your **email** address?"

User: "9876543210"
Response: "Got it, thanks for sharing your number! And what's your **name** so I know who I'm chatting with?"

User: "my name is Tushar and what programs do you offer?"
Response: "Nice to meet you, Tushar! We offer a wide range of academic programs including Engineering, Business, Arts, Sciences, and Medical degrees. We welcome students from all over the world.

What **program** are you interested in, Tushar?"

User: "Can you help me?"
Response: "Absolutely! I'd be happy to help you. We offer various undergraduate and postgraduate programs - so whatever your academic interests are, we've got a path for you.

What's your **name** so I know who I'm chatting with?"

## IMPORTANT RULES
- ALWAYS acknowledge what they provided (with the actual value if possible)
- ALWAYS answer their question if they asked one
- Keep responses warm, friendly, and conversational - NOT robotic
- Only ask for **{field_name}** - never ask for multiple fields
- ⚠️ NEVER ask for fields already in the "ALREADY COLLECTED" list above
- 2-3 sentences total, not too long

NOW RESPOND:"""


    
    # Log system prompt (abbreviated to avoid overwhelming logs)
    logger.info(f"\n\ud83d\udccb ENHANCER SYSTEM PROMPT (first 500 chars):")
    logger.info(system_prompt[:500] + "...")
    
    try:
        # Log LLM invocation - use lower temperature for consistency
        logger.info(f"\n\ud83e\udd16 Calling Enhancer LLM (temperature=0.5)...")
        
        llm = LLMGateway.get_llm(temperature=0.5)  # 🔴 FIX: Lowered from 0.9 to 0.5 for consistency
        
        # Gemini requires at least one HumanMessage
        # Keep it MINIMAL to not override the detailed system prompt
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Respond now.")  # Minimal instruction
        ]).content.strip()
        
        # 🔴 FIX: Deduplicate repeated content (LLM sometimes repeats itself)
        # Check if the first half is repeated in the second half
        if len(response) > 50:
            half_len = len(response) // 2
            first_half = response[:half_len]
            # Check if first 30 chars appear twice
            marker = response[:30]
            if response.count(marker) > 1:
                # Find the second occurrence and cut there
                second_start = response.find(marker, len(marker))
                if second_start > 0:
                    response = response[:second_start].strip()
                    logger.info(f"🔧 Deduplication: Removed repeated content")
        
        # Log enhancer output
        logger.info(f"\n\u2728 ENHANCER OUTPUT:")
        logger.info(f"   {response}")
        
        # Validate that the base question is included (anti-hallucination)
        base_lower = base_question.lower()
        response_lower = response.lower()
        
        # DYNAMIC KEYWORD EXTRACTION from base question
        # Extract meaningful words from the base question itself (no hardcoding!)
        # Filter out common stop words to get the key words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "about", "against", "i", "you", "your", "we", "us", "our",
            "what", "what's", "whats", "like", "would", "you'd", "youd"
        }
        
        # Extract keywords from base question
        base_words = base_lower.replace("?", "").replace("!", "").replace(".", "").replace(",", "").split()
        dynamic_keywords = [w for w in base_words if w not in stop_words and len(w) > 2]
        
        # Also add the current_key itself as a keyword (handles cases like "email", "phone")
        dynamic_keywords.append(current_key.lower().replace("_", " "))
        dynamic_keywords.append(current_key.lower().replace("_", ""))
        
        # Check if any of the dynamic keywords are in the response
        has_question = any(kw in response_lower for kw in dynamic_keywords)
        
        # Log validation results
        logger.info(f"\n\u2705 VALIDATION:")
        logger.info(f"   Question included: {has_question}")
        logger.info(f"   Keywords checked: {dynamic_keywords[:5]}...")
        
        # ALSO check if the base question itself is already in the response (prevent duplicates)
        base_already_present = base_lower in response_lower
        
        if not has_question and not base_already_present:
            # Enhancer failed to include the question - append it
            logger.warning(f"Enhancer didn't include question for {current_key}, appending (keywords: {dynamic_keywords})")
            response = f"{response} {base_question}"
        elif base_already_present:
            # Question is already there - no need to append
            logger.debug(f"Base question already present for {current_key}, not appending")
        
        logger.info("="*80)
        
        return response
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        # Fall back to a simple enhancement
        if user_last_message:
            return f"Thanks for that! {base_question}"
        return base_question


def should_enhance(current_key: str, collected_count: int) -> bool:
    """
    Decide whether to enhance a response.
    
    ALWAYS enhance to add user context and make responses richer.
    The enhancer uses a second LLM to add personalization without
    risking hallucination from the main question generator.
    """
    # 🔴 CUSTOM LOGIC: Disable enhancement for user_persona to prevent hallucination
    if current_key == "user_persona":
        return False

    # Always enhance - the enhancer adds context from user's message
    # This makes even the first question personalized (e.g., 
    # "To help your academic journey, what's your email?")
    return True


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RESPONSE ENHANCER TEST")
    print("=" * 60)
    
    persona = {
        "name": "Sara",
        "instructions": "Be warm, friendly, and conversational."
    }
    
    history = [
        {"role": "user", "content": "Python"},
        {"role": "assistant", "content": "Python is great! What's a good email to reach you?"},
        {"role": "user", "content": "test@example.com"},
    ]
    
    collected = {"email": "test@example.com"}
    
    base_question = "What number works best for you?"
    
    print(f"\nBase Question: {base_question}")
    print(f"History: {len(history)} messages")
    print(f"Collected: {collected}")
    print("\nEnhancing...")
    
    enhanced = enhance_response(
        base_question=base_question,
        persona=persona,
        conversation_history=history,
        collected_data=collected,
        current_key="phone",
        user_last_message="test@example.com"
    )
    
    print(f"\nEnhanced Response:\n{enhanced}")
