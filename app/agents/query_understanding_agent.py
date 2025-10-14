"""
Query Understanding Agent for Analytics Queries.

This agent uses LLM to understand and extract:
- Intent: What type of analysis the user wants (success_rate, failure_rate, etc.)
- Entities: What entities to analyze (domain_name, file_name, etc.)
- Completeness: Whether the query has all required information
"""
import json
import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from app.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


class QueryUnderstandingResult(BaseModel):
    """Structured result from query understanding."""
    intent: Optional[str] = Field(None, description="Detected user intent (success_rate, failure_rate, both_rate, domain_distribution, etc.)")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted slot values")
    confidence: float = Field(0.0, description="Confidence score (0.0 to 1.0)")
    missing_required: List[str] = Field(default_factory=list, description="List of required slots that are missing")
    is_complete: bool = Field(False, description="Whether query has all required information")
    clarification_needed: Optional[str] = Field(None, description="What to ask user if incomplete")


class QueryUnderstandingAgent:
    """
    Agent for understanding user queries and extracting intent and entities.
    
    Intents:
    - success_rate: Analyze success rates
    - failure_rate: Analyze failure rates
    
    Entities (Slots):
    - domain_name: Name of the domain to analyze
    - file_name: Name of the file to analyze
    """
    
    SYSTEM_PROMPT = """You are an expert query understanding agent for an analytics system.

Your task is to extract:
1. **INTENT** - What type of analysis does the user want?
2. **SLOTS** - What specific entities/parameters are mentioned?

📋 **SUPPORTED INTENTS:**
- "success_rate" - User wants to see success rates
- "failure_rate" - User wants to see failure rates
- "general_query" - General analytics question or unclear request
- "out_of_scope" - Non-analytics question (greetings, chitchat, unrelated topics)

🎯 **SUPPORTED SLOTS:**
- domain_name: Domain to analyze (e.g., "customer", "product", "order")
- file_name: File to analyze (e.g., "customer.csv", "sales.json")

⚠️ **CRITICAL RULES:**

1. **Intent Detection:**
   - Keywords "success", "successful" → intent: "success_rate"
   - Keywords "fail", "failure", "failed", "error" → intent: "failure_rate"
   - Greetings, chitchat, or non-analytics questions → intent: "out_of_scope"
   - Analytics-related but unclear → intent: "general_query"

2. **Out of Scope Detection:**
   - Greetings: "hello", "hi", "hey", "good morning"
   - Personal questions: "how are you", "what's your name", "who are you"
   - Unrelated topics: "weather", "sports", "news", "recipes"
   - Chitchat: "tell me a joke", "what can you do", "help"
   - When detected, set intent to "out_of_scope" and clarification_needed with helpful redirect

3. **Slot Extraction:**
   - File extensions (.csv, .json, .xlsx) → extract as file_name
   - Domain names without extension → extract as domain_name

4. **Required Slots per Intent:**
   - success_rate/failure_rate REQUIRES: domain_name OR file_name
   - general_query: May have partial information
   - out_of_scope: No slots required

5. **Validation:**
   - Check if all required slots are present
   - If missing, specify what's needed in "clarification_needed"
   - Set "is_complete" to true only if all required slots are present
   - For out_of_scope, always set is_complete=true (complete but not actionable)

📊 **EXAMPLES:**

Input: "show me success rate for customer.csv"
Output: {
  "intent": "success_rate",
  "slots": {"file_name": "customer.csv"},
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "failure rate for customer domain"
Output: {
  "intent": "failure_rate",
  "slots": {"domain_name": "customer"},
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "show me success rate"
Output: {
  "intent": "success_rate",
  "slots": {},
  "confidence": 0.8,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I need to know which file or domain to analyze. Please specify a file name (e.g., 'customer.csv') or domain name (e.g., 'customer domain')."
}


Input: "customer.csv"
Output: {
  "intent": "",
  "slots": {"file_name": "customer.csv"},
  "confidence": 0.8,
  "missing_required": ["intent"],
  "is_complete": false,
  "clarification_needed": null
}

Input: "customer domain"
Output: {
  "intent": "",
  "slots": {"domain_name": "customer"},
  "confidence": 0.8,
  "missing_required": ["intent"],
  "is_complete": false,
  "clarification_needed": null
}

Input: "generate a report or give me analytics report"
Output: {
  "intent": "general_query",
  "slots": {},
  "confidence": 0.6,
  "missing_required": ["report_type", "target"],
  "is_complete": false,
  "clarification_needed": "I can help you generate analytics reports! Please specify:\n1. What type of analysis? (success rate or failure rate)\n2. Which file or domain to analyze? (e.g., 'customer.csv' or 'customer domain')"
}

Input: "hello"
Output: {
  "intent": "out_of_scope",
  "slots": {},
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "Hello! I'm an analytics assistant. I can help you analyze success rates, failure rates, and generate reports for your data. What would you like to analyze?"
}

Input: "what's the weather today?"
Output: {
  "intent": "out_of_scope",
  "slots": {},
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "I'm specialized in data analytics and can't help with weather information. I can analyze success rates, failure rates, and generate reports from your data. Would you like to analyze any data?"
}

Input: "tell me a joke"
Output: {
  "intent": "out_of_scope",
  "slots": {},
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "I'm an analytics assistant focused on data analysis. I can help you with success rate analysis, failure rate reports, and data visualizations. What analytics would you like to see?"
}

🎯 **YOUR RESPONSE FORMAT:**
Return ONLY valid JSON matching the QueryUnderstandingResult schema. No additional text.
"""

    def __init__(self):
        """Initialize the query understanding agent."""
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,  # Deterministic for consistent extraction
            api_key=OPENAI_API_KEY
        )
    
    async def extract_intent_and_slots(self, user_query: str) -> QueryUnderstandingResult:
        """
        Extract intent and slots from user query.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            QueryUnderstandingResult with extracted intent, slots, and completeness info
        """
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Extract intent and slots from this query:\n\n{user_query}")
            ]
            
            logger.info(f"Understanding query: '{user_query[:100]}...'")
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                result_dict = json.loads(response_text)
                result = QueryUnderstandingResult(**result_dict)
                
                logger.info(f"Extracted intent: {result.intent}, slots: {result.slots}, complete: {result.is_complete}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.warning(f"Raw response: {response_text}")
                
                # Fallback: return incomplete result
                return QueryUnderstandingResult(
                    intent="general_query",
                    slots={},
                    confidence=0.0,
                    missing_required=["intent", "domain_name or file_name"],
                    is_complete=False,
                    clarification_needed="I couldn't understand your request. Please specify what you'd like to analyze and which file or domain."
                )
                
        except Exception as e:
            logger.exception(f"Error in query understanding: {e}")
            
            return QueryUnderstandingResult(
                intent="general_query",
                slots={},
                confidence=0.0,
                missing_required=["intent", "domain_name or file_name"],
                is_complete=False,
                clarification_needed="I encountered an error processing your request. Please try rephrasing your question."
            )
    
    def validate_completeness(self, result: QueryUnderstandingResult) -> QueryUnderstandingResult:
        """
        Validate if the extracted intent and slots are complete.
        Updates the result with missing required slots and clarification.
        
        Args:
            result: The initial extraction result
            
        Returns:
            Updated result with validation information
        """
        intent = result.intent
        result.clarification_needed = None
        
        # Handle out_of_scope intent - always complete but not actionable
        if intent == "out_of_scope":
            result.clarification_needed = (
                    "I'm an analytics assistant specialized in data analysis. "
                    "I can help you with success rates, failure rates, and data reports. "
                    "What would you like to analyze?"
                )
            return result
        
        elif intent == "general_query":
            # Check what's missing
            
            result.clarification_needed = (
                "I can help you with analytics! Please specify:\n"
                "1. What type of analysis? (success rate, failure rate)\n"
                "2. Which file or domain to analyze?"
            )
        
        
        return result


# Singleton instance
_query_understanding_agent = None

def get_query_understanding_agent() -> QueryUnderstandingAgent:
    """Get the singleton query understanding agent."""
    global _query_understanding_agent
    if _query_understanding_agent is None:
        _query_understanding_agent = QueryUnderstandingAgent()
    return _query_understanding_agent
