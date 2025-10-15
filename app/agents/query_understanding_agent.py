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
    query_type: Optional[str] = Field(None, description="Type of query: 'simple' or 'complex'")
    high_level_intent: Optional[str] = Field(None, description="High-level intent for complex queries: 'comparison', 'aggregation', etc.")
    comparison_targets: List[str] = Field(default_factory=list, description="List of targets for comparison queries")


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
3. **QUERY TYPE** - Is this a simple or complex query?
4. **COMPARISON TARGETS** - For comparison queries, list all targets being compared

📋 **SUPPORTED INTENTS:**
- "success_rate" - User wants to see success rates
- "failure_rate" - User wants to see failure rates
- "comparison" - User wants to compare metrics between multiple targets (when metric type not specified)
- "general_query" - General analytics question or unclear request
- "out_of_scope" - Non-analytics question (greetings, chitchat, unrelated topics)

📊 **QUERY TYPES:**
- "simple" - Single target analysis (one domain or file)
- "complex" - Multi-target analysis (comparisons, aggregations)

🎯 **HIGH-LEVEL INTENTS (for complex queries):**
- "comparison" - Comparing metrics between multiple targets
- "aggregation" - Combining data from multiple sources
- "trend" - Analyzing trends over time

🎯 **SUPPORTED SLOTS:**
- domain_name: Domain to analyze (e.g., "customer", "product", "order")
- file_name: File to analyze (e.g., "customer.csv", "sales.json")

⚠️ **CRITICAL RULES:**

1. **Intent Detection:**
   - Keywords "success", "successful" → intent: "success_rate"
   - Keywords "fail", "failure", "failed", "error" → intent: "failure_rate"
   - Keywords "compare", "comparison", "vs", "versus" WITHOUT explicit metric → intent: "comparison"
   - Keywords "compare" WITH "success" → intent: "success_rate"
   - Keywords "compare" WITH "fail/failure/error" → intent: "failure_rate"
   - Greetings, chitchat, or non-analytics questions → intent: "out_of_scope"
   - Analytics-related but unclear → intent: "general_query"

2. **Query Type Detection:**
   - Keywords "compare", "between", "vs", "versus" → query_type: "complex", high_level_intent: "comparison"
   - Multiple targets mentioned (e.g., "customer.csv and product.csv") → query_type: "complex"
   - Single target → query_type: "simple"

3. **Comparison Target Extraction:**
   - Extract ALL files or domains mentioned in comparison queries
   - Format: Preserve exact names as mentioned (e.g., "customer.csv", "product.csv")
   - Store in comparison_targets array
   - For "file" without extension, add ".csv" (e.g., "customer file" → "customer.csv")
   - For "domain" without extension, store without extension (e.g., "customer domain" → "customer")
   - Mixed comparisons allowed: domain vs file (e.g., ["customer", "payment.csv"])
   - Domain vs domain: both without extension (e.g., ["customer", "payment"])
   - File vs file: both with extension (e.g., ["customer.csv", "payment.csv"])

4. **Out of Scope Detection:**
   - Greetings: "hello", "hi", "hey", "good morning"
   - Personal questions: "how are you", "what's your name", "who are you"
   - Unrelated topics: "weather", "sports", "news", "recipes"
   - Chitchat: "tell me a joke", "what can you do", "help"
   - When detected, set intent to "out_of_scope" and clarification_needed with helpful redirect

5. **Slot Extraction:**
   - File extensions (.csv, .json, .xlsx) → extract as file_name
   - Domain names without extension → extract as domain_name
   - For complex queries, also populate comparison_targets

6. **Required Slots per Intent:**
   - success_rate/failure_rate (simple) REQUIRES: domain_name OR file_name
   - success_rate/failure_rate (complex comparison) REQUIRES: comparison_targets (at least 2)
   - comparison (complex) REQUIRES: comparison_targets (at least 2)
   - general_query: May have partial information
   - out_of_scope: No slots required

7. **Validation:**
   - Check if all required slots are present
   - If missing, specify what's needed in "clarification_needed"
   - Set "is_complete" to true only if all required slots are present
   - For out_of_scope, always set is_complete=true (complete but not actionable)

📊 **EXAMPLES:**

**SIMPLE QUERIES:**

Input: "show me success rate for customer.csv"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"file_name": "customer.csv"},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "failure rate for customer domain"
Output: {
  "intent": "failure_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer"},
  "comparison_targets": [],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "show me success rate"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.8,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I need to know which file or domain to analyze. Please specify a file name (e.g., 'customer.csv') or domain name (e.g., 'customer domain')."
}

**COMPLEX COMPARISON QUERIES (COMPLETE):**

Input: "Compare failure rates between customer.csv and product.csv"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "product.csv"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "Compare failure rates between customer file and product file"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "product.csv"],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "compare success rate for customer.csv vs payment.csv"
Output: {
  "intent": "success_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "payment.csv"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "compare success rate for customer domain vs payment.csv"
Output: {
  "intent": "success_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer", "payment.csv"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "compare success rate for customer domain vs payment domain"
Output: {
  "intent": "success_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer", "payment"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "compare customer.csv and product domain"
Output: {
  "intent": "comparison",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "product"],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": false,
  "clarification_needed": "I see you want to compare customer.csv and product. What type of analysis would you like? (success rate or failure rate)"
}

Input: "compare failure rate between customer.csv and product domain"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "product"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

**COMPLEX COMPARISON QUERIES (INCOMPLETE):**

Input: "Compare failure rates between customer.csv"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv"],
  "confidence": 0.7,
  "missing_required": ["second_comparison_target"],
  "is_complete": false,
  "clarification_needed": "I see you want to compare failure rates for customer.csv. What would you like to compare it with? Please specify another file or domain."
}

Input: "compare success rates"
Output: {
  "intent": "success_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.6,
  "missing_required": ["comparison_targets"],
  "is_complete": false,
  "clarification_needed": "I understand you want to compare success rates. Please specify which files or domains to compare (e.g., 'compare success rates between customer.csv and product.csv')."
}

Input: "customer.csv"
Output: {
  "intent": "",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"file_name": "customer.csv"},
  "comparison_targets": [],
  "confidence": 0.8,
  "missing_required": ["intent"],
  "is_complete": false,
  "clarification_needed": null
}

Input: "customer domain"
Output: {
  "intent": "",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer"},
  "comparison_targets": [],
  "confidence": 0.8,
  "missing_required": ["intent"],
  "is_complete": false,
  "clarification_needed": null
}

Input: "generate a report or give me analytics report"
Output: {
  "intent": "general_query",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.6,
  "missing_required": ["report_type", "target"],
  "is_complete": false,
  "clarification_needed": "I can help you generate analytics reports! Please specify:\n1. What type of analysis? (success rate or failure rate)\n2. Which file or domain to analyze? (e.g., 'customer.csv' or 'customer domain')"
}

Input: "hello"
Output: {
  "intent": "out_of_scope",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "Hello! I'm an analytics assistant. I can help you analyze success rates, failure rates, and generate reports for your data. What would you like to analyze?"
}

Input: "what's the weather today?"
Output: {
  "intent": "out_of_scope",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "I'm specialized in data analytics and can't help with weather information. I can analyze success rates, failure rates, and generate reports from your data. Would you like to analyze any data?"
}

Input: "tell me a joke"
Output: {
  "intent": "out_of_scope",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
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
        query_type = result.query_type or "simple"
        high_level_intent = result.high_level_intent
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
