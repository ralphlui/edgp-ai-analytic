import json
import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from config.app_config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts.query_understanding_prompts import QueryUnderstandingPrompt
from app.security.pii_redactor import PIIRedactionFilter, redact_pii

logger = logging.getLogger(__name__)

# Add PII redaction filter to this logger
pii_filter = PIIRedactionFilter()
logger.addFilter(pii_filter)


class QueryUnderstandingResult(BaseModel):
    """Structured result from query understanding."""
    intent: Optional[str] = Field(None, description="Detected user intent (success_rate, failure_rate, both_rate, domain_distribution, etc.)")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted slot values (domain_name, file_name, etc.)")
    chart_type: Optional[str] = Field(None, description="Preferred chart type (bar, pie, line, donut, area)")
    confidence: float = Field(0.0, description="Confidence score (0.0 to 1.0)")
    missing_required: List[str] = Field(default_factory=list, description="List of required slots that are missing")
    is_complete: bool = Field(False, description="Whether query has all required information")
    clarification_needed: Optional[str] = Field(None, description="What to ask user if incomplete")
    query_type: Optional[str] = Field(None, description="Type of query: 'simple' or 'complex'")
    high_level_intent: Optional[str] = Field(None, description="High-level intent for complex queries: 'comparison', 'aggregation', etc.")
    comparison_targets: List[str] = Field(default_factory=list, description="List of targets for comparison queries")


class QueryUnderstandingAgent:
    
    def __init__(self):
        """Initialize the query understanding agent with secure prompt template."""
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,  # Deterministic for consistent extraction
            api_key=OPENAI_API_KEY
        )
        # Initialize secure prompt template
        self.prompt_template = QueryUnderstandingPrompt()
    
    async def extract_intent_and_slots(self, user_query: str) -> QueryUnderstandingResult:
        """
        Extract intent and slots from user query.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            QueryUnderstandingResult with extracted intent, slots, and completeness info
        """
        try:
            # Get secure system prompt with leakage prevention
            system_prompt = self.prompt_template.get_system_prompt()
            
            # Format user message with security validation and structural isolation
            user_message = self.prompt_template.format_user_message(query=user_query)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            logger.info(f"Understanding query: '{user_query[:100]}...'")
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                result_dict = json.loads(response_text)
                
                # Validate response schema with security checks
                self.prompt_template.validate_response_schema(result_dict)
                
                # Extract chart_type from slots and move to top-level
                chart_type = result_dict.get('slots', {}).pop('chart_type', None)
                if chart_type:
                    result_dict['chart_type'] = chart_type
                
                result = QueryUnderstandingResult(**result_dict)
                
                logger.info(f"Extracted intent: {result.intent}, slots: {result.slots}, chart_type: {result.chart_type}, complete: {result.is_complete}")
                
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
        query_type = result.query_type
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
            # Check what's already provided to avoid asking for it again
            has_domain = "domain_name" in result.slots and result.slots["domain_name"]
            has_file = "file_name" in result.slots and result.slots["file_name"]

            # If either a domain or a file is provided, ask only for the metric type.
            if not has_domain or not has_file:
                
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
