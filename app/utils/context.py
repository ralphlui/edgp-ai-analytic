"""
Context and conversation handling utilities.
"""
import re
import os
from typing import List, Dict, Any, Optional
from .sanitization import sanitize_text_input, sanitize_filename, sanitize_numeric_value

# Optional: Use ReAct prompts if available
USE_REACT_PROMPTS = os.getenv("USE_REACT_PROMPTS", "true").lower() == "true"


def extract_context_insights(data: List[Dict[str, Any]]) -> List[str]:
    """Extract key insights from tool result data for context awareness."""
    insights = []

    if not data:
        return insights

    try:
        # Calculate basic metrics
        total_records = len(data)
        insights.append(f"📊 Dataset contains {total_records} records")

        # Extract success rates if available
        success_rates = []
        for record in data:
            if "success_rate" in record:
                try:
                    rate = float(record["success_rate"])
                    success_rates.append(rate)
                except (ValueError, TypeError):
                    continue

        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            min_success = min(success_rates)
            max_success = max(success_rates)

            insights.append(f"📈 Average success rate: {avg_success:.1f}%")
            insights.append(f"🎯 Success rate range: {min_success:.1f}% - {max_success:.1f}%")

            # Add performance insights
            if avg_success >= 95:
                insights.append("🌟 Excellent performance across the board")
            elif avg_success >= 80:
                insights.append("✅ Good overall performance")
            elif avg_success >= 60:
                insights.append("⚠️ Moderate performance - room for improvement")
            else:
                insights.append("🚨 Poor performance - requires immediate attention")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to extract context insights: {e}")

    return insights


def get_conversation_context(messages: List) -> List:
    """Extract recent conversation context for better continuity."""
    from langchain_core.messages import HumanMessage, AIMessage

    context_messages = []

    # Get last 10 messages for context (excluding current message)
    recent_messages = messages[-10:-1] if len(messages) > 1 else []

    for msg in recent_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            context_messages.append(msg)

    return context_messages


def create_interpretation_prompt(user_query: str, tool_results: List[Dict[str, Any]], report_type: str = None, context_insights: List[str] = None) -> str:
    """Create a context-aware interpretation prompt that aligns with the user's specific query intent."""

    # Option 1: Use ReAct-enhanced prompts if enabled
    if USE_REACT_PROMPTS:
        try:
            from app.prompts import ReActPrompts
            import json
            
            react_prompt = ReActPrompts.get_react_interpretation_prompt()
            return react_prompt.format(
                user_query=sanitize_text_input(user_query, 200),
                tool_results=json.dumps(tool_results, indent=2),
                context_insights="\n".join(context_insights) if context_insights else "No additional insights"
            )
        except ImportError:
            pass  # Fall back to legacy prompts
    
    # Option 2: Legacy prompt format (default)
    # Build the interpretation prompt
    prompt_parts = []
    
    # Add user query context first for proper alignment
    if user_query:
        safe_user_query = sanitize_text_input(user_query, 200)
        prompt_parts.append("USER QUERY:")
        prompt_parts.append(f"└── {safe_user_query}")
        prompt_parts.append("")

    # Add context insights with sanitization
    if context_insights:
        prompt_parts.append("CONTEXT INSIGHTS:")
        for insight in context_insights:
            safe_insight = sanitize_text_input(insight, 100)
            if safe_insight:
                prompt_parts.append(f"├── {safe_insight}")
        prompt_parts.append("")

    # Add tool results summary with sanitization
    prompt_parts.append("TOOL RESULTS:")
    for i, result in enumerate(tool_results):
        prompt_parts.append(f"├── Result {i+1}:")
        if isinstance(result, dict):
            for key, value in result.items():
                safe_key = sanitize_text_input(str(key), 50)
                if key == "data" and isinstance(value, list):
                    prompt_parts.append(f"│   ├── {safe_key}: {len(value)} records")
                    if value and len(value) > 0:
                        # Show sample of first record with sanitization
                        sample = value[0]
                        if isinstance(sample, dict):
                            sample_keys = list(sample.keys())[:3]  # Show first 3 keys
                            safe_sample_keys = [sanitize_text_input(str(k), 30) for k in sample_keys]
                            prompt_parts.append(f"│   │   ├── Sample keys: {', '.join(safe_sample_keys)}")
                else:
                    safe_value = sanitize_text_input(str(value), 100)
                    prompt_parts.append(f"│   ├── {safe_key}: {safe_value}")
        else:
            safe_content = sanitize_text_input(str(result), 100)
            prompt_parts.append(f"│   ├── Raw content: {safe_content}...")
        prompt_parts.append("")

    # Simplified LLM Interpretation: Let AI decide what related message should be replied
    # Add intelligent context-aware interpretation instructions
    prompt_parts.append("LLM INTERPRETATION TASK:")
    prompt_parts.append("├── Analyze the data and determine what related message should be prioritized")
    prompt_parts.append("├── CRITICAL: Always reference the exact file names and entities mentioned in the USER QUERY above")
    prompt_parts.append("├── Let the AI intelligently decide the focus based on:")
    prompt_parts.append("│   ├── User query intent and semantic meaning (match exact file names)")
    prompt_parts.append("│   ├── Data patterns and significant findings")
    prompt_parts.append("│   ├── Context relevance and user needs")
    prompt_parts.append("│   └── Most appropriate response approach for this specific query")
    prompt_parts.append("├── Adaptive Response Guidelines:")
    prompt_parts.append("│   ├── Match tone and focus to user's specific query intent")
    prompt_parts.append("│   ├── Prioritize insights that directly answer the user's question")
    prompt_parts.append("│   ├── Use EXACT file names and entities from the USER QUERY (not examples, use the actual query)")
    prompt_parts.append("│   ├── Use language style that aligns with the query context")
    prompt_parts.append("│   └── Emphasize the most relevant data story for this specific query")
    prompt_parts.append("├── Keep response concise but impactful - 3-4 sentences maximum")
    prompt_parts.append("├── Focus on actionable insights that matter most to this query")
    prompt_parts.append("└── Let the data and user intent guide the narrative direction")

    return "\n".join(prompt_parts)


