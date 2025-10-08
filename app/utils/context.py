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


# Legacy functions removed - these are no longer used in V2 graph builder
# The V2 implementation uses build_interpretation_summary() from message_compression.py
# which provides better token management and PII protection


