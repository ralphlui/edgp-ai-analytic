"""
Enhanced analytic service with improved reference resolution and LLM context.
"""
import asyncio
import json
import logging
from typing import List, Dict, Any
from app.tools import (
    ANALYSIS_TOOLS
)
from app.utils.report_type import get_report_type
from app.utils.sanitization import sanitize_filename
from app.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


class AnalyticService:
    """Handles LLM tool selection, execution, chart generation, and interpretation."""

    # Use the centralized tool registry
    TOOLS = ANALYSIS_TOOLS

    @staticmethod
    async def process_query(prompt: str, user_id: str = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for processing analytic queries with enhanced reference resolution.
        
        Args:
            prompt: The user's query
            session_id: Session identifier for tracking
            conversation_history: Optional conversation history for context

        Returns a dict with keys: success (bool), message (str), chart_image (str base64)
        """
        try:
            # Use unified hybrid classification (handles regex + LLM internally)
            report_type = await get_report_type(prompt)
            
            logger.info(f"Detected report type: '{report_type}' for prompt: '{prompt[:50]}...'")
            
            # Debug: Double-check the classification
            if "fail" in prompt.lower() or "failure" in prompt.lower():
                logger.info(f"Prompt contains failure keywords, report_type should be 'failure' but got '{report_type}'")

            # Process the query with the detected report type
            return await AnalyticService.process_query_with_report_type(
                prompt=prompt,
                report_type=report_type,
                user_id=user_id,
                conversation_history=conversation_history
            )

        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            return {"success": False, "message": str(e), "chart_image": None}  
        
    @staticmethod
    def filter_chart_data_by_report_type(chart_data: List[Dict], report_type: str) -> List[Dict]:
        """Filter chart data based on report type."""
        if report_type == "both":
            return chart_data

        # Check if this is domain analytics data (flexible format)
        if chart_data and any(
            'country' in item or 
            'customer_count' in item or
            any(key.endswith('_count') for key in item.keys()) or
            any(key in ['category', 'region', 'location', 'type'] for key in item.keys())
            for item in chart_data
        ):
            # Domain analytics data doesn't need report type filtering
            return chart_data

        # Apply report type filtering for success/failure data
        filtered_data = []
        for item in chart_data:
            status = item.get('status', '').lower()
            if report_type == "success" and status == 'success':
                filtered_data.append(item)
            elif report_type == "failure" and status in ['fail', 'failure']:
                filtered_data.append(item)

        return filtered_data

    @staticmethod
    def _create_enhanced_system_message(current_date: str, conversation_insights: str) -> str:
        """Create comprehensive system message with reference resolution instructions."""
        from app.config import SYSTEM
        from app.prompts.system_prompts import SystemPrompts
        
        base_system = SYSTEM.format(current_date=current_date)
        
        enhanced_instructions = f"""

CONVERSATION CONTEXT:
{conversation_insights}

CRITICAL: PRIORITIZE CURRENT USER PROMPT
- ALWAYS analyze and respond to the CURRENT user prompt first and foremost
- If current prompt is NOT analytics-related (e.g., general conversation, coding questions, weather, news, etc.), respond with: {SystemPrompts.get_non_analytics_fallback().format()}
- For analytics queries: Conversation history is provided for context ONLY - do not let it override current request
- If current prompt is different from previous requests, follow the current prompt completely
- Only use conversation history to fill in gaps when current prompt is incomplete AND analytics-related

IMPORTANT GUIDELINES FOR TOOL USE:
- This system supports TWO types of analysis:
  1. SUCCESS/FAILURE RATE ANALYSIS: Requires file_name/domain_name + report_type ('success'/'failure'/'both')
  2. DOMAIN DISTRIBUTION ANALYSIS: Shows "how many [domain] by [column]" - only needs domain_name (no report_type needed)

- SMART CLARIFICATION RULES (based on CURRENT prompt):
  * If current prompt has NO success/fail keywords AND NO file/domain names → Ask user to specify what they want to analyze
  * If current prompt says only "success" or "failure" → Ask which file or domain they want analyzed
  * If current prompt says only "customer.csv" or "customer domain" → Ask if they want success/failure rates OR domain distribution
  * If current prompt uses VAGUE REFERENCES like "previous", "that", "this domain", "this file" → Ask user to specify exactly which domain/file they mean
  * If current prompt says "how many [domain] by [column]" → Proceed directly with domain analysis (no success/failure needed)
  * If current prompt provides complete context (has both analysis type AND target) → Proceed directly without asking
  * Use conversation history ONLY for incomplete current prompts, not to override clear current requests

- TOOL SELECTION PRIORITY:
  * If user mentions file extensions (.csv, .json, .xlsx) or specific filenames → Use FILE-based analysis tools
  * If user mentions domain names without file extension → Use DOMAIN-based analysis tools
  * Keywords "success", "failure", "rate" → Use success/failure analysis tools  
  * Keywords "how many", "by country", "by category", "distribution" → Use domain distribution tools

- EXAMPLES:
  * "show me data" → ASK FOR CLARIFICATION (no file/domain, no analysis type)
  * "generate report" → ASK FOR CLARIFICATION (no file/domain, no analysis type)  
  * "customer.csv" → ASK what type of analysis (file provided, but analysis type unclear)
  * "success rate" → ASK which file/domain (analysis type provided, but target unclear)
  * "success rate for that file" → ASK which specific file (vague reference)
  * "analyze the previous domain" → ASK which specific domain (vague reference)
  * "show me this data" → ASK which specific file/domain (vague reference)
  * "success rate for customer.csv" → PROCEED with file success/failure analysis
  * "how many customers by country" → PROCEED with domain distribution analysis

- For chart_type: Choose automatically based on data type, or use user-specified type.
"""
        
        return base_system + enhanced_instructions

    @staticmethod
    def _extract_conversation_insights(conversation_history: List[Dict[str, Any]]) -> str:
        """Extract key insights from conversation history for LLM context."""
        if not conversation_history:
            return "No previous conversation history."
        
        insights = []
        recent_interactions = conversation_history[-3:]
        
        for i, interaction in enumerate(recent_interactions, 1):
            response = interaction.get("response_summary", {})
            
            insight_parts = [f"Interaction {i}:"]
            
            if response.get("file_name"):
                safe_filename = sanitize_filename(response['file_name'])
                insight_parts.append(f"  - Analyzed file: {safe_filename}")
            
            if response.get("domain_name"):
                domain = response['domain_name']
                insight_parts.append(f"  - Analyzed domain: {domain}")
            
            if response.get("report_type"):
                report_type = response['report_type']
                insight_parts.append(f"  - Report type: {report_type}")
            
            if response.get("row_count"):
                row_count = response['row_count']
                insight_parts.append(f"  - Data points: {row_count}")
            
            insights.append("\n".join(insight_parts))
        
        return "\n".join(insights) if insights else "No significant previous interactions."

    @staticmethod
    async def process_query_with_report_type(prompt: str, report_type: str, user_id: str = None,
                                           conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process analytic query with enhanced LLM context and reference resolution.
        """
        try:
            # Import here to avoid circular imports
            from .graph_builder import build_analytics_graph
            from app.generators.chart_generator import chart_generator
            from app.utils.sanitization import sanitize_text_input
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            from datetime import date

            logger.info(f"Processing prompt: '{prompt[:100]}...'")
            logger.info(f"Report type: '{report_type}'")
            logger.info(f"conversation_history type: '{conversation_history}'")

            # Build Graph V2 (typed state + compression + PII protection)
            app_graph = build_analytics_graph()
            logger.info("Using Graph Builder V2 (typed state + compression)")

        except Exception as e:
            return {"success": False, "message": str(e), "chart_image": None}

        # Get current date for context
        current_date = date.today().strftime('%Y-%m-%d')

        # Get reference context from memory service
        # Extract conversation insights
        # conversation_insights = AnalyticService._extract_conversation_insights(conversation_history or [])
        # print(f"Conversation insights for LLM: {conversation_insights}")
        # print(f"Conversation conversation_history for LLM: {conversation_history}")

        # Create enhanced system message with reference resolution
        #conversation_insights = AnalyticService._extract_conversation_insights(conversation_history or [])
        system_message = AnalyticService._create_enhanced_system_message(
            current_date, conversation_history
        )
        
        messages = [SystemMessage(content=system_message)]
        logger.info("Added conversation insights to context")

        # Add simplified conversation history
        if conversation_history:
            recent_interactions = conversation_history[-10:] 
            logger.info(f"Reduced conversation history to last 10 interactions: {recent_interactions}")
            
            for interaction in recent_interactions:
                if interaction.get("user_prompt"):
                    safe_prompt = sanitize_text_input(interaction['user_prompt'], 150)
                    messages.append(HumanMessage(content=safe_prompt))
                    logger.info(f"Added user prompt to message: '{messages}...'")

                
                # Skip adding previous response summaries to avoid misleading the LLM
                # The LLM should focus on the current query rather than cached summaries

        # Add the current prompt
        safe_prompt = sanitize_text_input(prompt, 300)
        messages.append(HumanMessage(content=safe_prompt))

        # Prepare state with V2 typed state structure
        # Note: Don't pass pre-classified report_type to LLM - let LLM decide based on prompt
        # Note: user_id is NOT included in state - multi-tenant isolation uses contextvars
        # (see app.utils.request_context.USER_ID_CTX set at API layer)
        state = {
            "messages": messages,
            "loop_count": 0,
            "tool_results": [],
            "context_insights": [],
            "compression_applied": False,
            "total_tokens_estimate": 0
        }
        logger.info("Using V2 typed state structure")

        loop = asyncio.get_running_loop()
        try:
            # Execute the graph
            compiled_result = await loop.run_in_executor(None, lambda: app_graph.invoke(state))
        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            
            error_message = f"""
I encountered an issue processing your {report_type} analysis request.

This might be due to:
- Database connectivity issues
- Invalid file/domain references
- Authentication problems
- System resource constraints

Please verify your file or domain references and try again.
"""
            return {
                "success": False,
                "error": str(e),
                "message": error_message.strip(),
                "chart_image": None
            }

        # Extract tool results and generate chart
        tool_results = []
        chart_data = []
        file_name = None
        domain_name = None
        row_count = 0
        date_filter_used = None
        chart_type = "bar"
        chart_type_source = "default"  # one of: default, auto, requested
        llm_detected_report_type = None  # Track LLM-detected report type

        for m in compiled_result.get("messages", []):
            if hasattr(m, 'content') and hasattr(m, '__class__'):
                # Handle ToolMessage
                if m.__class__.__name__ == 'ToolMessage':
                    try:
                        tool_data = json.loads(m.content) if isinstance(m.content, str) else m.content
                        tool_results.append(tool_data)
                        logger.info(f"Tool result: {tool_data}")

                        if tool_data.get("success") and "chart_data" in tool_data:
                            chart_data = tool_data.get("chart_data", [])
                            file_name = tool_data.get("file_name")
                            domain_name = tool_data.get("domain_name")
                            row_count = tool_data.get("row_count", 0)

                            # Respect chart type returned by tool. If it was explicitly requested, mark as requested; otherwise mark as auto.
                            if "chart_type" in tool_data:
                                # Do not override an explicitly requested chart type detected earlier
                                if chart_type_source != "requested":
                                    chart_type = tool_data.get("chart_type", chart_type or "bar")
                                    chart_type_source = "requested" if tool_data.get("chart_type_requested") else "auto"
                                    logger.info(f"Chart type from tool: {chart_type} (source: {chart_type_source})")

                            # Extract LLM-detected report type
                            if tool_data.get("report_type_requested"):
                                llm_detected_report_type = tool_data.get("report_type")
                                logger.info(f"LLM detected report_type: {llm_detected_report_type}")

                            if tool_data.get("date_filter"):
                                date_filter_used = tool_data.get("date_filter")
                    except Exception as e:
                        logger.warning(f"Failed to parse tool message: {e}")
                        tool_results.append(str(m.content))

                # Handle AIMessage with tool calls
                elif m.__class__.__name__ == 'AIMessage' and getattr(m, 'tool_calls', None):
                    for tool_call in m.tool_calls or []:
                        if tool_call.get('args'):
                            args = tool_call['args']
                            if args.get('start_date') or args.get('end_date'):
                                date_filter_used = {
                                    'start_date': args.get('start_date'),
                                    'end_date': args.get('end_date')
                                }
                            if args.get('chart_type'):
                                chart_type = args.get('chart_type')
                                chart_type_source = "requested"
                            
                            # Extract report_type from tool call arguments
                            if args.get('report_type'):
                                llm_detected_report_type = args.get('report_type')
                                logger.info(f"LLM tool call specified report_type: {llm_detected_report_type}")
    
        # Smart decision logic: prioritize clear pre-classification over ambiguous LLM detection
        final_report_type = llm_detected_report_type
        if  final_report_type is None:
            final_report_type = report_type
        
        
        logger.info(f"Final report_type: {final_report_type} (LLM: {llm_detected_report_type}, Pre-classified: {report_type})")

        # Filter chart data based on final report type  
        # Store original only if debug mode is enabled to save memory
        from app.config import DEBUG
        original_chart_data = chart_data.copy() if DEBUG else None
        filtered_chart_data = AnalyticService.filter_chart_data_by_report_type(chart_data, final_report_type)

        # Optionally confirm chart type before generation if it wasn't explicitly requested
        chart_image = None
        chart_generated = False
        if filtered_chart_data:
            # If the chart type was auto/default (user didn't specify), return a confirmation prompt first
            if chart_type_source in ("auto", "default"):
                confirmation_message = (
                    f"I'll use a '{chart_type or 'bar'}' chart for this analysis. "
                    "Reply 'yes' to confirm, or specify a chart type (pie, line, donut, stacked)."
                )
                return {
                    "success": False,
                    "requires_confirmation": True,
                    "message": confirmation_message,
                    "proposed_chart_type": chart_type or "bar",
                    "file_name": file_name,
                    "domain_name": domain_name,
                    "row_count": row_count,
                    "report_type": final_report_type
                }

            try:
                # Ensure chart_type has a sensible default
                chart_type = chart_type or "bar"
                chart_image = chart_generator.generate_chart(
                    chart_data=filtered_chart_data,
                    chart_type=chart_type,
                    file_name=file_name or domain_name,
                    report_type=final_report_type
                )
                chart_generated = True
                logger.info(f"Generated {chart_type} chart for {file_name or domain_name}")
            except Exception as e:
                logger.exception(f"Failed to generate chart: {e}")

        # Extract LLM interpretation from graph results
        interpretation = None
        for m in compiled_result.get("messages", []):
            if hasattr(m, 'content') and hasattr(m, '__class__'):
                # Get the final AI interpretation message
                if m.__class__.__name__ == 'AIMessage' and not getattr(m, 'tool_calls', None):
                    interpretation = m.content
                    break
    # Note: Do not log raw interpretation before filtering to avoid leaking sensitive data
        
        # Fallback if no interpretation found - use basic formatting
        if not interpretation:
            from app.utils.formatting import format_basic_message
            
            logger.info(f"No interpretation found, using format_basic_message for query: '{prompt[:50]}...'")
            interpretation = format_basic_message(
                chart_data=filtered_chart_data,
                file_name=file_name or domain_name,
                row_count=row_count,
                chart_type=chart_type,
                report_type=final_report_type,
                date_filter_used=date_filter_used,
                original_chart_data=original_chart_data
            )

        # Apply Responsible AI output filters (PII/secrets redaction)
        redaction_stats = None
        try:
            from app.config import (
                ENABLE_PII_REDACTION,
                ENABLE_SECRET_REDACTION,
                INCLUDE_REDACTION_METADATA,
                ENABLE_JWT_REDACTION,
                ENABLE_URL_CREDENTIAL_REDACTION,
                ENABLE_BASE64_REDACTION,
                BASE64_MIN_LEN,
            )
            from app.utils.responsible_ai import apply_responsible_output_filters, RedactionConfig

            cfg = RedactionConfig(
                enable_pii_redaction=ENABLE_PII_REDACTION,
                enable_secret_redaction=ENABLE_SECRET_REDACTION,
                enable_jwt_redaction=ENABLE_JWT_REDACTION,
                enable_url_credential_redaction=ENABLE_URL_CREDENTIAL_REDACTION,
                enable_base64_redaction=ENABLE_BASE64_REDACTION,
                base64_min_len=BASE64_MIN_LEN,
            )
            filtered_text, redaction_stats = apply_responsible_output_filters(interpretation, cfg)
            interpretation = filtered_text

        except Exception as _filter_err:
            # Don't fail the request if filtering fails; just log
            logger.warning(f"Responsible output filtering failed: {_filter_err}")

        # If chart type was auto-selected (user didn't specify), append a concise confirmation note to the message
        try:
            if isinstance(interpretation, str) and chart_generated and chart_type_source in ("auto", "default"):
                interpretation = (
                    interpretation.rstrip() +
                    f"\n\nNote: No chart type was provided, so I selected a '{chart_type}' chart automatically. "
                    "If you prefer a different style (pie, line, donut, stacked), reply with: chart: <type>."
                )
        except Exception:
            # Don't fail if post-processing note injection has issues
            pass

        # Safe logging after filtering
        try:
            if redaction_stats:
                logger.info(f"Applied output redactions: {redaction_stats}")
            # Log a truncated, filtered message
            if isinstance(interpretation, str):
                logger.info(f"Final message (filtered): {interpretation[:200]}...")
        except Exception:
            pass


        result = {
            "success": True,
            "message": interpretation,
            "chart_image": chart_image,
            "file_name": file_name,
            "domain_name": domain_name,
            "row_count": row_count,
            "report_type": final_report_type,
            "chart_type": chart_type
        }

        # Optionally include metadata about redactions for observability (no sensitive data included)
        # try:
        #     if INCLUDE_REDACTION_METADATA and 'redaction_stats' in locals() and redaction_stats:
        #         result["redaction_stats"] = redaction_stats
        # except Exception:
        #     pass


        return result