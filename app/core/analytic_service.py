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
        # logger.info(f"Base system prompt: {base_system}...")
        
        enhanced_instructions = f"""

🔒 STRICT SCOPE ENFORCEMENT:
- I am EXCLUSIVELY an analytics assistant for data analysis, reporting, and visualization
- If the user's query is NOT about data analytics (e.g., "What's the weather?", "Write me a story", "Help me code a function", "What is 2+2?"), I MUST respond with:
  
  "{SystemPrompts.get_non_analytics_fallback().format()}"

- ONLY process queries related to: data analysis, success/failure rates, charts, visualizations, file analytics, domain analytics, customer reports
- Do NOT attempt to answer general knowledge, coding, creative, or any non-analytics questions

CONVERSATION CONTEXT:
{conversation_insights}

CRITICAL: PRIORITIZE CURRENT USER PROMPT
- ALWAYS analyze and respond to the CURRENT user prompt first and foremost
- FIRST, determine if the current prompt is analytics-related or not
- If NOT analytics-related → Use the non-analytics fallback message and STOP
- If analytics-related → Continue with normal processing below
- For analytics queries: Conversation history is provided for context ONLY - do not let it override current request
- If current prompt is different from previous requests, follow the current prompt completely
- Only use conversation history to fill in gaps when current prompt is incomplete AND analytics-related

IMPORTANT GUIDELINES FOR TOOL USE:
- This system supports TWO types of analysis:
  1. SUCCESS/FAILURE RATE ANALYSIS: Requires file_name/domain_name + report_type ('success'/'failure'/'both')
  2. DOMAIN DISTRIBUTION ANALYSIS: Shows "how many [domain] by [column]" - only needs domain_name (no report_type needed)

- SMART CLARIFICATION RULES (based on CURRENT prompt):
  * If user asks for "report" without details → Ask which type of report and provide options:
    "I can generate several types of reports for you:
    
    📊 **Available Reports:**
    1. **Success/Failure Rate Report** - Analysis of success vs failure rates
       - For files: Specify file name (e.g., 'customer.csv')
       - For domains: Specify domain name (e.g., 'customer domain')
    
    2. **Domain Distribution Report** - Show data distribution by categories
       - Example: 'How many customers by country'
       - Example: 'Product distribution by category'
    
    3. **Data Quality Report** - Data quality validation metrics
    
    📈 **Available Chart Types:**
    - Bar Chart (default)
    - Pie Chart
    - Donut Chart
    - Line Chart
    - Stacked Chart
    
    What type of report would you like to generate? Please specify:
    - What data source (file name or domain)?
    - What type of analysis?
    - Preferred chart type (optional)?"
  
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
  * "What's the weather? or greeting messages such as hi, hello" → Use non-analytics fallback (NOT analytics-related)
  * "Write me a Python function" → Use non-analytics fallback (NOT analytics-related)
  * "Tell me a joke" → Use non-analytics fallback (NOT analytics-related)
  * "generate report" → ASK FOR REPORT TYPE with available options (analytics-related but incomplete)
  * "create a report" → ASK FOR REPORT TYPE with available options (analytics-related but incomplete)
  * "show me report" → ASK FOR REPORT TYPE with available options (analytics-related but incomplete)
  * "I need a report" → ASK FOR REPORT TYPE with available options (analytics-related but incomplete)
  * "show me data" → ASK FOR CLARIFICATION (analytics-related but vague)
  * "customer.csv" → ASK what type of analysis (file provided, but analysis type unclear)
  * "success rate" → ASK which file/domain (analysis type provided, but target unclear)
  * "success rate for that file" → ASK which specific file (vague reference)
  * "analyze the previous domain" → ASK which specific domain (vague reference)
  * "show me this data" → ASK which specific file/domain (vague reference)
  * "success rate for customer.csv" → PROCEED with file success/failure analysis
  * "generate success rate report for customer.csv as pie chart" → PROCEED with complete parameters
  * "how many customers by country" → PROCEED with domain distribution analysis

- For chart_type: Choose automatically based on data type, or use user-specified type.
"""
        
        return base_system + enhanced_instructions


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

                
        # Add the current prompt
        safe_prompt = sanitize_text_input(prompt, 300)
        messages.append(HumanMessage(content=safe_prompt))

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
        # Initialize variables for data extraction
        chart_data = []
        file_name = None
        domain_name = None
        row_count = 0
        date_filter_used = None
        chart_type = "bar"
        chart_type_source = "default"  # one of: default, auto, requested
        llm_detected_report_type = None  # Track LLM-detected report type

        # Extract data from messages (process in order: AI decisions first, then tool results)
        for m in compiled_result.get("messages", []):
            if not (hasattr(m, 'content') and hasattr(m, '__class__')):
                continue
            
            message_type = m.__class__.__name__
            
            # 1. Extract LLM's intent/decisions from AIMessage with tool_calls
            if message_type == 'AIMessage' and getattr(m, 'tool_calls', None):
                for tool_call in m.tool_calls:
                    args = tool_call.get('args', {})
                    # Extract report type detected by LLM
                    if args.get('report_type'):
                        llm_detected_report_type = args.get('report_type')
                        logger.info(f"Report type from LLM: {llm_detected_report_type}")
            
            # 2. Extract actual data from ToolMessage (tool execution results)
            elif message_type == 'ToolMessage':
                try:
                    tool_data = json.loads(m.content) if isinstance(m.content, str) else m.content
                    tool_results.append(tool_data)
                    
                    # Extract actual data from tool execution
                    chart_data = tool_data.get("chart_data", chart_data)
                    file_name = tool_data.get("file_name", file_name)
                    domain_name = tool_data.get("domain_name", domain_name)
                    row_count = tool_data.get("row_count", row_count)
                    
                    # Extract actual date filter used in database query (more reliable than LLM parsing)
                    if tool_data.get("date_filter"):
                        date_filter_used = tool_data["date_filter"]
                        logger.info(f"Date filter actually used: {date_filter_used}")
                    
                    # Tool may suggest chart type based on data analysis
                    #if "chart_type" in tool_data and chart_type_source != "requested":
                    if "chart_type" in tool_data :
                        chart_type = tool_data.get("chart_type", chart_type)
                        chart_type_source = "auto"
                        logger.info(f"Chart type auto-selected by tool: {chart_type}")
                    
                    logger.info(f"Tool executed - file: {file_name}, domain: {domain_name}, rows: {row_count}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool message as JSON: {e}")
                    tool_results.append(str(m.content))
                except Exception as e:
                    logger.warning(f"Error processing tool message: {e}")
                    tool_results.append(str(m.content))
    
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

        try:
                # Ensure chart_type has a sensible default
                #chart_type = chart_type or "bar"
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

        return result