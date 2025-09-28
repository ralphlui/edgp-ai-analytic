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

logger = logging.getLogger(__name__)


class AnalyticService:
    """Handles LLM tool selection, execution, chart generation, and interpretation."""

    # Use the centralized tool registry
    TOOLS = ANALYSIS_TOOLS

    @staticmethod
    async def process_query(prompt: str, session_id: str = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                session_id=session_id,
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
    def _create_enhanced_system_message(current_date: str, reference_context: str, conversation_insights: str) -> str:
        """Create comprehensive system message with reference resolution instructions."""
        from app.config import SYSTEM
        
        base_system = SYSTEM.format(current_date=current_date)
        
        enhanced_instructions = f"""

REFERENCE RESOLUTION INSTRUCTIONS:
When users refer to "that file", "it", "the data", "that domain", or similar pronouns/references, 
use the context below to resolve these references to specific file names or domains.

{reference_context}

CONVERSATION CONTEXT:
{conversation_insights}

IMPORTANT: When selecting tools and parameters, always resolve ambiguous references to specific 
file names or domain names based on the context above. If a reference is unclear, ask for clarification.
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
    async def process_query_with_report_type(prompt: str, report_type: str, session_id: str = None,
                                           conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process analytic query with enhanced LLM context and reference resolution.
        """
        try:
            # Import here to avoid circular imports
            from .graph_builder import build_app
            from app.generators.chart_generator import chart_generator
            from app.utils.sanitization import sanitize_text_input
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            from datetime import date
            from app.services.memory import memory_service

            logger.info(f"Processing prompt: '{prompt[:100]}...'")
            logger.info(f"Report type: '{report_type}'")

            app_graph = build_app()

        except Exception as e:
            return {"success": False, "message": str(e), "chart_image": None}

        # Get current date for context
        current_date = date.today().strftime('%Y-%m-%d')

        # Get reference context from memory service
        reference_context = memory_service.get_reference_context_for_llm(session_id) if session_id else ""
        print(f"Reference context for LLM: {reference_context}")
        
        # Extract conversation insights
        conversation_insights = AnalyticService._extract_conversation_insights(conversation_history or [])
        print(f"Conversation insights for LLM: {conversation_insights}")
        print(f"Conversation conversation_history for LLM: {conversation_history}")

        # Create enhanced system message with reference resolution
        system_message = AnalyticService._create_enhanced_system_message(
            current_date, reference_context, conversation_insights
        )
        
        messages = [SystemMessage(content=system_message)]

        # Add simplified conversation history
        if conversation_history:
            recent_interactions = conversation_history[-2:]  # Reduced to avoid context overload
            
            for interaction in recent_interactions:
                if interaction.get("user_prompt"):
                    safe_prompt = sanitize_text_input(interaction['user_prompt'], 150)
                    messages.append(HumanMessage(content=safe_prompt))
                
                if interaction.get("response_summary", {}).get("message"):
                    # Keep only key metrics from previous responses
                    prev_response = interaction["response_summary"]["message"]
                    
                    # Extract key information (success rates, file names, etc.)
                    import re
                    key_info = []
                    
                    # Extract percentage rates
                    rates = re.findall(r'\d+\.?\d*%', prev_response)
                    if rates:
                        key_info.append(f"Rates: {', '.join(rates[:3])}")
                    
                    # Extract file/domain info
                    if interaction["response_summary"].get("file_name"):
                        key_info.append(f"File: {interaction['response_summary']['file_name']}")
                    
                    if key_info:
                        summary = " | ".join(key_info)
                        messages.append(AIMessage(content=f"Previous: {summary}"))

        # Add the current prompt
        safe_prompt = sanitize_text_input(prompt, 300)
        messages.append(HumanMessage(content=safe_prompt))

        # Prepare state with enhanced context
        # Note: Don't pass pre-classified report_type to LLM - let LLM decide based on prompt
        state = {
            "messages": messages,
            "session_id": session_id,
            
            # Add processing hints for LLM
            "processing_context": {
                "has_reference_context": bool(reference_context),
                "conversation_length": len(conversation_history or []),
                "current_date": current_date
            }
        }

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
        llm_detected_report_type = None  # Track LLM-detected report type

        for m in compiled_result.get("messages", []):
            if hasattr(m, 'content') and hasattr(m, '__class__'):
                # Handle ToolMessage
                if m.__class__.__name__ == 'ToolMessage':
                    try:
                        tool_data = json.loads(m.content) if isinstance(m.content, str) else m.content
                        tool_results.append(tool_data)

                        if tool_data.get("success") and "chart_data" in tool_data:
                            chart_data = tool_data.get("chart_data", [])
                            file_name = tool_data.get("file_name")
                            domain_name = tool_data.get("domain_name")
                            row_count = tool_data.get("row_count", 0)

                            if tool_data.get("chart_type_requested"):
                                chart_type = tool_data.get("chart_type", "bar")
                                logger.info(f"LLM detected chart_type: {tool_data}")

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
                elif m.__class__.__name__ == 'AIMessage' and hasattr(m, 'tool_calls'):
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
                            
                            # Extract report_type from tool call arguments
                            if args.get('report_type'):
                                llm_detected_report_type = args.get('report_type')
                                logger.info(f"LLM tool call specified report_type: {llm_detected_report_type}")

        # Smart decision logic: prioritize clear pre-classification over ambiguous LLM detection
        final_report_type = None
        
        # If pre-classification is confident (success/failure) but LLM says "both", trust pre-classification
        if report_type in ["success", "failure"] and llm_detected_report_type == "both":
            final_report_type = report_type
            logger.info(f"Override: Pre-classification '{report_type}' overrides LLM 'both' for clear cases")
        
        # If LLM has a specific decision (success/failure), use it
        elif llm_detected_report_type in ["success", "failure"]:
            final_report_type = llm_detected_report_type
            logger.info(f"Using LLM-detected specific report_type: {llm_detected_report_type}")
        
        # If LLM says "both" and pre-classification agrees or is uncertain, use "both"
        elif llm_detected_report_type == "both":
            final_report_type = "both"
            logger.info(f"Using LLM-detected 'both' report_type")
        
        # Fallback to pre-classification
        else:
            final_report_type = report_type if report_type != "uncertain" else "both"
            logger.info(f"Fallback to pre-classified: {report_type}")
        
        logger.info(f"Final report_type: {final_report_type} (LLM: {llm_detected_report_type}, Pre-classified: {report_type})")

        # Filter chart data based on final report type  
        # Store original only if debug mode is enabled to save memory
        from app.config import DEBUG
        original_chart_data = chart_data.copy() if DEBUG else None
        filtered_chart_data = AnalyticService.filter_chart_data_by_report_type(chart_data, final_report_type)

        # Generate chart image if we have data
        chart_image = None
        chart_generated = False
        if filtered_chart_data:
            try:
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
                if m.__class__.__name__ == 'AIMessage' and not hasattr(m, 'tool_calls'):
                    interpretation = m.content
                    break
        
        # Fallback if no interpretation found
        if not interpretation:
            from app.utils.formatting import format_basic_message
            interpretation = format_basic_message(
                chart_data=filtered_chart_data,
                file_name=file_name or domain_name,
                row_count=row_count,
                chart_type=chart_type,
                report_type=final_report_type,
                date_filter_used=date_filter_used,
                original_chart_data=original_chart_data
            )

            #   "tool": result.get("tool"),
            #                 "file_name": result.get("file_name"),
            #                 "domain_name": result.get("domain_name"),
            #                 "row_count": result.get("row_count", 0),
            #                 "report_type": result.get("report_type", "both"),
            #                 "message": result.get("message", "")[:200]



        result = {
            "success": True,
            "message": interpretation,
            "chart_image": chart_image,
            "file_name": file_name,
            "domain_name": domain_name,
            "row_count": row_count,
            "report_type": final_report_type,
            "tool":""
        }


        return result