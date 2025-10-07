"""
Tool-specific prompts for guiding LLM tool usage.
"""
from .base import PromptTemplate, PromptVersion


class ToolPrompts:
    """Tool-specific prompts and usage guidelines."""
    
    @staticmethod
    def get_tool_selection_prompt() -> PromptTemplate:
        """Get prompt for intelligent tool selection."""
        return PromptTemplate(
            content="""TOOL SELECTION GUIDE

Query: {query}

Available Tools:
{tool_descriptions}

Select the most appropriate tool(s) by considering:

1. Query Intent Analysis:
   - What is the user fundamentally asking for?
   - What type of data do they need?
   - What format should the response be in?

2. Tool Capability Matching:
   - Which tool provides the needed data?
   - Does one tool cover everything or do we need multiple?
   - Are there any tool combinations that work well together?

3. Parameter Extraction:
   - What parameters can we extract from the query?
   - Are there implicit parameters we should infer?
   - What are sensible defaults for missing parameters?

4. Efficiency Consideration:
   - Can we minimize the number of tool calls?
   - Should we call tools in parallel or sequence?
   - Are there redundant calls we can avoid?

Selected Tool(s): [Your selection]
Rationale: [Why this/these tool(s)]
Parameters: [Extracted parameters]""",
            version=PromptVersion.V2_0,
            description="Guide for selecting appropriate tools",
            tags=["tools", "selection", "reasoning"],
            variables={
                "query": "",
                "tool_descriptions": ""
            }
        )
    
    @staticmethod
    def get_parameter_extraction_prompt() -> PromptTemplate:
        """Get prompt for extracting tool parameters from queries."""
        return PromptTemplate(
            content="""PARAMETER EXTRACTION

User Query: {query}
Target Tool: {tool_name}

Required Parameters:
{required_parameters}

Optional Parameters:
{optional_parameters}

Extract parameters following these rules:

1. EXACT MATCHING:
   - Use exact file names, domains, dates from query
   - Don't normalize or modify user-provided values
   - Preserve quotes, spaces as user intended

2. TEMPORAL EXTRACTION:
   - "from DATE" → start_date=DATE, end_date=today
   - "since DATE" → start_date=DATE, end_date=today  
   - "on DATE" → start_date=DATE, end_date=DATE
   - "between X and Y" → start_date=X, end_date=Y
   - No dates mentioned → omit date parameters

3. DOMAIN EXTRACTION:
   - "customer domain" → domain_name="customer"
   - Remove "_domain" suffix if present
   - Use word before "domain" keyword

4. REPORT TYPE DETECTION:
   - "success rate" → report_type="success"
   - "failure rate" → report_type="failure"
   - "analyze" or "both" → report_type="both"

5. CHART TYPE DETECTION:
   - "pie chart" → chart_type="pie"
   - "donut" → chart_type="donut"
   - "line chart" → chart_type="line"
   - "stacked" → chart_type="stacked"
   - Default → chart_type="bar"

Extracted Parameters:
{extracted_parameters}""",
            version=PromptVersion.V2_0,
            description="Parameter extraction from user queries",
            tags=["tools", "parameters", "extraction"],
            variables={
                "query": "",
                "tool_name": "",
                "required_parameters": "",
                "optional_parameters": "",
                "extracted_parameters": "{}"
            }
        )
    
    @staticmethod
    def get_tool_result_validation_prompt() -> PromptTemplate:
        """Get prompt for validating tool results."""
        return PromptTemplate(
            content="""TOOL RESULT VALIDATION

Tool Called: {tool_name}
Parameters Used: {parameters}

Result Received:
{tool_result}

Validate the result:

1. COMPLETENESS CHECK:
   ☐ Did the tool return data?
   ☐ Is the data structure as expected?
   ☐ Are key fields present?
   ☐ Is the data sufficient to answer the query?

2. QUALITY CHECK:
   ☐ Are values within reasonable ranges?
   ☐ Are there any obvious errors or anomalies?
   ☐ Is the data consistent with the query?
   ☐ Are timestamps and dates valid?

3. RELEVANCE CHECK:
   ☐ Does this data answer the user's question?
   ☐ Are there any missing pieces?
   ☐ Do we need additional tool calls?
   ☐ Can we form a complete response?

4. ERROR HANDLING:
   ☐ Are there error messages in the result?
   ☐ Did the tool fail or return empty data?
   ☐ Should we retry with different parameters?
   ☐ Should we try an alternative tool?

Validation Status: {validation_status}
Next Action: {next_action}""",
            version=PromptVersion.V2_0,
            description="Validation checklist for tool results",
            tags=["tools", "validation", "quality"],
            variables={
                "tool_name": "",
                "parameters": "{}",
                "tool_result": "",
                "validation_status": "Pending",
                "next_action": "TBD"
            }
        )
    
    @staticmethod
    def get_no_data_handling_prompt() -> PromptTemplate:
        """Get prompt for handling no-data scenarios gracefully."""
        return PromptTemplate(
            content="""NO DATA SCENARIO HANDLING

Query: {query}
Tool: {tool_name}
Parameters: {parameters}
Result: No data found

Provide a helpful response that:

1. ACKNOWLEDGES:
   - Confirm what the user was looking for
   - State clearly that no data was found
   - Reference the specific file/domain/date range

2. PROVIDES CONTEXT:
   - Explain possible reasons:
     • File/domain might not exist in the system
     • Date range might be outside data coverage
     • Filters might be too restrictive
     • Data might not have been uploaded yet

3. SUGGESTS ALTERNATIVES:
   - Try different date ranges
   - Check file/domain name spelling
   - Remove optional filters
   - List available files/domains if possible
   - Suggest broader queries

4. MAINTAINS HELPFULNESS:
   - Don't just say "no data found"
   - Be constructive and solution-oriented
   - Offer to help with related queries
   - Keep tone positive and supportive

Example Response:
"I searched for {specific_entity} but didn't find any data. This could mean {possible_reasons}. You might want to try {suggestions}. I'm here to help if you'd like to explore other options!"

Your response: {response}""",
            version=PromptVersion.V2_0,
            description="Handling no-data scenarios gracefully",
            tags=["tools", "error-handling", "user-experience"],
            variables={
                "query": "",
                "tool_name": "",
                "parameters": "{}",
                "response": ""
            }
        )
    
    @staticmethod
    def get_multi_tool_coordination_prompt() -> PromptTemplate:
        """Get prompt for coordinating multiple tool calls."""
        return PromptTemplate(
            content="""MULTI-TOOL COORDINATION

Query requires multiple tools: {query}

Tools to coordinate:
{tools_list}

Coordination strategy:

1. DEPENDENCY ANALYSIS:
   - Which tools can run in parallel?
   - Which tools depend on others' outputs?
   - What's the optimal execution order?

2. DATA FLOW:
   Tool 1: {tool_1} → Provides: {output_1}
   Tool 2: {tool_2} → Needs: {input_2} → Provides: {output_2}
   Tool 3: {tool_3} → Needs: {input_3} → Provides: {output_3}

3. EXECUTION PLAN:
   Phase 1: [Parallel tools]
   Phase 2: [Tools dependent on Phase 1]
   Phase 3: [Final synthesis]

4. RESULT COMBINATION:
   - How to merge results from multiple tools?
   - Which results take priority?
   - How to present combined insights?

Execution sequence: {execution_sequence}""",
            version=PromptVersion.V2_0,
            description="Coordinating multiple tool calls efficiently",
            tags=["tools", "coordination", "multi-step"],
            variables={
                "query": "",
                "tools_list": "",
                "tool_1": "",
                "output_1": "",
                "tool_2": "",
                "input_2": "",
                "output_2": "",
                "tool_3": "",
                "input_3": "",
                "output_3": "",
                "execution_sequence": ""
            }
        )
    
    @staticmethod
    def get_tool_error_recovery_prompt() -> PromptTemplate:
        """Get prompt for recovering from tool errors."""
        return PromptTemplate(
            content="""TOOL ERROR RECOVERY

Error encountered:
Tool: {tool_name}
Parameters: {parameters}
Error: {error_message}

Recovery strategy:

1. ERROR CLASSIFICATION:
   - Is this a parameter error? → Fix parameters and retry
   - Is this a data error? → Try alternative tools or queries
   - Is this a system error? → Inform user gracefully
   - Is this a timeout? → Retry with smaller scope

2. RETRY LOGIC:
   - Should we retry? {retry_decision}
   - With same parameters? {same_params}
   - With modified parameters? {modified_params}
   - Maximum retries: 2

3. FALLBACK OPTIONS:
   - Alternative tool: {alternative_tool}
   - Simplified query: {simplified_query}
   - Partial results: {partial_results_available}
   - User notification: {user_message}

4. GRACEFUL DEGRADATION:
   - Can we answer part of the query?
   - Can we provide related information?
   - Can we suggest next steps?

Recovery action: {recovery_action}""",
            version=PromptVersion.V2_0,
            description="Recovering from tool execution errors",
            tags=["tools", "error-recovery", "resilience"],
            variables={
                "tool_name": "",
                "parameters": "{}",
                "error_message": "",
                "retry_decision": "TBD",
                "same_params": "false",
                "modified_params": "{}",
                "alternative_tool": "None",
                "simplified_query": "",
                "partial_results_available": "false",
                "user_message": "",
                "recovery_action": "TBD"
            }
        )
