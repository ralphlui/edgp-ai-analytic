"""
ReAct (Reasoning, Acting, Observing) prompt templates.

ReAct is a paradigm where the LLM:
1. Reasons about the current state
2. Acts by calling tools
3. Observes the results
4. Repeats until the task is complete
"""
from .base import PromptTemplate, PromptVersion


class ReActPrompts:
    """ReAct pattern prompts for systematic reasoning and action."""
    
    @staticmethod
    def get_react_system_prompt() -> PromptTemplate:
        """Get the ReAct system prompt that guides reasoning-action cycles."""
        return PromptTemplate(
            content="""You follow the ReAct (Reasoning, Acting, Observing) pattern to solve problems systematically.

REACT CYCLE:
1. REASON: Think through what you know and what you need to find out
2. ACT: Use tools to gather information or perform actions
3. OBSERVE: Analyze the results from your actions
4. REPEAT: Continue until you have enough information to answer

REASONING GUIDELINES:
├── State what you currently know from the conversation
├── Identify what information is missing to answer the query
├── Determine which tool(s) would provide the needed information
├── Consider if you need multiple steps or can answer directly
└── Check if results are sufficient or if more investigation is needed

ACTION GUIDELINES:
├── Choose the most appropriate tool for the current need
├── Extract parameters carefully from user query
├── Use exact values from the query (file names, dates, domains)
├── Apply filters only when explicitly mentioned
└── Call tools with all required parameters

OBSERVATION GUIDELINES:
├── Analyze what the tool returned
├── Check if data is sufficient to answer the query
├── Identify any patterns, trends, or anomalies
├── Determine if additional tool calls are needed
└── Prepare insights for the final response

COMPLETION CRITERIA:
├── You have all data needed to answer the user's question
├── No further tool calls would add value
├── You can provide actionable insights
└── The response directly addresses the user's intent""",
            version=PromptVersion.V2_0,
            description="ReAct system prompt for reasoning-action-observation cycles",
            tags=["react", "reasoning", "system"]
        )
    
    @staticmethod
    def get_reasoning_prompt() -> PromptTemplate:
        """Get prompt for the reasoning phase."""
        return PromptTemplate(
            content="""REASONING PHASE:

Current Query: {query}

What I know:
{known_facts}

What I need to find out:
{missing_information}

Next steps:
{planned_actions}

Tool selection rationale:
{tool_rationale}""",
            version=PromptVersion.V2_0,
            description="Structured reasoning phase prompt",
            tags=["react", "reasoning", "planning"],
            variables={
                "query": "",
                "known_facts": "- No prior information",
                "missing_information": "- To be determined",
                "planned_actions": "- To be determined",
                "tool_rationale": "- To be determined"
            }
        )
    
    @staticmethod
    def get_observation_prompt() -> PromptTemplate:
        """Get prompt for the observation phase."""
        return PromptTemplate(
            content="""OBSERVATION PHASE:

Tool called: {tool_name}
Parameters used: {parameters}

Results summary:
{results_summary}

Key findings:
{key_findings}

Sufficiency check:
{sufficiency_analysis}

Next action needed:
{next_action}""",
            version=PromptVersion.V2_0,
            description="Structured observation phase prompt",
            tags=["react", "observation", "analysis"],
            variables={
                "tool_name": "Unknown",
                "parameters": "{}",
                "results_summary": "No results",
                "key_findings": "- To be analyzed",
                "sufficiency_analysis": "- To be determined",
                "next_action": "- To be determined"
            }
        )
    
    @staticmethod
    def get_react_interpretation_prompt() -> PromptTemplate:
        """Get ReAct-enhanced interpretation prompt."""
        return PromptTemplate(
            content="""Apply ReAct reasoning to interpret these results:

USER QUERY: {user_query}

REASONING:
- What was the user asking for?
- What data did we retrieve?
- Does this fully answer their question?

OBSERVATION:
Tool Results:
{tool_results}

Context Insights:
{context_insights}

ACTION - Your Response:
- Directly address the user's specific question
- Use exact entities mentioned (file names, domains, dates)
- Highlight key metrics and patterns
- Provide actionable insights
- Keep it concise (3-4 sentences maximum)

Remember: Base your response ONLY on the actual data retrieved, never fabricate information.""",
            version=PromptVersion.V2_0,
            description="ReAct-enhanced result interpretation prompt",
            tags=["react", "interpretation", "response"],
            variables={
                "user_query": "",
                "tool_results": "",
                "context_insights": ""
            }
        )
    
    @staticmethod
    def get_multi_step_react_prompt() -> PromptTemplate:
        """Get prompt for multi-step ReAct reasoning."""
        return PromptTemplate(
            content="""MULTI-STEP REACT ANALYSIS:

Query: {query}

Step 1 - REASON:
What do I need to accomplish? Break down the query into sub-tasks.

Step 2 - PLAN:
What sequence of actions will get me there?
1. First action: {step_1}
2. Second action: {step_2}
3. Final action: {step_3}

Step 3 - ACT:
Execute each step systematically, observing results after each action.

Step 4 - SYNTHESIZE:
Combine insights from all steps to form a complete answer.

Current Step: {current_step}
Previous Observations: {previous_observations}""",
            version=PromptVersion.V2_0,
            description="Multi-step ReAct reasoning for complex queries",
            tags=["react", "multi-step", "planning"],
            variables={
                "query": "",
                "step_1": "To be determined",
                "step_2": "To be determined", 
                "step_3": "To be determined",
                "current_step": "1",
                "previous_observations": "None"
            }
        )
