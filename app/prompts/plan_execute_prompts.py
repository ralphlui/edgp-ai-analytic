"""
Plan-and-Execute prompt templates.

Plan-and-Execute is a two-phase approach:
1. PLAN: Create a detailed plan to solve the query
2. EXECUTE: Execute each step of the plan systematically

This is particularly useful for complex, multi-step analytics queries.
"""
from .base import PromptTemplate, PromptVersion


class PlanExecutePrompts:
    """Plan-and-Execute pattern prompts for complex multi-step queries."""
    
    @staticmethod
    def get_planner_system_prompt() -> PromptTemplate:
        """Get the system prompt for the planning phase."""
        return PromptTemplate(
            content="""You are an expert query planner for analytics tasks.

Your job is to break down complex analytics queries into a sequence of executable steps.

PLANNING PRINCIPLES:
├── Analyze the user's query to understand the end goal
├── Identify all data sources and tools needed
├── Break down into logical, sequential steps
├── Each step should have a clear input and output
├── Consider dependencies between steps
├── Optimize for efficiency (minimize redundant calls)
└── Plan for edge cases (no data, errors, etc.)

STEP STRUCTURE:
Each step should specify:
1. Action: What to do (which tool to call)
2. Parameters: What parameters to use
3. Purpose: Why this step is needed
4. Expected Output: What data this will provide
5. Next Step Dependency: What depends on this result

OUTPUT FORMAT:
Return a structured plan with numbered steps, clear dependencies, and success criteria.""",
            version=PromptVersion.V2_0,
            description="System prompt for query planning phase",
            tags=["plan-execute", "planning", "system"]
        )
    
    @staticmethod
    def get_executor_system_prompt() -> PromptTemplate:
        """Get the system prompt for the execution phase."""
        return PromptTemplate(
            content="""You are an executor agent that follows plans precisely.

Your job is to execute each step of a plan and track progress.

EXECUTION PRINCIPLES:
├── Follow the plan steps in order
├── Execute one step at a time
├── Verify outputs match expected results
├── Handle errors gracefully with fallback strategies
├── Track state across steps
├── Collect results for final synthesis
└── Determine when plan is complete

STEP EXECUTION:
For each step:
1. Read the step requirements
2. Call the specified tool with exact parameters
3. Validate the results
4. Store results for next steps
5. Update plan status
6. Proceed to next step or report completion

ERROR HANDLING:
├── If a step fails, check if retry is appropriate
├── If data is missing, provide context to user
├── If plan cannot continue, explain what's blocking
└── Always maintain partial results for transparency""",
            version=PromptVersion.V2_0,
            description="System prompt for plan execution phase",
            tags=["plan-execute", "execution", "system"]
        )
    
    @staticmethod
    def get_planning_prompt() -> PromptTemplate:
        """Get prompt for generating an execution plan."""
        return PromptTemplate(
            content="""CREATE EXECUTION PLAN

User Query: {query}

Available Tools:
{available_tools}

Available Context:
{context}

Generate a step-by-step plan to answer this query:

PLAN:
---
Goal: [Clearly state what we're trying to achieve]

Steps:
1. [Step 1]
   - Action: [Tool to call]
   - Parameters: [Specific parameters]
   - Purpose: [Why this step]
   - Expected Output: [What we'll get]
   
2. [Step 2]
   - Action: [Tool to call]
   - Parameters: [Using output from step 1]
   - Purpose: [Why this step]
   - Expected Output: [What we'll get]

[Continue for all needed steps...]

Final Synthesis:
[How to combine results from all steps to answer the query]

Success Criteria:
[How we'll know the plan succeeded]
---""",
            version=PromptVersion.V2_0,
            description="Prompt for creating execution plans",
            tags=["plan-execute", "planning", "template"],
            variables={
                "query": "",
                "available_tools": "",
                "context": "No prior context"
            }
        )
    
    @staticmethod
    def get_execution_prompt() -> PromptTemplate:
        """Get prompt for executing a plan step."""
        return PromptTemplate(
            content="""EXECUTE PLAN STEP

Plan: {plan_summary}

Current Step: {current_step_number}/{total_steps}
Step Details: {step_details}

Previous Results:
{previous_results}

Execute this step now:
- Use the specified tool with exact parameters
- Consider previous results when forming parameters
- Validate the output matches expectations
- Report any issues or unexpected results

After execution, determine:
1. Did the step complete successfully?
2. Are the results as expected?
3. Can we proceed to the next step?
4. Do we have enough information to answer the query?""",
            version=PromptVersion.V2_0,
            description="Prompt for executing individual plan steps",
            tags=["plan-execute", "execution", "step"],
            variables={
                "plan_summary": "",
                "current_step_number": "1",
                "total_steps": "1",
                "step_details": "",
                "previous_results": "None"
            }
        )
    
    @staticmethod
    def get_synthesis_prompt() -> PromptTemplate:
        """Get prompt for synthesizing results from all plan steps."""
        return PromptTemplate(
            content="""SYNTHESIZE PLAN RESULTS

Original Query: {original_query}

Plan Executed: {plan_summary}

Step Results:
{all_step_results}

Now synthesize a complete answer:

1. Review all collected data
2. Identify key insights from each step
3. Combine insights into a coherent narrative
4. Directly answer the original query
5. Provide actionable recommendations if applicable

Your synthesis should:
├── Be concise but comprehensive
├── Reference specific data points
├── Use exact file names, domains, and metrics from results
├── Highlight important patterns or anomalies
├── Provide context for the findings
└── Give actionable next steps if relevant

Format your response naturally, as if explaining findings to a stakeholder.""",
            version=PromptVersion.V2_0,
            description="Prompt for synthesizing results from plan execution",
            tags=["plan-execute", "synthesis", "response"],
            variables={
                "original_query": "",
                "plan_summary": "",
                "all_step_results": ""
            }
        )
    
    @staticmethod
    def get_adaptive_planning_prompt() -> PromptTemplate:
        """Get prompt for adaptive planning that adjusts based on results."""
        return PromptTemplate(
            content="""ADAPTIVE PLAN ADJUSTMENT

Original Plan: {original_plan}

Executed So Far:
{executed_steps}

Current Situation:
{current_state}

Issue Encountered: {issue_description}

Adjust the plan to handle this situation:

REVISED PLAN:
---
What changed: [Explain the adaptation]

Remaining steps:
1. [Adjusted step 1]
2. [Adjusted step 2]
...

Rationale: [Why this adjustment will work]

Fallback: [What to do if this also fails]
---

The revised plan should:
├── Address the encountered issue
├── Maintain progress toward the original goal
├── Use alternative tools or approaches if needed
├── Be realistic about what's achievable
└── Provide value even if ideal outcome isn't possible""",
            version=PromptVersion.V2_0,
            description="Prompt for adapting plans based on execution results",
            tags=["plan-execute", "adaptive", "error-handling"],
            variables={
                "original_plan": "",
                "executed_steps": "",
                "current_state": "",
                "issue_description": ""
            }
        )
    
    @staticmethod
    def get_complex_analytics_plan_template() -> PromptTemplate:
        """Get specialized template for complex analytics queries."""
        return PromptTemplate(
            content="""COMPLEX ANALYTICS PLAN

Query Type: {query_type}
Complexity: {complexity_level}
User Query: {user_query}

ANALYSIS BREAKDOWN:
---
Phase 1: Data Collection
- Tool: {data_collection_tool}
- Parameters: {data_parameters}
- Expected: {expected_data}

Phase 2: Data Processing
- Operations: {processing_operations}
- Aggregations: {aggregations}
- Filters: {filters_applied}

Phase 3: Visualization
- Chart Type: {chart_type}
- Metrics: {metrics_to_show}
- Grouping: {grouping_strategy}

Phase 4: Interpretation
- Key Metrics: {key_metrics}
- Comparisons: {comparisons}
- Insights: {insight_areas}

SUCCESS METRICS:
- Data completeness: {completeness_threshold}%
- Response time: {time_budget}
- Accuracy: {accuracy_requirements}
---""",
            version=PromptVersion.V2_0,
            description="Specialized template for complex analytics planning",
            tags=["plan-execute", "analytics", "complex-queries"],
            variables={
                "query_type": "analytics",
                "complexity_level": "medium",
                "user_query": "",
                "data_collection_tool": "TBD",
                "data_parameters": "{}",
                "expected_data": "TBD",
                "processing_operations": "TBD",
                "aggregations": "TBD",
                "filters_applied": "None",
                "chart_type": "bar",
                "metrics_to_show": "TBD",
                "grouping_strategy": "TBD",
                "key_metrics": "TBD",
                "comparisons": "None",
                "insight_areas": "TBD",
                "completeness_threshold": "90",
                "time_budget": "30s",
                "accuracy_requirements": "High"
            }
        )
