"""
Planner Agent - Creates execution plans for complex analytical queries.

This agent uses LLM to break down complex queries (comparison, aggregation, trend)
into structured execution plans that can be executed by the Query Executor.

Key Responsibilities:
- Analyze user intent and query structure
- Generate step-by-step execution plans
- Define dependencies between steps
- Optimize execution order for efficiency

NO TOOLS EXECUTED - Only plan generation using LLM.
"""
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger("planner_agent")


# ============================================================================
# PYDANTIC MODELS FOR PLAN STRUCTURE
# ============================================================================

class PlanStep(BaseModel):
    """
    A single step in an execution plan.
    
    Each step represents one action to be executed, with parameters
    and dependencies on previous steps.
    """
    step_id: int = Field(..., description="Unique step identifier (1, 2, 3...)")
    action: str = Field(..., description="Action type to execute (e.g., 'query_analytics', 'compare_results')")
    description: str = Field(..., description="Human-readable description of what this step does")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    depends_on: List[int] = Field(default_factory=list, description="List of step_ids that must complete before this step")
    critical: bool = Field(default=True, description="If True, plan execution stops if this step fails")


class ExecutionPlan(BaseModel):
    """
    Complete execution plan for a complex query.
    
    Contains all steps, metadata, and execution configuration.
    """
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}", description="Unique plan identifier")
    query_type: str = Field(..., description="Type of query: 'comparison', 'aggregation', 'trend', 'multi_step'")
    intent: str = Field(..., description="User's intent: 'success_rate', 'failure_rate', 'general_query'")
    steps: List[PlanStep] = Field(..., description="Ordered list of execution steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional plan metadata")


# ============================================================================
# AVAILABLE ACTIONS CATALOG
# ============================================================================

AVAILABLE_ACTIONS = """
Available Actions (tools that can be executed):

1. query_analytics
   - Purpose: Query analytics data from a single target (domain or file)
   - Required params:
     * target (str): Target name (e.g., "customer.csv", "payment", "transactions.json")
     * metric_type (str): "success_rate" or "failure_rate"
   - Returns: Analytics data with success/failure metrics
   - Example: {{"target": "customer.csv", "metric_type": "success_rate"}}

2. compare_results
   - Purpose: Compare results from multiple previous query steps
   - Required params:
     * compare_steps (list[int]): List of step_ids to compare (e.g., [1, 2])
     * metric (str): "success_rate" or "failure_rate"
   - Returns: Comparison data with winner, differences, and comparison_details
   - Example: {{"compare_steps": [1, 2], "metric": "success_rate"}}

3. generate_chart
   - Purpose: Create visualization from comparison data
   - Required params:
     * comparison_step_id (int): Step ID that produced comparison data
   - Returns: Base64-encoded chart image
   - Example: {{"comparison_step_id": 3}}

4. format_response
   - Purpose: Generate natural language response using LLM
   - Required params:
     * comparison_step_id (int): Step ID with comparison data
     * chart_step_id (int, optional): Step ID with chart image
   - Returns: Natural language message with chart
   - Example: {{"comparison_step_id": 3, "chart_step_id": 4}}
"""


# ============================================================================
# PLANNER SYSTEM PROMPT
# ============================================================================

PLANNER_SYSTEM_PROMPT = f"""You are an expert query planner for analytics systems. Your job is to create 
efficient, step-by-step execution plans for analytical queries.

{AVAILABLE_ACTIONS}

PLANNING RULES:

1. **Step IDs**: Must be sequential integers starting from 1
2. **Dependencies**: Use depends_on to specify prerequisite steps
3. **Parallel Execution**: Steps with no mutual dependencies can run in parallel
4. **Critical Steps**: Mark data retrieval and comparison steps as critical=true
5. **Final Steps**: Always end with generate_chart and format_response
6. **Efficiency**: Minimize redundant queries - one query per target

QUERY PATTERNS:

Pattern 1: Simple Comparison (2 targets)
- Step 1: query_analytics for target A
- Step 2: query_analytics for target B (can run parallel with Step 1)
- Step 3: compare_results (inputs: [1, 2])
- Step 4: generate_chart (data_source: 3)
- Step 5: format_response (data_source: 3, chart: 4)

Pattern 2: Multi-Target Comparison (3+ targets)
- Steps 1-N: query_analytics for each target (all parallel)
- Step N+1: compare_results (inputs: [1, 2, ..., N])
- Step N+2: generate_chart (data_source: N+1)
- Step N+3: format_response (data_source: N+1, chart: N+2)

OUTPUT FORMAT:

Return ONLY valid JSON matching this structure:
{{{{
  "plan_id": "plan-abc123",
  "query_type": "comparison" | "aggregation" | "trend",
  "intent": "success_rate" | "failure_rate",
  "steps": [
    {{{{
      "step_id": 1,
      "action": "query_analytics",
      "description": "Query success rate for customer.csv",
      "params": {{{{"target": "customer.csv", "metric_type": "success_rate"}}}},
      "depends_on": [],
      "critical": true
    }}}},
    ...
  ],
  "metadata": {{{{
    "estimated_duration": "2-3 seconds",
    "complexity": "medium"
  }}}}
}}}}

EXAMPLE: Compare success rates between customer.csv and payment.csv
{{{{
  "plan_id": "plan-001",
  "query_type": "comparison",
  "intent": "success_rate",
  "steps": [
    {{{{
      "step_id": 1,
      "action": "query_analytics",
      "description": "Query success rate for customer.csv",
      "params": {{{{"target": "customer.csv", "metric_type": "success_rate"}}}},
      "depends_on": [],
      "critical": true
    }}}},
    {{{{
      "step_id": 2,
      "action": "query_analytics",
      "description": "Query success rate for payment.csv",
      "params": {{{{"target": "payment.csv", "metric_type": "success_rate"}}}},
      "depends_on": [],
      "critical": true
    }}}},
    {{{{
      "step_id": 3,
      "action": "compare_results",
      "description": "Compare success rates between the two targets",
      "params": {{{{"compare_steps": [1, 2], "metric": "success_rate"}}}},
      "depends_on": [1, 2],
      "critical": true
    }}}},
    {{{{
      "step_id": 4,
      "action": "generate_chart",
      "description": "Create comparison bar chart",
      "params": {{{{"comparison_step_id": 3}}}},
      "depends_on": [3],
      "critical": false
    }}}},
    {{{{
      "step_id": 5,
      "action": "format_response",
      "description": "Generate natural language summary",
      "params": {{{{"comparison_step_id": 3, "chart_step_id": 4}}}},
      "depends_on": [3],
      "critical": false
    }}}}
  ],
  "metadata": {{{{
    "estimated_duration": "2-3 seconds",
    "complexity": "medium",
    "targets_count": 2
  }}}}
}}}}

Now create an optimal execution plan based on the user's query.
"""


# ============================================================================
# PLANNER AGENT FUNCTIONS
# ============================================================================

def create_execution_plan(
    intent: str,
    comparison_targets: Optional[List[str]] = None,
    user_query: str = "",
    query_type: str = "comparison"
) -> ExecutionPlan:
    """
    Create an execution plan for a complex query using LLM.
    
    This function uses an LLM to intelligently generate a structured execution
    plan based on the user's query and extracted parameters.
    
    Args:
        intent: User's intent (e.g., "success_rate", "failure_rate")
        comparison_targets: List of targets to compare (e.g., ["customer.csv", "payment.csv"])
        user_query: Original user question for context
        query_type: Type of query (default: "comparison")
    
    Returns:
        ExecutionPlan object with structured steps
    
    Raises:
        ValueError: If LLM fails to generate valid plan
        json.JSONDecodeError: If LLM output is not valid JSON
    
    Example:
        >>> plan = create_execution_plan(
        ...     intent="success_rate",
        ...     comparison_targets=["customer.csv", "payment.csv"],
        ...     user_query="Compare success rates between customer and payment",
        ...     query_type="comparison"
        ... )
        >>> print(f"Plan has {len(plan.steps)} steps")
        Plan has 5 steps
    """
    logger.info("=" * 60)
    logger.info("PLANNER AGENT: Creating execution plan")
    logger.info("=" * 60)
    logger.info(f"User Query: {user_query}")
    logger.info(f"Intent: {intent}")
    logger.info(f"Query Type: {query_type}")
    logger.info(f"Comparison Targets: {comparison_targets}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,  # Deterministic planning
        api_key=OPENAI_API_KEY
    )
    
    # Build user prompt with query details
    user_prompt = f"""Create an execution plan for this analytical query:

**User Query**: "{user_query}"

**Extracted Parameters**:
- Intent: {intent}
- Query Type: {query_type}
- Comparison Targets: {comparison_targets or "None"}

**Requirements**:
1. Create optimal plan with minimal steps
2. Enable parallel execution where possible (steps with no dependencies)
3. Include chart generation for visualization
4. End with natural language response formatting
5. Return valid JSON only

Generate the execution plan now:
"""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", user_prompt)
    ])
    
    # Create chain
    chain = prompt | llm
    
    try:
        # Invoke LLM to generate plan
        logger.info("Invoking LLM to generate plan...")
        response = chain.invoke({})
        
        logger.info(f"LLM Response received ({len(response.content)} chars)")
        logger.debug(f"Raw LLM output:\n{response.content}")
        
        # Parse JSON response
        # Handle cases where LLM wraps JSON in markdown code blocks
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove closing ```
        content = content.strip()
        
        # Parse JSON
        plan_dict = json.loads(content)
        
        # Validate and create ExecutionPlan object
        plan = ExecutionPlan(**plan_dict)
        
        # Validate plan structure
        validate_plan(plan)
        
        logger.info("=" * 60)
        logger.info(f"PLAN CREATED: {plan.plan_id}")
        logger.info("=" * 60)
        logger.info(f"Query Type: {plan.query_type}")
        logger.info(f"Intent: {plan.intent}")
        logger.info(f"Total Steps: {len(plan.steps)}")
        
        for step in plan.steps:
            logger.info(f"  Step {step.step_id}: {step.action} - {step.description}")
            logger.info(f"    Params: {step.params}")
            logger.info(f"    Depends on: {step.depends_on if step.depends_on else 'None (can run immediately)'}")
            logger.info(f"    Critical: {step.critical}")
        
        logger.info("=" * 60)
        
        return plan
        
    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"LLM output was:\n{response.content}")
        raise ValueError(f"LLM did not return valid JSON: {e}")
    
    except Exception as e:
        logger.exception(f"Error creating execution plan: {e}")
        raise


def validate_plan(plan: ExecutionPlan) -> None:
    """
    Validate execution plan for correctness.
    
    Checks:
    - Step IDs are sequential
    - Dependencies reference valid step IDs
    - No circular dependencies
    - At least one step exists
    
    Args:
        plan: ExecutionPlan to validate
    
    Raises:
        ValueError: If plan is invalid
    """
    logger.info(f"Validating plan {plan.plan_id}...")
    
    # Check: At least one step
    if not plan.steps:
        raise ValueError("Plan has no steps")
    
    # Check: Step IDs are sequential starting from 1
    step_ids = [step.step_id for step in plan.steps]
    expected_ids = list(range(1, len(plan.steps) + 1))
    if step_ids != expected_ids:
        raise ValueError(f"Step IDs must be sequential 1-{len(plan.steps)}, got {step_ids}")
    
    # Check: Dependencies reference valid step IDs
    valid_ids = set(step_ids)
    for step in plan.steps:
        for dep_id in step.depends_on:
            if dep_id not in valid_ids:
                raise ValueError(f"Step {step.step_id} depends on non-existent step {dep_id}")
            if dep_id >= step.step_id:
                raise ValueError(f"Step {step.step_id} cannot depend on step {dep_id} (forward dependency)")
    
    # Check: No circular dependencies (simple check - no step depends on itself)
    for step in plan.steps:
        if step.step_id in step.depends_on:
            raise ValueError(f"Step {step.step_id} has circular dependency (depends on itself)")
    
    logger.info(f"Plan {plan.plan_id} is valid")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_comparison_plan(
    comparison_targets: List[str],
    intent: str = "success_rate",
    user_query: str = ""
) -> ExecutionPlan:
    """
    Convenience function to create a comparison plan.
    
    Args:
        comparison_targets: List of targets to compare
        intent: "success_rate" or "failure_rate"
        user_query: Original user query for context
    
    Returns:
        ExecutionPlan for comparison query
    """
    return create_execution_plan(
        intent=intent,
        comparison_targets=comparison_targets,
        user_query=user_query,
        query_type="comparison"
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test planner agent with sample queries."""
    
    print("\n" + "=" * 80)
    print("PLANNER AGENT TEST")
    print("=" * 80)
    
    # Test 1: Simple comparison
    print("\n📝 Test 1: Simple Comparison - 2 Targets")
    print("-" * 80)
    
    plan1 = create_comparison_plan(
        comparison_targets=["customer.csv", "payment.csv"],
        intent="success_rate",
        user_query="Compare success rates between customer.csv and payment.csv"
    )
    
    print(f"\n✅ Plan Created: {plan1.plan_id}")
    print(f"   Query Type: {plan1.query_type}")
    print(f"   Intent: {plan1.intent}")
    print(f"   Steps: {len(plan1.steps)}")
    
    print("\n📋 Execution Steps:")
    for step in plan1.steps:
        deps = f" (depends on: {step.depends_on})" if step.depends_on else " (no dependencies)"
        critical = "⚠️ CRITICAL" if step.critical else "ℹ️  optional"
        print(f"\n   Step {step.step_id}: {step.action} {critical}")
        print(f"   └─ {step.description}")
        print(f"   └─ Params: {step.params}")
        print(f"   └─ Dependencies: {deps}")
    
    # Test 2: Multi-target comparison
    print("\n" + "=" * 80)
    print("📝 Test 2: Multi-Target Comparison - 3 Targets")
    print("-" * 80)
    
    plan2 = create_comparison_plan(
        comparison_targets=["customer.csv", "payment.csv", "transactions.csv"],
        intent="failure_rate",
        user_query="Compare failure rates across customer, payment, and transactions"
    )
    
    print(f"\n✅ Plan Created: {plan2.plan_id}")
    print(f"   Query Type: {plan2.query_type}")
    print(f"   Intent: {plan2.intent}")
    print(f"   Steps: {len(plan2.steps)}")
    
    print("\n" + "=" * 80)
    print("✅ All Tests Complete")
    print("=" * 80)
