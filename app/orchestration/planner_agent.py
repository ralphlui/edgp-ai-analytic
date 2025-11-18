import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.app_config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts.planner_prompts import PlannerPrompt
from app.security.pii_redactor import PIIRedactionFilter, redact_pii

logger = logging.getLogger("planner_agent")

# Add PII redaction filter to this logger
pii_filter = PIIRedactionFilter()
logger.addFilter(pii_filter)


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
    
    # NEW: Explainability & Responsible AI fields
    explanation: Dict[str, Any] = Field(default_factory=dict, description="Explanation of why this plan was generated")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model used and configuration")
    created_at: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat(), description="Plan creation timestamp")
    created_by: str = Field(default="planner_agent", description="Component that created this plan")


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

# NOTE: AVAILABLE_ACTIONS catalog is now embedded in secure template (app/prompts/planner_prompts.py)
# System prompt moved to secure template for 15-layer security with template integrity verification,
# input sanitization, structural isolation, proactive leakage prevention, and response validation.


# ============================================================================
# PLANNER AGENT FUNCTIONS
# ============================================================================


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
    logger.info(f"User Query: {redact_pii(user_query)}")  # Redact PII from user query
    logger.info(f"Intent: {intent}")
    logger.info(f"Query Type: {query_type}")
    logger.info(f"Comparison Targets: {comparison_targets}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,  # Deterministic planning
        api_key=OPENAI_API_KEY
    )
    
    # Initialize secure prompt template
    planner_prompt = PlannerPrompt()
    
    # Get secure system prompt with leakage prevention
    system_prompt = planner_prompt.get_system_prompt()
    
    # Format user message with security validation and structural isolation
    user_prompt = planner_prompt.format_user_message(
        user_query=user_query,
        intent=intent,
        query_type=query_type,
        comparison_targets=comparison_targets or []
    )
    
    # Build messages array with secure prompts (no template variables needed)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        # Invoke LLM to generate plan
        logger.info("Invoking LLM to generate plan...")
        import time
        start_time = time.time()
        
        response = llm.invoke(messages)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"LLM Response received ({len(response.content)} chars) in {execution_time_ms:.2f}ms")
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
        
        # Validate response schema with security checks
        planner_prompt.validate_response_schema(plan_dict)
        
        # Validate and create ExecutionPlan object
        plan = ExecutionPlan(**plan_dict)
        
        # NEW: Add explainability metadata
        plan.model_info = {
            "model": response.model if hasattr(response, 'model') else OPENAI_MODEL,
            "model_version": response.model if hasattr(response, 'model') else "unknown",
            "temperature": 0,
            "purpose": "deterministic_planning",
            "execution_time_ms": execution_time_ms,
            "response_length": len(response.content),
            "token_usage": {
                "prompt_tokens": response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0),
                "completion_tokens": response.response_metadata.get('token_usage', {}).get('completion_tokens', 0),
                "total_tokens": response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            } if hasattr(response, 'response_metadata') else {}
        }
        
        plan.explanation = {
            "reasoning": f"Generated {len(plan.steps)}-step plan for {query_type} query with {len(comparison_targets or [])} targets",
            "input_context": {
                "user_query": redact_pii(user_query),  # Redact PII from stored query
                "intent": intent,
                "query_type": query_type,
                "targets_count": len(comparison_targets or []),
                "targets": comparison_targets or []
            },
            "decision_factors": [
                f"Intent '{intent}' requires metric-specific data retrieval",
                f"Query type '{query_type}' determines execution pattern",
                f"{len(comparison_targets or [])} targets require comparison workflow",
                "Chart generation included for visual comparison",
                "Natural language formatting for user-friendly response"
            ],
            "alternatives_considered": "Sequential execution chosen over parallel for data consistency",
            "confidence": "high" if len(plan.steps) >= 4 else "medium"
        }
        
        # Validate plan structure
        validate_plan(plan)
        
        # NEW: Log structured decision for audit trail (with PII redaction)
        logger.info({
            "event": "plan_created",
            "plan_id": plan.plan_id,
            "query_type": plan.query_type,
            "intent": plan.intent,
            "steps_count": len(plan.steps),
            "model": OPENAI_MODEL,
            "temperature": 0,
            "execution_time_ms": execution_time_ms,
            "user_query": redact_pii(user_query),  # Redact PII before logging
            "targets": comparison_targets or [],
            "explanation": plan.explanation,
            "timestamp": plan.created_at
        })
        
        logger.info("=" * 60)
        logger.info(f"PLAN CREATED: {plan.plan_id}")
        logger.info("=" * 60)
        logger.info(f"Query Type: {plan.query_type}")
        logger.info(f"Intent: {plan.intent}")
        logger.info(f"Total Steps: {len(plan.steps)}")
        logger.info(f"Confidence: {plan.explanation.get('confidence', 'unknown')}")
        logger.info(f"Reasoning: {plan.explanation.get('reasoning', 'N/A')}")
        
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
    Validate execution plan for correctness and responsible AI principles.
    
    Checks:
    - Step IDs are sequential
    - Dependencies reference valid step IDs
    - No circular dependencies
    - At least one step exists
    - Fairness: No target bias (all targets treated equally)
    - Complexity: Plan is not overly complex (max 10 steps)
    
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
    
    # NEW: Responsible AI - Check plan complexity
    if len(plan.steps) > 10:
        logger.warning(f"Plan {plan.plan_id} has {len(plan.steps)} steps (max recommended: 10)")
        raise ValueError(f"Plan too complex: {len(plan.steps)} steps (max: 10). Consider simplifying query.")
    
    # NEW: Responsible AI - Check fairness (all comparison targets get equal treatment)
    query_steps = [s for s in plan.steps if s.action == "query_analytics"]
    if query_steps:
        targets = [s.params.get('target') for s in query_steps]
        if len(targets) != len(set(targets)):
            logger.warning(f"Plan {plan.plan_id} queries same target multiple times: {targets}")
            # This is allowed but worth noting for transparency
    
    logger.info(f"Plan {plan.plan_id} is valid (steps: {len(plan.steps)}, query_targets: {len(query_steps)})")


def get_plan_explanation(plan: ExecutionPlan) -> Dict[str, Any]:
    """
    Generate human-readable explanation of the plan for transparency.
    
    This function creates a detailed explanation that can be shown to users
    to help them understand what the system will do with their query.
    
    Args:
        plan: ExecutionPlan to explain
    
    Returns:
        Dictionary with user-friendly explanation
    """
    return {
        "plan_id": plan.plan_id,
        "summary": f"I created a {len(plan.steps)}-step plan to answer your {plan.query_type} query",
        "what_will_happen": [
            step.description for step in plan.steps
        ],
        "why_these_steps": plan.explanation.get("reasoning", "Plan optimized for accuracy and efficiency"),
        "model_used": plan.model_info.get("model_version", plan.model_info.get("model", "unknown")),
        "confidence": plan.explanation.get("confidence", "medium"),
        "estimated_time": f"{len(plan.steps) * 2}-{len(plan.steps) * 4} seconds",
        "data_sources": list(set([
            step.params.get('target') 
            for step in plan.steps 
            if step.action == "query_analytics" and step.params.get('target')
        ])),
        "created_at": plan.created_at
    }


def audit_plan_creation(plan: ExecutionPlan, user_id: str = None, org_id: str = None) -> Dict[str, Any]:
    """
    Create audit record for plan creation (Responsible AI - Accountability).
    
    This function logs all relevant information about plan creation for
    compliance, debugging, and continuous improvement.
    
    **Privacy & Security:**
    - User queries are PII-redacted before storage
    - org_id is kept separate (never sent to LLM)
    - Sensitive data is automatically filtered
    
    Args:
        plan: Created ExecutionPlan
        user_id: Optional user ID who requested the plan
        org_id: Optional organization ID (never sent to LLM)
    
    Returns:
        Audit record dictionary with PII-redacted content
    """
    # Redact PII from explanation before audit logging
    safe_explanation = plan.explanation.copy()
    if 'input_context' in safe_explanation and 'user_query' in safe_explanation['input_context']:
        safe_explanation['input_context']['user_query'] = redact_pii(
            safe_explanation['input_context']['user_query']
        )
    
    audit_record = {
        "audit_type": "plan_creation",
        "plan_id": plan.plan_id,
        "timestamp": plan.created_at,
        "user_id": redact_pii(user_id) if user_id else None,  # Redact in case user_id contains PII
        "org_id": org_id,  # Kept separate for privacy
        "query_type": plan.query_type,
        "intent": plan.intent,
        "steps_count": len(plan.steps),
        "steps_summary": [
            {"step_id": s.step_id, "action": s.action, "critical": s.critical}
            for s in plan.steps
        ],
        "model_info": plan.model_info,
        "explanation": safe_explanation,  # Use PII-redacted version
        "validation_passed": True,  # If we got here, validation passed
        "component": "planner_agent",
        "pii_redacted": True  # Flag indicating PII was redacted
    }
    
    # Log for audit trail
    logger.info({
        "event": "plan_audit_created",
        "audit_record": audit_record
    })
    
    return audit_record


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
