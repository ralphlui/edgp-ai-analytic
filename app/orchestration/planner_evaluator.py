"""
Simple planner wrapper with LangSmith evaluation.
Evaluates every plan and logs to LangSmith dashboard.
"""

import logging
from typing import List, Optional
from langsmith import traceable

from app.orchestration.planner_agent import (
    create_execution_plan as create_execution_plan_internal,
    ExecutionPlan
)
from app.services.evaluation_service import evaluate_plan
from config.evaluation_config import EvaluationConfig

# Initialize config
evaluation_config = EvaluationConfig()

logger = logging.getLogger("planner_evaluator")


@traceable(
    name="create_execution_plan_with_evaluation",
    project_name=evaluation_config.LANGSMITH_PROJECT,
    tags=["planner", "evaluation"]
)
def create_execution_plan_with_evaluation(
    intent: str,
    comparison_targets: Optional[List[str]] = None,
    user_query: str = "",
    query_type: str = "comparison"
) -> ExecutionPlan:
    """
    Create execution plan and evaluate with LangSmith.
    
    This wrapper:
    1. Calls create_execution_plan() (unchanged)
    2. Evaluates the plan
    3. Logs results to LangSmith
    4. Returns plan with evaluation metadata
    
    Args:
        intent: User's intent
        comparison_targets: List of targets to compare
        user_query: Original user query
        query_type: Type of query
    
    Returns:
        ExecutionPlan with evaluation results in metadata
    """
    
    logger.info("="*60)
    logger.info("PLANNER WITH LANGSMITH EVALUATION")
    logger.info("="*60)
    
    # 1. Create plan using existing function (NO CHANGES)
    plan = create_execution_plan_internal(
        intent=intent,
        comparison_targets=comparison_targets,
        user_query=user_query,
        query_type=query_type
    )
    
    # 2. Evaluate plan and log to LangSmith
    evaluation_result = evaluate_plan(
        plan=plan,
        user_query=user_query
    )
    
    # 3. Add evaluation metadata to plan
    if evaluation_result:
        plan.metadata["evaluation"] = evaluation_result
        logger.info(
            f"Plan evaluated: {plan.plan_id}, "
            f"score: {evaluation_result['overall_score']:.2f}, "
            f"status: {'✅ passed' if evaluation_result['passed'] else '❌ failed'}"
        )
    
    logger.info("="*60)
    
    return plan
