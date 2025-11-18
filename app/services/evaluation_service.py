"""
Simple LangSmith evaluation service for planner quality monitoring.
"""

import logging
import time
from typing import Dict, Any, Optional
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree
from config.evaluation_config import EvaluationConfig
from app.orchestration.planner_agent import ExecutionPlan, validate_plan

# Initialize config
evaluation_config = EvaluationConfig()

logger = logging.getLogger("evaluation_service")

# Try to import LangSmith
try:
    from langsmith import Client
    client = Client()
    LANGSMITH_AVAILABLE = True
    logger.info("✅ LangSmith connected successfully")
except ImportError:
    logger.error("❌ LangSmith not installed. Install with: pip install langsmith")
    LANGSMITH_AVAILABLE = False
    client = None
except Exception as e:
    logger.error(f"❌ LangSmith connection failed: {e}")
    logger.error("Check LANGCHAIN_API_KEY environment variable")
    LANGSMITH_AVAILABLE = False
    client = None


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_plan(
    plan: ExecutionPlan,
    user_query: str
) -> Dict[str, Any]:
    """
    Evaluate plan quality and log to LangSmith.
    
    Checks:
    - Correctness (plan validation)
    - Complexity (step count)
    - Confidence (from plan explanation)
    - Explainability (has reasoning)
    - Model tracking (has version info)
    
    Args:
        plan: Execution plan to evaluate
        user_query: Original user query
    
    Returns:
        Evaluation result with overall score and breakdown
    """
    
    if not evaluation_config.ENABLE_EVALUATION:
        logger.debug(f"Evaluation disabled, skipping: {plan.plan_id}")
        return None
    
    start_time = time.time()
    
    try:
        logger.info(f"Evaluating plan: {plan.plan_id}")
        
        scores = {}
        
        # 1. Correctness check - does plan validate?
        try:
            validate_plan(plan)
            scores["correctness"] = 1.0
            logger.debug(f"✅ Correctness: plan validation passed")
        except Exception as e:
            scores["correctness"] = 0.0
            logger.warning(f"❌ Correctness: plan validation failed - {e}")
        
        # 2. Complexity check - reasonable number of steps?
        step_count = len(plan.steps)
        if step_count <= 7:
            scores["complexity"] = 1.0
            logger.debug(f"✅ Complexity: {step_count} steps (reasonable)")
        elif step_count <= 10:
            scores["complexity"] = 0.75
            logger.debug(f"⚠️  Complexity: {step_count} steps (high)")
        else:
            scores["complexity"] = 0.5
            logger.warning(f"⚠️  Complexity: {step_count} steps (very high)")
        
        # 3. Confidence check - does plan have confidence rating?
        confidence_map = {
            "high": 1.0,
            "medium": 0.75,
            "low": 0.5,
            "unknown": 0.0
        }
        confidence = plan.explanation.get("confidence", "unknown")
        scores["confidence"] = confidence_map.get(confidence, 0.0)
        logger.debug(f"Confidence: {confidence} → {scores['confidence']}")
        
        # 4. Explainability check - has reasoning?
        has_explanation = bool(plan.explanation and plan.explanation.get("reasoning"))
        scores["explainability"] = 1.0 if has_explanation else 0.0
        logger.debug(f"{'✅' if has_explanation else '❌'} Explainability: has reasoning")
        
        # 5. Model tracking check - has version info?
        has_model_info = bool(plan.model_info and plan.model_info.get("model_version"))
        scores["model_tracking"] = 1.0 if has_model_info else 0.5
        logger.debug(f"{'✅' if has_model_info else '⚠️'} Model tracking: has version")
        
        # Calculate overall score (average)
        overall_score = sum(scores.values()) / len(scores)
        passed = overall_score >= evaluation_config.QUALITY_THRESHOLD
        
        evaluation_time_ms = (time.time() - start_time) * 1000
        
        result = {
            "plan_id": plan.plan_id,
            "overall_score": overall_score,
            "scores": scores,
            "passed": passed,
            "evaluation_time_ms": evaluation_time_ms,
            "step_count": step_count,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        # Log to LangSmith
        # Get the current trace run_id from LangSmith context
        if LANGSMITH_AVAILABLE and client:
            try:
                # Get current run tree to extract run_id
                run_tree = get_current_run_tree()
                run_id = run_tree.id if run_tree else None
                
                if run_id:
                    # Create feedback linked to the trace with full explainability data
                    feedback = client.create_feedback(
                        run_id=run_id,
                        key="plan_quality_evaluation",
                        score=overall_score,
                        comment=f"Plan: {plan.plan_id} | Quality: {'✅ passed' if passed else '❌ failed'} | Confidence: {confidence} | Steps: {step_count}",
                        feedback_source_type="model",
                        value={
                            # Evaluation scores
                            "plan_id": plan.plan_id,
                            "scores": scores,
                            "threshold": evaluation_config.QUALITY_THRESHOLD,
                            "evaluation_time_ms": evaluation_time_ms,
                            "step_count": step_count,
                            "confidence": confidence,
                            
                            # EXPLAINABILITY: Why this plan was created
                            "explanation": plan.explanation,
                            
                            # RESPONSIBLE AI: Model tracking & audit trail
                            "model_info": plan.model_info,
                            "created_at": plan.created_at,
                            "created_by": plan.created_by,
                            
                            # Additional metadata for compliance
                            "query_type": plan.query_type,
                            "intent": plan.intent,
                            "metadata": plan.metadata
                        }
                    )
                    logger.info(f"📊 Logged to LangSmith (run {run_id}): {plan.plan_id} → {overall_score:.2f}")
                else:
                    logger.warning("No active LangSmith trace found - feedback not logged")
            except Exception as e:
                logger.error(f"Failed to log to LangSmith: {e}")
        else:
            logger.warning(f"LangSmith not available, evaluation logged locally only")
        
        # Log summary
        status_icon = "✅" if passed else "❌"
        logger.info(
            f"{status_icon} Evaluation complete: {plan.plan_id} | "
            f"Score: {overall_score:.2f} | "
            f"Correctness: {scores['correctness']:.1f} | "
            f"Complexity: {scores['complexity']:.2f} | "
            f"Confidence: {scores['confidence']:.2f} | "
            f"Time: {evaluation_time_ms:.0f}ms"
        )
        
        return result
    
    except Exception as e:
        logger.exception(f"Evaluation failed for plan {plan.plan_id}: {e}")
        return {
            "plan_id": plan.plan_id,
            "overall_score": 0.0,
            "passed": False,
            "error": str(e)
        }
