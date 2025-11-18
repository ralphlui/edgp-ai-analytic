"""
Evaluation configuration for LangSmith integration.
Simple approach: evaluate all plans and log to LangSmith.
"""


class EvaluationConfig:
    """Configuration for LangSmith evaluation."""
    
    # Enable/disable evaluation
    ENABLE_EVALUATION = True
    
    # Quality threshold (0.7 = 70%)
    QUALITY_THRESHOLD = 0.7
    
    # LangSmith project name
    LANGSMITH_PROJECT = "edgp-ai-analytic"


# Singleton config instance
evaluation_config = EvaluationConfig()
