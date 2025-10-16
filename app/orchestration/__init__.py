"""
Orchestration Module

This module contains all query orchestration components that coordinate
the end-to-end query processing workflow.

Components:
- query_understanding_agent: Step 1 - Understands user intent and extracts entities
- planner_agent: Step 2 - Creates execution plans for complex comparison queries
- complex_query_executor: Step 3a - Executes multi-step comparison plans
- simple_query_executor: Step 3b - Executes simple analytics queries

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Query Processor                          │
│                   (Main Entry Point)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              query_understanding_agent                      │
│           (Classify intent & extract targets)               │
└─────────────────────────────────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │                 │
                   ▼                 ▼
         ┌──────────────┐    ┌──────────────┐
         │Simple Query  │    │Complex Query │
         └──────────────┘    └──────────────┘
                   │                 │
                   │                 ▼
                   │         ┌──────────────┐
                   │         │planner_agent │
                   │         │(Create Plan) │
                   │         └──────────────┘
                   │                 │
                   ▼                 ▼
         ┌──────────────┐    ┌──────────────┐
         │simple_query_ │    │complex_query_│
         │executor      │    │executor      │
         └──────────────┘    └──────────────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                  ┌──────────────────┐
                  │  Final Response  │
                  └──────────────────┘
"""

from app.orchestration.query_understanding_agent import (
    get_query_understanding_agent,
    QueryUnderstandingAgent,
    QueryUnderstandingResult
)

from app.orchestration.planner_agent import (
    create_execution_plan,
    ExecutionPlan,
    PlanStep
)

from app.orchestration.complex_query_executor import (
    execute_plan,
    ExecutionState
)

from app.orchestration.simple_query_executor import (
    run_analytics_query,
    AnalyticsState
)

__all__ = [
    # Query Understanding
    "get_query_understanding_agent",
    "QueryUnderstandingAgent",
    "QueryUnderstandingResult",
    
    # Planning
    "create_execution_plan",
    "ExecutionPlan",
    "PlanStep",
    
    # Execution
    "execute_plan",
    "ExecutionState",
    "run_analytics_query",
    "AnalyticsState",
]
