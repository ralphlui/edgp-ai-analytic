"""
Tests for Planner Agent - Execution plan creation for complex queries.

This module tests the plan creation logic, validation, and edge cases.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from app.orchestration.planner_agent import (
    PlanStep,
    ExecutionPlan,
    create_execution_plan,
    validate_plan,
    create_comparison_plan,
    AVAILABLE_ACTIONS
)
# Note: PLANNER_SYSTEM_PROMPT moved to secure template (app/prompts/planner_prompts.py)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_plan_step():
    """Sample plan step for testing."""
    return PlanStep(
        step_id=1,
        action="query_analytics",
        description="Query success rate for customer.csv",
        params={"target": "customer.csv", "metric_type": "success_rate"},
        depends_on=[],
        critical=True
    )


@pytest.fixture
def sample_execution_plan():
    """Sample execution plan for testing."""
    return ExecutionPlan(
        plan_id="plan-test123",
        query_type="comparison",
        intent="success_rate",
        steps=[
            PlanStep(
                step_id=1,
                action="query_analytics",
                description="Query success rate for customer.csv",
                params={"target": "customer.csv", "metric_type": "success_rate"},
                depends_on=[],
                critical=True
            ),
            PlanStep(
                step_id=2,
                action="query_analytics",
                description="Query success rate for payment.csv",
                params={"target": "payment.csv", "metric_type": "success_rate"},
                depends_on=[],
                critical=True
            ),
            PlanStep(
                step_id=3,
                action="compare_results",
                description="Compare success rates",
                params={"compare_steps": [1, 2], "metric": "success_rate"},
                depends_on=[1, 2],
                critical=True
            )
        ],
        metadata={"targets_count": 2}
    )


@pytest.fixture
def valid_llm_response():
    """Valid LLM response JSON for 2-target comparison."""
    return {
        "plan_id": "plan-abc123",
        "query_type": "comparison",
        "intent": "success_rate",
        "steps": [
            {
                "step_id": 1,
                "action": "query_analytics",
                "description": "Query success rate for customer.csv",
                "params": {"target": "customer.csv", "metric_type": "success_rate"},
                "depends_on": [],
                "critical": True
            },
            {
                "step_id": 2,
                "action": "query_analytics",
                "description": "Query success rate for payment.csv",
                "params": {"target": "payment.csv", "metric_type": "success_rate"},
                "depends_on": [],
                "critical": True
            },
            {
                "step_id": 3,
                "action": "compare_results",
                "description": "Compare success rates",
                "params": {"compare_steps": [1, 2], "metric": "success_rate"},
                "depends_on": [1, 2],
                "critical": True
            },
            {
                "step_id": 4,
                "action": "generate_chart",
                "description": "Create comparison bar chart",
                "params": {"comparison_step_id": 3},
                "depends_on": [3],
                "critical": False
            },
            {
                "step_id": 5,
                "action": "format_response",
                "description": "Generate natural language summary",
                "params": {"comparison_step_id": 3, "chart_step_id": 4},
                "depends_on": [3],
                "critical": False
            }
        ],
        "metadata": {
            "estimated_duration": "2-3 seconds",
            "complexity": "medium",
            "targets_count": 2
        }
    }


def mock_llm_chain(json_content):
    """Helper to create mock LLM chain that returns given JSON content."""
    mock_response = MagicMock()
    mock_response.content = json_content
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response
    
    return mock_chain


# ============================================================================
# TEST PYDANTIC MODELS
# ============================================================================

class TestPlanStepModel:
    """Test PlanStep Pydantic model."""
    
    def test_plan_step_creation_success(self):
        """Test creating valid PlanStep."""
        step = PlanStep(
            step_id=1,
            action="query_analytics",
            description="Test step",
            params={"target": "test.csv"},
            depends_on=[],
            critical=True
        )
        
        assert step.step_id == 1
        assert step.action == "query_analytics"
        assert step.description == "Test step"
        assert step.params == {"target": "test.csv"}
        assert step.depends_on == []
        assert step.critical is True
    
    def test_plan_step_defaults(self):
        """Test PlanStep with default values."""
        step = PlanStep(
            step_id=1,
            action="query_analytics",
            description="Test step"
        )
        
        assert step.params == {}
        assert step.depends_on == []
        assert step.critical is True
    
    def test_plan_step_missing_required_fields(self):
        """Test PlanStep fails without required fields."""
        with pytest.raises(ValidationError):
            PlanStep(step_id=1, action="query_analytics")  # Missing description


class TestExecutionPlanModel:
    """Test ExecutionPlan Pydantic model."""
    
    def test_execution_plan_creation_success(self, sample_execution_plan):
        """Test creating valid ExecutionPlan."""
        plan = sample_execution_plan
        
        assert plan.query_type == "comparison"
        assert plan.intent == "success_rate"
        assert len(plan.steps) == 3
        assert plan.metadata == {"targets_count": 2}
    
    def test_execution_plan_auto_generates_plan_id(self):
        """Test ExecutionPlan auto-generates plan_id if not provided."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test")
            ]
        )
        
        assert plan.plan_id.startswith("plan-")
        assert len(plan.plan_id) > 5
    
    def test_execution_plan_empty_steps(self):
        """Test ExecutionPlan with no steps (should be valid at model level)."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[]
        )
        
        assert len(plan.steps) == 0  # Valid at model level, but validate_plan will reject


# ============================================================================
# TEST PLAN VALIDATION
# ============================================================================

class TestValidatePlan:
    """Test plan validation logic."""
    
    def test_validate_plan_success(self, sample_execution_plan):
        """Test validating a valid plan succeeds."""
        # Should not raise any exception
        validate_plan(sample_execution_plan)
    
    def test_validate_plan_no_steps(self):
        """Test validation fails for plan with no steps."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[]
        )
        
        with pytest.raises(ValueError, match="Plan has no steps"):
            validate_plan(plan)
    
    def test_validate_plan_non_sequential_step_ids(self):
        """Test validation fails for non-sequential step IDs."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test 1"),
                PlanStep(step_id=3, action="test", description="Test 3"),  # Skips 2
            ]
        )
        
        with pytest.raises(ValueError, match="Step IDs must be sequential"):
            validate_plan(plan)
    
    def test_validate_plan_step_ids_not_starting_from_1(self):
        """Test validation fails if step IDs don't start from 1."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=2, action="test", description="Test 2"),
                PlanStep(step_id=3, action="test", description="Test 3"),
            ]
        )
        
        with pytest.raises(ValueError, match="Step IDs must be sequential"):
            validate_plan(plan)
    
    def test_validate_plan_invalid_dependency(self):
        """Test validation fails for dependency on non-existent step."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test 1"),
                PlanStep(step_id=2, action="test", description="Test 2", depends_on=[99]),  # Invalid
            ]
        )
        
        with pytest.raises(ValueError, match="depends on non-existent step"):
            validate_plan(plan)
    
    def test_validate_plan_forward_dependency(self):
        """Test validation fails for forward dependency (step depends on later step)."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test 1", depends_on=[2]),  # Forward dep
                PlanStep(step_id=2, action="test", description="Test 2"),
            ]
        )
        
        with pytest.raises(ValueError, match="forward dependency"):
            validate_plan(plan)
    
    def test_validate_plan_self_dependency(self):
        """Test validation fails for circular dependency (step depends on itself)."""
        plan = ExecutionPlan(
            query_type="comparison",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test 1", depends_on=[1]),  # Self-dep
            ]
        )
        
        # Self-dependency is caught by forward dependency check first
        with pytest.raises(ValueError, match="forward dependency|circular dependency"):
            validate_plan(plan)


# ============================================================================
# TEST CREATE EXECUTION PLAN
# ============================================================================

class TestCreateExecutionPlan:
    """Test execution plan creation using LLM."""
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_success(self, mock_chat_openai, valid_llm_response):
        """Test successful plan creation with valid LLM response."""
        # Mock LLM response - use string directly so len() works
        json_content = json.dumps(valid_llm_response)
        mock_response = MagicMock()
        mock_response.content = json_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Create plan
        plan = create_execution_plan(
            intent="success_rate",
            comparison_targets=["customer.csv", "payment.csv"],
            user_query="Compare success rates",
            query_type="comparison"
        )
        
        # Verify plan structure
        assert plan.query_type == "comparison"
        assert plan.intent == "success_rate"
        assert len(plan.steps) == 5
        assert plan.steps[0].action == "query_analytics"
        assert plan.steps[2].action == "compare_results"
        assert plan.steps[3].action == "generate_chart"
        assert plan.steps[4].action == "format_response"
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_with_markdown_wrapped_json(self, mock_chat_openai, valid_llm_response):
        """Test plan creation when LLM wraps JSON in markdown code blocks."""
        # Mock LLM response with markdown wrapper
        json_content = f"```json\n{json.dumps(valid_llm_response)}\n```"
        mock_response = MagicMock()
        mock_response.content = json_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Create plan
        plan = create_execution_plan(
            intent="success_rate",
            comparison_targets=["customer.csv", "payment.csv"],
            user_query="Compare success rates",
            query_type="comparison"
        )
        
        # Should successfully parse despite markdown wrapper
        assert plan.query_type == "comparison"
        assert len(plan.steps) == 5
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_with_three_targets(self, mock_chat_openai):
        """Test plan creation for 3-target comparison."""
        # Mock LLM response for 3 targets
        three_target_response = {
            "plan_id": "plan-xyz789",
            "query_type": "comparison",
            "intent": "failure_rate",
            "steps": [
                {
                    "step_id": 1,
                    "action": "query_analytics",
                    "description": "Query failure rate for customer.csv",
                    "params": {"target": "customer.csv", "metric_type": "failure_rate"},
                    "depends_on": [],
                    "critical": True
                },
                {
                    "step_id": 2,
                    "action": "query_analytics",
                    "description": "Query failure rate for payment.csv",
                    "params": {"target": "payment.csv", "metric_type": "failure_rate"},
                    "depends_on": [],
                    "critical": True
                },
                {
                    "step_id": 3,
                    "action": "query_analytics",
                    "description": "Query failure rate for transactions.csv",
                    "params": {"target": "transactions.csv", "metric_type": "failure_rate"},
                    "depends_on": [],
                    "critical": True
                },
                {
                    "step_id": 4,
                    "action": "compare_results",
                    "description": "Compare failure rates",
                    "params": {"compare_steps": [1, 2, 3], "metric": "failure_rate"},
                    "depends_on": [1, 2, 3],
                    "critical": True
                },
                {
                    "step_id": 5,
                    "action": "generate_chart",
                    "description": "Create comparison chart",
                    "params": {"comparison_step_id": 4},
                    "depends_on": [4],
                    "critical": False
                },
                {
                    "step_id": 6,
                    "action": "format_response",
                    "description": "Generate summary",
                    "params": {"comparison_step_id": 4, "chart_step_id": 5},
                    "depends_on": [4],
                    "critical": False
                }
            ],
            "metadata": {"targets_count": 3}
        }
        
        json_content = json.dumps(three_target_response)
        mock_response = MagicMock()
        mock_response.content = json_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Create plan
        plan = create_execution_plan(
            intent="failure_rate",
            comparison_targets=["customer.csv", "payment.csv", "transactions.csv"],
            user_query="Compare failure rates",
            query_type="comparison"
        )
        
        # Verify 3-target plan
        assert len(plan.steps) == 6
        assert plan.steps[0].params["target"] == "customer.csv"
        assert plan.steps[1].params["target"] == "payment.csv"
        assert plan.steps[2].params["target"] == "transactions.csv"
        assert plan.steps[3].params["compare_steps"] == [1, 2, 3]
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_invalid_json(self, mock_chat_openai):
        """Test plan creation fails with invalid JSON from LLM."""
        # Mock invalid LLM response
        invalid_content = "This is not valid JSON at all!"
        mock_response = MagicMock()
        mock_response.content = invalid_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="did not return valid JSON"):
            create_execution_plan(
                intent="success_rate",
                comparison_targets=["customer.csv", "payment.csv"],
                user_query="Compare success rates",
                query_type="comparison"
            )
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_invalid_plan_structure(self, mock_chat_openai):
        """Test plan creation fails with invalid plan (triggers validation error)."""
        # Mock LLM response with invalid plan (non-sequential step IDs)
        invalid_plan = {
            "plan_id": "plan-bad",
            "query_type": "comparison",
            "intent": "success_rate",
            "steps": [
                {
                    "step_id": 1,
                    "action": "query_analytics",
                    "description": "Test",
                    "params": {},
                    "depends_on": [],
                    "critical": True
                },
                {
                    "step_id": 99,  # Non-sequential!
                    "action": "compare_results",
                    "description": "Test",
                    "params": {},
                    "depends_on": [1],
                    "critical": True
                }
            ],
            "metadata": {}
        }
        
        json_content = json.dumps(invalid_plan)
        mock_response = MagicMock()
        mock_response.content = json_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Should raise ValueError from validate_plan
        with pytest.raises(ValueError, match="Step IDs must be sequential"):
            create_execution_plan(
                intent="success_rate",
                comparison_targets=["customer.csv", "payment.csv"],
                user_query="Compare success rates",
                query_type="comparison"
            )
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_llm_exception(self, mock_chat_openai):
        """Test plan creation handles LLM exceptions."""
        # Mock LLM raising exception
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("LLM API error")
        
        mock_llm = Mock()
        mock_llm.__or__ = Mock(return_value=mock_chain)
        mock_chat_openai.return_value = mock_llm
        
        # Should re-raise exception (any exception is fine since chain.invoke fails early)
        with pytest.raises(Exception):
            create_execution_plan(
                intent="success_rate",
                comparison_targets=["customer.csv", "payment.csv"],
                user_query="Compare success rates",
                query_type="comparison"
            )


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestCreateComparisonPlan:
    """Test convenience function for comparison plans."""
    
    @patch('app.orchestration.planner_agent.create_execution_plan')
    def test_create_comparison_plan_success(self, mock_create_plan):
        """Test create_comparison_plan calls create_execution_plan correctly."""
        mock_plan = Mock()
        mock_create_plan.return_value = mock_plan
        
        result = create_comparison_plan(
            comparison_targets=["customer.csv", "payment.csv"],
            intent="success_rate",
            user_query="Compare success rates"
        )
        
        # Verify create_execution_plan was called with correct args
        mock_create_plan.assert_called_once_with(
            intent="success_rate",
            comparison_targets=["customer.csv", "payment.csv"],
            user_query="Compare success rates",
            query_type="comparison"
        )
        
        assert result == mock_plan
    
    @patch('app.orchestration.planner_agent.create_execution_plan')
    def test_create_comparison_plan_defaults(self, mock_create_plan):
        """Test create_comparison_plan with default parameters."""
        mock_plan = Mock()
        mock_create_plan.return_value = mock_plan
        
        result = create_comparison_plan(
            comparison_targets=["test.csv"]
        )
        
        # Verify defaults
        mock_create_plan.assert_called_once_with(
            intent="success_rate",  # Default
            comparison_targets=["test.csv"],
            user_query="",  # Default
            query_type="comparison"
        )


# ============================================================================
# TEST CONSTANTS
# ============================================================================

class TestConstants:
    """Test module constants are defined."""
    
    def test_available_actions_defined(self):
        """Test AVAILABLE_ACTIONS constant is defined."""
        assert isinstance(AVAILABLE_ACTIONS, str)
        assert "query_analytics" in AVAILABLE_ACTIONS
        assert "compare_results" in AVAILABLE_ACTIONS
        assert "generate_chart" in AVAILABLE_ACTIONS
        assert "format_response" in AVAILABLE_ACTIONS
    
    def test_planner_system_prompt_defined(self):
        """Test PLANNER_SYSTEM_PROMPT moved to secure template."""
        # System prompt now in secure template: app/prompts/planner_prompts.py
        from app.prompts.planner_prompts import PlannerPrompt
        
        planner_prompt = PlannerPrompt()
        system_prompt = planner_prompt.get_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert "query planner" in system_prompt.lower()
        assert "query_analytics" in system_prompt


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_plan_step_with_empty_params(self):
        """Test PlanStep with empty params dict."""
        step = PlanStep(
            step_id=1,
            action="test",
            description="Test",
            params={}
        )
        
        assert step.params == {}
    
    def test_plan_step_with_complex_params(self):
        """Test PlanStep with nested complex params."""
        step = PlanStep(
            step_id=1,
            action="test",
            description="Test",
            params={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "mixed": {"a": [1, 2], "b": {"c": "d"}}
            }
        )
        
        assert step.params["nested"]["key"] == "value"
        assert step.params["list"] == [1, 2, 3]
    
    def test_execution_plan_with_single_step(self):
        """Test ExecutionPlan with only one step."""
        plan = ExecutionPlan(
            query_type="simple",
            intent="success_rate",
            steps=[
                PlanStep(step_id=1, action="test", description="Test")
            ]
        )
        
        # Should pass validation
        validate_plan(plan)
        assert len(plan.steps) == 1
    
    @patch('app.orchestration.planner_agent.ChatOpenAI')
    def test_create_execution_plan_with_no_targets(self, mock_chat_openai, valid_llm_response):
        """Test plan creation with None comparison_targets."""
        json_content = json.dumps(valid_llm_response)
        mock_response = MagicMock()
        mock_response.content = json_content
        
        # Mock LLM to return response directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Should handle None targets gracefully
        plan = create_execution_plan(
            intent="success_rate",
            comparison_targets=None,
            user_query="General query",
            query_type="aggregation"
        )
        
        assert plan is not None
        assert len(plan.steps) > 0
