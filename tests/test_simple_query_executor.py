"""
Tests for simple_query_executor.py - Analytics orchestration with LangGraph.

This tests the Pattern B orchestrator that coordinates:
- Tool execution (LLM-based selection with deterministic fallback)
- Chart generation
- LLM response formatting
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from app.orchestration.simple_query_executor import (
    AnalyticsState,
    execute_analytics_tool,
    _deterministic_fallback,
    generate_chart_node,
    format_response_with_llm,
    build_analytics_orchestrator,
    run_analytics_query
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_state():
    """Sample analytics state for testing."""
    return {
        "user_query": "Show me success rate for customer domain",
        "extracted_data": {
            "report_type": "success_rate",
            "domain_name": "customer",
            "file_name": None
        },
        "org_id": "org-123",
        "tool_result": None,
        "chart_image": None,
        "final_response": None
    }


@pytest.fixture
def sample_tool_result():
    """Sample tool result data."""
    return {
        "success": True,
        "data": {
            "target_type": "domain",
            "target_value": "customer",
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "success_rate": 95.0,
            "failure_rate": 5.0,
            "_report_type": "success_rate"
        }
    }


# ============================================================================
# Test AnalyticsState
# ============================================================================

class TestAnalyticsState:
    """Tests for AnalyticsState TypedDict structure."""
    
    def test_analytics_state_structure(self, sample_state):
        """Test that state has all required fields."""
        assert "user_query" in sample_state
        assert "extracted_data" in sample_state
        assert "org_id" in sample_state
        assert "tool_result" in sample_state
        assert "chart_image" in sample_state
        assert "final_response" in sample_state


# ============================================================================
# Test execute_analytics_tool
# ============================================================================

class TestExecuteAnalyticsTool:
    """Tests for execute_analytics_tool with LLM selection."""
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_execute_analytics_tool_llm_success_rate(
        self, mock_chat, mock_get_tools, sample_state, sample_tool_result
    ):
        """Test LLM successfully selects success_rate tool."""
        # Mock tool
        mock_tool = Mock()
        mock_tool.name = "generate_success_rate_report"
        mock_tool.invoke.return_value = sample_tool_result
        mock_get_tools.return_value = [mock_tool]
        
        # Mock LLM response with tool call
        mock_response = Mock()
        mock_response.tool_calls = [{
            "name": "generate_success_rate_report",
            "args": {"domain_name": "customer"}
        }]
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        # Execute
        result = execute_analytics_tool(sample_state)
        
        # Verify
        assert "tool_result" in result
        assert result["tool_result"]["success"] == True
        assert result["tool_result"]["data"]["_report_type"] == "success_rate"
        
        # Verify org_id was passed to tool
        call_args = mock_tool.invoke.call_args[0][0]
        assert call_args["org_id"] == "org-123"
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_execute_analytics_tool_llm_failure_rate(
        self, mock_chat, mock_get_tools, sample_state
    ):
        """Test LLM successfully selects failure_rate tool."""
        sample_state["extracted_data"]["report_type"] = "failure_rate"
        sample_state["user_query"] = "Show me failure rate for customer"
        
        # Mock tool
        mock_tool = Mock()
        mock_tool.name = "generate_failure_rate_report"
        mock_tool.invoke.return_value = {
            "success": True,
            "data": {
                "failure_rate": 5.0,
                "_report_type": "failure_rate"
            }
        }
        mock_get_tools.return_value = [mock_tool]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.tool_calls = [{
            "name": "generate_failure_rate_report",
            "args": {"domain_name": "customer"}
        }]
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        result = execute_analytics_tool(sample_state)
        
        assert result["tool_result"]["success"] == True
        assert result["tool_result"]["data"]["_report_type"] == "failure_rate"
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_execute_analytics_tool_file_target(
        self, mock_chat, mock_get_tools, sample_state, sample_tool_result
    ):
        """Test tool execution with file_name target."""
        sample_state["extracted_data"]["domain_name"] = None
        sample_state["extracted_data"]["file_name"] = "customer.csv"
        
        mock_tool = Mock()
        mock_tool.name = "generate_success_rate_report"
        mock_tool.invoke.return_value = sample_tool_result
        mock_get_tools.return_value = [mock_tool]
        
        mock_response = Mock()
        mock_response.tool_calls = [{
            "name": "generate_success_rate_report",
            "args": {"file_name": "customer.csv"}
        }]
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        result = execute_analytics_tool(sample_state)
        
        assert result["tool_result"]["success"] == True
        call_args = mock_tool.invoke.call_args[0][0]
        assert "file_name" in call_args
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    @patch('app.orchestration.simple_query_executor._deterministic_fallback')
    def test_execute_analytics_tool_no_tool_calls(
        self, mock_fallback, mock_chat, mock_get_tools, sample_state
    ):
        """Test fallback when LLM doesn't call any tool."""
        mock_get_tools.return_value = []
        
        # Mock LLM response with NO tool calls
        mock_response = Mock()
        mock_response.tool_calls = []
        mock_response.content = "I'm not sure which tool to use"
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        mock_fallback.return_value = {"tool_result": {"success": False}}
        
        result = execute_analytics_tool(sample_state)
        
        # Should call fallback
        assert mock_fallback.called
        assert "tool_result" in result
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_execute_analytics_tool_tool_not_found(
        self, mock_chat, mock_get_tools, sample_state
    ):
        """Test when LLM calls a tool that doesn't exist."""
        mock_get_tools.return_value = []  # No tools available
        
        mock_response = Mock()
        mock_response.tool_calls = [{
            "name": "nonexistent_tool",
            "args": {}
        }]
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        result = execute_analytics_tool(sample_state)
        
        assert result["tool_result"]["success"] == False
        assert "not found" in result["tool_result"]["error"]
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    @patch('app.orchestration.simple_query_executor._deterministic_fallback')
    def test_execute_analytics_tool_llm_exception(
        self, mock_fallback, mock_chat, mock_get_tools, sample_state
    ):
        """Test fallback when LLM raises exception."""
        mock_get_tools.return_value = []
        
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.bind_tools.return_value.invoke.side_effect = Exception("LLM error")
        mock_chat.return_value = mock_llm
        
        mock_fallback.return_value = {"tool_result": {"success": False}}
        
        result = execute_analytics_tool(sample_state)
        
        # Should call fallback after exception
        assert mock_fallback.called


# ============================================================================
# Test _deterministic_fallback
# ============================================================================

class TestDeterministicFallback:
    """Tests for _deterministic_fallback function."""
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    def test_fallback_with_report_type_domain(self, mock_get_tools, sample_state, sample_tool_result):
        """Test fallback with explicit report_type and domain."""
        mock_tool = Mock()
        mock_tool.name = "generate_success_rate_report"
        mock_tool.invoke.return_value = sample_tool_result
        tools = [mock_tool]
        
        result = _deterministic_fallback(
            sample_state,
            tools,
            report_type="success_rate",
            domain_name="customer",
            file_name=None
        )
        
        assert result["tool_result"]["success"] == True
        assert mock_tool.invoke.called
    
    @patch('app.tools.analytics_tools.get_analytics_tools')
    def test_fallback_with_report_type_file(self, mock_get_tools, sample_state, sample_tool_result):
        """Test fallback with report_type and file_name."""
        mock_tool = Mock()
        mock_tool.name = "generate_failure_rate_report"
        mock_tool.invoke.return_value = sample_tool_result
        tools = [mock_tool]
        
        result = _deterministic_fallback(
            sample_state,
            tools,
            report_type="failure_rate",
            domain_name=None,
            file_name="customer.csv"
        )
        
        assert result["tool_result"]["success"] == True
        call_args = mock_tool.invoke.call_args[0][0]
        assert call_args["file_name"] == "customer.csv"
    
    def test_fallback_no_report_type_needs_clarification(self, sample_state):
        """Test fallback when no report_type specified."""
        tools = []
        
        result = _deterministic_fallback(
            sample_state,
            tools,
            report_type=None,
            domain_name="customer",
            file_name=None
        )
        
        assert result["tool_result"]["success"] == False
        assert "Would you like to see" in result["tool_result"]["error"]
    
    def test_fallback_no_valid_parameters(self, sample_state):
        """Test fallback with no valid parameters returns error."""
        tools = []
        
        result = _deterministic_fallback(
            sample_state,
            tools,
            report_type=None,
            domain_name=None,
            file_name=None
        )
        
        assert result["tool_result"]["success"] == False
        assert "error" in result["tool_result"]


# ============================================================================
# Test generate_chart_node
# ============================================================================

class TestGenerateChartNode:
    """Tests for generate_chart_node function."""
    
    @patch('app.services.chart_service.generate_analytics_chart')
    def test_generate_chart_success(self, mock_generate_chart, sample_state, sample_tool_result):
        """Test successful chart generation."""
        sample_state["tool_result"] = sample_tool_result
        mock_generate_chart.return_value = "base64_chart_image"
        
        result = generate_chart_node(sample_state)
        
        assert result["chart_image"] == "base64_chart_image"
        assert mock_generate_chart.called
    
    @patch('app.services.chart_service.generate_analytics_chart')
    def test_generate_chart_returns_none(self, mock_generate_chart, sample_state, sample_tool_result):
        """Test chart generation returns None gracefully."""
        sample_state["tool_result"] = sample_tool_result
        mock_generate_chart.return_value = None
        
        result = generate_chart_node(sample_state)
        
        assert result["chart_image"] is None
    
    @patch('app.services.chart_service.generate_analytics_chart')
    def test_generate_chart_exception(self, mock_generate_chart, sample_state, sample_tool_result):
        """Test chart generation handles exceptions."""
        sample_state["tool_result"] = sample_tool_result
        mock_generate_chart.side_effect = Exception("Chart error")
        
        result = generate_chart_node(sample_state)
        
        assert result["chart_image"] is None
    
    def test_generate_chart_tool_failed(self, sample_state):
        """Test chart generation skips when tool failed."""
        sample_state["tool_result"] = {"success": False, "error": "Tool error"}
        
        result = generate_chart_node(sample_state)
        
        assert result["chart_image"] is None


# ============================================================================
# Test format_response_with_llm
# ============================================================================

class TestFormatResponseWithLLM:
    """Tests for format_response_with_llm function."""
    
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_format_response_success_with_chart(
        self, mock_chat, sample_state, sample_tool_result
    ):
        """Test LLM response formatting with chart."""
        sample_state["tool_result"] = sample_tool_result
        sample_state["chart_image"] = "base64_chart_image"
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "The customer domain has a 95% success rate."
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        result = format_response_with_llm(sample_state)
        
        assert result["final_response"]["success"] == True
        assert "95% success rate" in result["final_response"]["message"]
        assert result["final_response"]["chart_image"] == "base64_chart_image"
    
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_format_response_success_without_chart(
        self, mock_chat, sample_state, sample_tool_result
    ):
        """Test LLM response formatting without chart."""
        sample_state["tool_result"] = sample_tool_result
        sample_state["chart_image"] = None
        
        mock_response = Mock()
        mock_response.content = "The analysis is complete."
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        result = format_response_with_llm(sample_state)
        
        assert result["final_response"]["success"] == True
        assert result["final_response"]["chart_image"] is None
    
    def test_format_response_tool_failure(self, sample_state):
        """Test formatting when tool execution failed."""
        sample_state["tool_result"] = {
            "success": False,
            "error": "Database connection failed"
        }
        
        result = format_response_with_llm(sample_state)
        
        assert result["final_response"]["success"] == False
        assert "Database connection failed" in result["final_response"]["message"]
    
    def test_format_response_needs_clarification(self, sample_state):
        """Test formatting when clarification needed (treated as error)."""
        sample_state["tool_result"] = {
            "success": False,
            "needs_clarification": True,
            "message": "Would you like to see success rate or failure rate?"
        }
        
        result = format_response_with_llm(sample_state)
        
        # Function treats this as error, extracts message field
        assert result["final_response"]["success"] == False
        assert "I encountered an error" in result["final_response"]["message"]
    
    @patch('app.orchestration.simple_query_executor.ChatOpenAI')
    def test_format_response_llm_exception(
        self, mock_chat, sample_state, sample_tool_result
    ):
        """Test that LLM exception propagates (no error handling in function)."""
        sample_state["tool_result"] = sample_tool_result
        sample_state["chart_image"] = None
        
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_chat.return_value = mock_llm
        
        # Function doesn't handle exceptions, so it should raise
        with pytest.raises(Exception, match="LLM error"):
            format_response_with_llm(sample_state)


# ============================================================================
# Test build_analytics_orchestrator
# ============================================================================

class TestBuildAnalyticsOrchestrator:
    """Tests for build_analytics_orchestrator function."""
    
    def test_build_orchestrator_returns_compiled_graph(self):
        """Test that build_analytics_orchestrator returns compiled graph."""
        graph = build_analytics_orchestrator()
        
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    def test_build_orchestrator_has_invoke_method(self):
        """Test compiled graph has invoke capability."""
        graph = build_analytics_orchestrator()
        
        assert callable(getattr(graph, 'ainvoke', None))


# ============================================================================
# Test run_analytics_query
# ============================================================================

class TestRunAnalyticsQuery:
    """Tests for run_analytics_query main entry point."""
    
    @pytest.mark.asyncio
    @patch('app.orchestration.simple_query_executor.build_analytics_orchestrator')
    async def test_run_analytics_query_success(self, mock_build_graph, sample_tool_result):
        """Test successful analytics query execution."""
        # Mock compiled graph
        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke.return_value = {
            "user_query": "Show me success rate",
            "extracted_data": {"report_type": "success_rate", "domain_name": "customer", "file_name": None},
            "org_id": "org-123",
            "tool_result": sample_tool_result,
            "chart_image": "base64_image",
            "final_response": {
                "success": True,
                "message": "Success rate is 95%",
                "chart_image": "base64_image"
            }
        }
        
        mock_build_graph.return_value = mock_compiled_graph
        
        result = await run_analytics_query(
            user_query="Show me success rate",
            extracted_data={"report_type": "success_rate", "domain_name": "customer", "file_name": None},
            org_id="org-123"
        )
        
        assert result["success"] == True
        assert "95%" in result["message"]
    
    @pytest.mark.asyncio
    @patch('app.orchestration.simple_query_executor.build_analytics_orchestrator')
    async def test_run_analytics_query_without_org_id(self, mock_build_graph, sample_tool_result):
        """Test query execution without org_id."""
        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke.return_value = {
            "user_query": "Show me success rate",
            "extracted_data": {"report_type": "success_rate", "domain_name": "customer", "file_name": None},
            "org_id": None,
            "tool_result": sample_tool_result,
            "chart_image": None,
            "final_response": {
                "success": True,
                "message": "Success rate is 95%",
                "chart_image": None
            }
        }
        
        mock_build_graph.return_value = mock_compiled_graph
        
        result = await run_analytics_query(
            user_query="Show me success rate",
            extracted_data={"report_type": "success_rate", "domain_name": "customer", "file_name": None}
        )
        
        assert result["success"] == True
    
    @pytest.mark.asyncio
    @patch('app.orchestration.simple_query_executor.build_analytics_orchestrator')
    async def test_run_analytics_query_exception(self, mock_build_graph):
        """Test query execution doesn't handle exceptions (propagates)."""
        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke.side_effect = Exception("Graph execution error")
        
        mock_build_graph.return_value = mock_compiled_graph
        
        # Function doesn't handle exceptions, so it should raise
        with pytest.raises(Exception, match="Graph execution error"):
            await run_analytics_query(
                user_query="Show me success rate",
                extracted_data={"report_type": "success_rate", "domain_name": "customer", "file_name": None},
                org_id="org-123"
            )
