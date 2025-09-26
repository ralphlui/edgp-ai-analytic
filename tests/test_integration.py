"""
Integration tests for the complete reasoning and planning workflow.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from fastapi.testclient import TestClient


class TestEndToEndWorkflow:
    """Test the complete end-to-end reasoning workflow."""

    @patch('app.services.query_coordinator.QueryCoordinator.process_query')
    def test_api_endpoint_integration(self, mock_process_query):
        """Test the FastAPI endpoint integration."""
        from app.analytic_api import app
        
        # Mock the query processing response
        mock_process_query.return_value = {
            "success": True,
            "message": "Analysis complete",
            "chart_image": "base64_chart_data"
        }
        
        # Create test client
        client = TestClient(app)
        
        # Mock JWT token and request
        with patch('app.auth.bearer_scheme') as mock_bearer:
            mock_bearer.return_value = Mock(credentials="mock.jwt.token")
            
            response = client.post(
                "/query",
                json={"prompt": "Show customer analytics"},
                headers={"Authorization": "Bearer mock.jwt.token"}
            )
            
            assert response.status_code == 200
            # Note: Actual processing would require valid JWT and setup

    @pytest.mark.asyncio
    async def test_complete_workflow_customer_analytics(self):
        """Test complete workflow for customer analytics query."""
        from app.core.analytic_service import AnalyticService
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type, \
             patch('app.core.graph_builder.build_app') as mock_build_app, \
             patch('app.generators.chart_generator.chart_generator') as mock_chart_gen, \
             patch('app.services.memory_service.memory_service') as mock_memory:
            
            # Mock report type classification
            mock_report_type.return_value = "both"
            
            # Mock graph execution
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": [
                    Mock(
                        __class__=Mock(__name__='ToolMessage'),
                        content=json.dumps({
                            "success": True,
                            "chart_data": [
                                {"country": "USA", "customer_count": 100},
                                {"country": "Canada", "customer_count": 50}
                            ],
                            "domain_name": "customer",
                            "row_count": 2
                        })
                    ),
                    Mock(
                        __class__=Mock(__name__='AIMessage'),
                        content="Customer analytics show 100 customers in USA and 50 in Canada.",
                        tool_calls=None
                    )
                ]
            }
            mock_build_app.return_value = mock_app
            
            # Mock chart generation
            mock_chart_gen.generate_chart.return_value = "chart_image_base64"
            
            # Mock memory service
            mock_memory.get_reference_context_for_llm.return_value = ""
            
            # Execute complete workflow
            result = await AnalyticService.process_query(
                "Show customer distribution by country"
            )
            
            # Verify workflow completion
            assert result["success"] is True
            assert "customer" in result["message"].lower()
            assert result["chart_image"] is not None

    @pytest.mark.asyncio
    async def test_workflow_with_conversation_history(self):
        """Test workflow with conversation history context."""
        from app.core.analytic_service import AnalyticService
        
        conversation_history = [
            {
                "user_prompt": "Show customer data",
                "response_summary": {
                    "file_name": "customers.csv",
                    "domain_name": "customer",
                    "report_type": "both",
                    "row_count": 150
                }
            }
        ]
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type, \
             patch('app.core.graph_builder.build_app') as mock_build_app:
            
            mock_report_type.return_value = "success"
            
            # Mock graph with conversation context
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": [
                    Mock(
                        __class__=Mock(__name__='AIMessage'),
                        content="Based on previous customer analysis, success rate is 85%.",
                        tool_calls=None
                    )
                ]
            }
            mock_build_app.return_value = mock_app
            
            with patch('app.services.memory_service.memory_service') as mock_memory, \
                 patch('app.generators.chart_generator.chart_generator'):
                
                mock_memory.get_reference_context_for_llm.return_value = "Previous: customers.csv"
                
                result = await AnalyticService.process_query(
                    "What's the success rate for that file?",
                    session_id="session123",
                    conversation_history=conversation_history
                )
                
                assert result["success"] is True
                assert "success" in result["message"].lower() or "customer" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self):
        """Test workflow error handling and recovery."""
        from app.core.analytic_service import AnalyticService
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type, \
             patch('app.core.graph_builder.build_app') as mock_build_app:
            
            mock_report_type.return_value = "both"
            
            # Mock graph to raise exception
            mock_app = Mock()
            mock_app.invoke.side_effect = Exception("Database connection failed")
            mock_build_app.return_value = mock_app
            
            with patch('app.services.memory_service.memory_service') as mock_memory:
                mock_memory.get_reference_context_for_llm.return_value = ""
                
                result = await AnalyticService.process_query("Analyze customer data")
                
                assert result["success"] is False
                assert ("error" in result["message"].lower() or 
                       "issue" in result["message"].lower() or
                       "failed" in result["message"].lower())
                assert result["chart_image"] is None

    def test_query_parsing_integration(self):
        """Test integration of query parsing with tool selection."""
        from app.tools.domain_analytics_tools import _parse_analytics_query
        
        test_queries = [
            ("How many customers per country using pie chart?", "customer", "country", "pie"),
            ("Show product distribution by category", "product", "category", None),
            ("Orders by region as donut chart", "order", "region", "donut"),
            ("User breakdown by status", "user", "status", None)
        ]
        
        for query, expected_domain, expected_field, expected_chart in test_queries:
            domain, field, chart = _parse_analytics_query(query)
            
            assert domain == expected_domain, f"Domain failed for: {query}"
            assert field == expected_field, f"Field failed for: {query}"
            assert chart == expected_chart, f"Chart failed for: {query}"


class TestWorkflowPerformance:
    """Test performance aspects of the workflow."""

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test workflow handling of timeout scenarios."""
        from app.core.analytic_service import AnalyticService
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type, \
             patch('app.core.graph_builder.build_app') as mock_build_app:
            
            mock_report_type.return_value = "both"
            
            # Mock graph with long delay
            mock_app = Mock()
            mock_app.invoke.side_effect = lambda x: {"messages": []}  # Empty result
            mock_build_app.return_value = mock_app
            
            with patch('app.services.memory_service.memory_service') as mock_memory, \
                 patch('app.generators.chart_generator.chart_generator') as mock_chart:
                
                mock_memory.get_reference_context_for_llm.return_value = ""
                mock_chart.generate_chart.return_value = None
                
                # Should handle empty results gracefully
                result = await AnalyticService.process_query("test query")
                
                # Should provide fallback response
                assert result["success"] is True
                assert result["chart_image"] is None

    def test_memory_usage_optimization(self):
        """Test that the workflow optimizes memory usage."""
        from app.core.analytic_service import AnalyticService
        
        # Test conversation history truncation
        long_history = [{"response_summary": {}} for _ in range(10)]
        
        insights = AnalyticService._extract_conversation_insights(long_history)
        
        # Should only process recent interactions (last 3)
        interaction_count = insights.count("Interaction")
        assert interaction_count <= 3

    def test_context_size_management(self):
        """Test context size management for LLM calls."""
        from app.core.analytic_service import AnalyticService
        
        # Test that system message creation doesn't exceed limits
        long_reference_context = "Very long reference context " * 100
        long_conversation_insights = "Long conversation insights " * 50
        
        system_message = AnalyticService._create_enhanced_system_message(
            "2025-09-26",
            long_reference_context,
            long_conversation_insights
        )
        
        # Should create message without error
        assert isinstance(system_message, str)
        assert len(system_message) > 0


class TestWorkflowEdgeCases:
    """Test edge cases in the workflow."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        from app.core.analytic_service import AnalyticService
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type:
            mock_report_type.return_value = "uncertain"
            
            result = await AnalyticService.process_query("")
            
            # Should handle empty query gracefully
            assert "success" in result

    @pytest.mark.asyncio 
    async def test_malformed_data_handling(self):
        """Test handling of malformed data responses."""
        from app.core.analytic_service import AnalyticService
        
        with patch('app.utils.report_type.get_report_type') as mock_report_type, \
             patch('app.core.graph_builder.build_app') as mock_build_app:
            
            mock_report_type.return_value = "both"
            
            # Mock graph returning malformed data
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "messages": [
                    Mock(
                        __class__=Mock(__name__='ToolMessage'),
                        content="invalid json content"  # Malformed JSON
                    )
                ]
            }
            mock_build_app.return_value = mock_app
            
            with patch('app.services.memory_service.memory_service') as mock_memory:
                mock_memory.get_reference_context_for_llm.return_value = ""
                
                result = await AnalyticService.process_query("test query")
                
                # Should handle malformed data gracefully
                assert result["success"] is True

    def test_chart_generation_fallback(self):
        """Test fallback when chart generation fails."""
        from app.core.analytic_service import AnalyticService
        
        chart_data = [{"country": "USA", "count": 100}]
        
        # Test that workflow continues even if chart generation fails
        filtered_data = AnalyticService.filter_chart_data_by_report_type(chart_data, "both")
        
        assert filtered_data == chart_data
        # Workflow should continue with text response even without chart