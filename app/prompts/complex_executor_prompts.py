import hashlib
from app.prompts.base_prompt import SecurePromptTemplate, PromptSecurityError


class ComplexExecutorToolSelectionPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Complex Executor tool selection.
    
    Used in query_analytics steps to select appropriate analytics tool
    based on metric_type and target.
    """
    
    TEMPLATE = """You are an analytics tool selector. Your job is to:

1. Analyze the requested metric_type
2. Select the appropriate analytics tool
3. Format the tool call with correct parameters

Available Tools:
- generate_success_rate_report: For success rate analysis
- generate_failure_rate_report: For failure rate analysis

Rules:
- metric_type = "success_rate" → Use generate_success_rate_report
- metric_type = "failure_rate" → Use generate_failure_rate_report
- Provide EXACTLY ONE of domain_name or file_name (never both)
- Set the unused parameter to null

Example:
metric_type: "success_rate"
target_type: "file_name"
target: "product.csv"

→ Tool: generate_success_rate_report
→ Args: {"file_name": "product.csv", "domain_name": null}
"""
    
    # Calculate SHA-256 hash for template integrity verification
    TEMPLATE_HASH = hashlib.sha256(TEMPLATE.encode('utf-8')).hexdigest()
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt with proactive leakage prevention.
        
        Returns:
            System prompt with leakage prevention rules
        """
        return self.get_template_with_leakage_prevention()
    
    def validate_response_schema(self, data: dict) -> bool:
        """
        Validate tool call response structure.
        
        Expected structure:
        {
            "tool": "generate_success_rate_report" | "generate_failure_rate_report",
            "args": {
                "domain_name": str | None,
                "file_name": str | None
            }
        }
        
        Args:
            data: Parsed response dictionary
            
        Returns:
            True if schema is valid
            
        Raises:
            PromptSecurityError: If schema validation fails
        """
        required_keys = ['tool', 'args']
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise PromptSecurityError(f"Response missing required keys: {missing}")
        
        # Validate tool name (strict allowlist)
        tool_allowlist = ['generate_success_rate_report', 'generate_failure_rate_report']
        if data['tool'] not in tool_allowlist:
            raise PromptSecurityError(f"tool must be one of {tool_allowlist}, got: {data['tool']}")
        
        # Validate args structure
        if not isinstance(data['args'], dict):
            raise PromptSecurityError("'args' must be a dict")
        
        required_arg_keys = ['domain_name', 'file_name']
        for key in required_arg_keys:
            if key not in data['args']:
                raise PromptSecurityError(f"args missing required key: {key}")
        
        # Validate exactly one is set (not both, not neither)
        domain = data['args']['domain_name']
        file = data['args']['file_name']
        
        if domain is None and file is None:
            raise PromptSecurityError("Either domain_name or file_name must be set")
        if domain is not None and file is not None:
            raise PromptSecurityError("Cannot set both domain_name and file_name")
        
        # Sanitize the non-null value
        if domain is not None:
            if not isinstance(domain, str):
                raise PromptSecurityError(f"domain_name must be string, got {type(domain)}")
            if len(domain) > 500:
                raise PromptSecurityError("domain_name exceeds maximum length (500 chars)")
            data['args']['domain_name'] = self._sanitize_user_input(str(domain))
        
        if file is not None:
            if not isinstance(file, str):
                raise PromptSecurityError(f"file_name must be string, got {type(file)}")
            if len(file) > 500:
                raise PromptSecurityError("file_name exceeds maximum length (500 chars)")
            data['args']['file_name'] = self._sanitize_user_input(str(file))
        
        return True
    
    def _format_message(self, metric_type: str, target_type: str, target: str) -> str:
        """
        Format tool selection message with secure input handling.
        
        Uses structural isolation to prevent prompt injection.
        All inputs are validated and sanitized.
        
        Args:
            metric_type: Metric to analyze (will be validated against allowlist)
            target_type: Type of target (will be validated against allowlist)
            target: Target name (will be sanitized)
            
        Returns:
            Formatted message for LLM with security boundaries
            
        Raises:
            PromptSecurityError: If inputs fail validation
        """
        # Input validation
        metric_allowlist = ['success_rate', 'failure_rate']
        if metric_type not in metric_allowlist:
            raise PromptSecurityError(f"metric_type must be one of {metric_allowlist}, got: {metric_type}")
        
        target_type_allowlist = ['domain_name', 'file_name']
        if target_type not in target_type_allowlist:
            raise PromptSecurityError(f"target_type must be one of {target_type_allowlist}, got: {target_type}")
        
        if not target or not isinstance(target, str):
            raise PromptSecurityError("target must be a non-empty string")
        if len(target) > 500:
            raise PromptSecurityError("target exceeds maximum length (500 chars)")
        
        # Build tool parameters section with structural isolation
        # Note: metric_type and target_type are already validated against allowlists
        # Only target needs sanitization as it's user-provided
        safe_target = self._sanitize_user_input(target)
        
        params_content = f"""metric_type: {metric_type}
target_type: {target_type}
target: {safe_target}"""
        
        params_section = self.build_user_section(
            section_id="TOOL_PARAMS",
            user_input=params_content,
            header="Tool Selection Parameters",
            metadata={
                "metric_type": metric_type,
                "target_type": target_type,
                "sanitized": True
            }
        )
        
        return f"""Select the analytics tool for:

{params_section}

Call the appropriate tool with the correct parameters."""


class ComplexExecutorResponseFormattingPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Complex Executor response formatting.
    
    Formats comparison results into natural language with insights.
    """
    
    TEMPLATE = """You are a helpful analytics assistant specialized in presenting comparison results.

Your role:
- Analyze comparison data and present it in a conversational, easy-to-understand format
- Highlight key insights and differences between targets
- Use a friendly, professional tone
- Make the data accessible to non-technical users

Output requirements:
1. Directly answer the user's question
2. Clearly identify the winner and explain why
3. Highlight key differences between targets
4. Provide actionable insights about the comparison
5. Use appropriate emojis (🏆 for winner, 📊 for stats, etc.) for visual clarity
6. Keep the tone conversational but professional
7. Mention when a chart visualization is available
8. Return ONLY the message text (not JSON)

Format guidelines:
- Start with a direct answer to the user's question
- Use bullet points or structured paragraphs for clarity
- Include specific numbers and percentages
- End with a summary or recommendation if appropriate
"""
    
    # Calculate SHA-256 hash for template integrity verification
    TEMPLATE_HASH = hashlib.sha256(TEMPLATE.encode('utf-8')).hexdigest()
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt with proactive leakage prevention.
        
        Returns:
            System prompt with leakage prevention rules
        """
        return self.get_template_with_leakage_prevention()
    
    def validate_response_schema(self, data: dict) -> bool:
        """
        Validate response format.
        
        For plain text responses:
        - Must be non-empty string
        - Must be reasonable length
        - Must not contain leakage patterns
        
        Args:
            data: Response data (string or dict with 'message' key)
            
        Returns:
            True if valid
            
        Raises:
            PromptSecurityError: If validation fails
        """
        if not isinstance(data, (str, dict)):
            raise PromptSecurityError(f"Response must be string or dict, got {type(data)}")
        
        # If dict, extract message
        if isinstance(data, dict):
            if 'message' not in data:
                raise PromptSecurityError("Response dict must contain 'message' key")
            response_text = data['message']
        else:
            response_text = data
        
        # Validate response text
        if not response_text or not isinstance(response_text, str):
            raise PromptSecurityError("Response must be non-empty string")
        
        if len(response_text) > 50000:
            raise PromptSecurityError("Response exceeds maximum length (50,000 chars)")
        
        # Check for leakage (but only log, don't block - this is response formatting)
        # The response is supposed to be natural language, not structured data
        is_leaking, alert = self.detect_prompt_leakage(response_text)
        if is_leaking:
            # Log warning but don't block - formatted responses may contain these words naturally
            import logging
            logging.warning(f"Potential leakage in formatted response: {alert}")
        
        return True
    
    def _format_message(self, user_query: str, targets: list, winner: str, 
                       metric: str, details: list, has_chart: bool) -> str:
        """
        Format comparison response message with secure input handling.
        
        Uses structural isolation to prevent prompt injection.
        All user inputs are validated and sanitized.
        
        Args:
            user_query: Original user question (will be sanitized)
            targets: List of target names (will be validated)
            winner: Winning target name (will be sanitized)
            metric: Metric type (will be validated against allowlist)
            details: List of comparison detail dicts (will be validated and sanitized)
            has_chart: Whether chart is available (will be type-checked)
            
        Returns:
            Formatted message for LLM with security boundaries
            
        Raises:
            PromptSecurityError: If inputs fail validation
        """
        # Input validation
        if not user_query or not isinstance(user_query, str):
            raise PromptSecurityError("user_query must be a non-empty string")
        if len(user_query) > 5000:
            raise PromptSecurityError("user_query exceeds maximum length (5000 chars)")
        
        if not isinstance(targets, list):
            raise PromptSecurityError("targets must be a list")
        if len(targets) > 50:
            raise PromptSecurityError("targets exceeds maximum count (50)")
        
        metric_allowlist = ['success_rate', 'failure_rate']
        if metric not in metric_allowlist:
            raise PromptSecurityError(f"metric must be one of {metric_allowlist}, got: {metric}")
        
        if not isinstance(has_chart, bool):
            raise PromptSecurityError(f"has_chart must be a boolean, got {type(has_chart)}")
        
        # Validate details structure
        if not isinstance(details, list):
            raise PromptSecurityError("details must be a list")
        if len(details) > 50:
            raise PromptSecurityError("details exceeds maximum count (50)")
        
        required_keys = ['target', 'metric_value', 'total_requests', 'successful_requests', 'failed_requests']
        for idx, detail in enumerate(details):
            if not isinstance(detail, dict):
                raise PromptSecurityError(f"detail {idx} must be a dict")
            for key in required_keys:
                if key not in detail:
                    raise PromptSecurityError(f"detail {idx} missing required key: {key}")
        
        # Structural isolation for user query
        query_section = self.build_user_section(
            section_id="USER_QUERY",
            user_input=user_query,
            header="User Question",
            metadata={"source": "user", "sanitized": True}
        )
        
        # Build comparison results safely
        results_text = f"I compared {metric.replace('_', ' ')} across {len(targets)} targets:\n\n"
        
        for detail in details:
            is_winner = detail["target"] == winner
            # Sanitize each detail value
            safe_target = self._sanitize_user_input(str(detail['target']))
            safe_value = self._sanitize_user_input(str(detail['metric_value']))
            
            results_text += f"{'🏆 ' if is_winner else ''}**{safe_target}**:\n"
            results_text += f"  - {metric.replace('_', ' ').title()}: {safe_value}%\n"
            results_text += f"  - Total Requests: {int(detail['total_requests'])}\n"
            results_text += f"  - Successful: {int(detail['successful_requests'])}\n"
            results_text += f"  - Failed: {int(detail['failed_requests'])}\n\n"
        
        results_section = self.build_user_section(
            section_id="COMPARISON_RESULTS",
            user_input=results_text,
            header="Comparison Data",
            metadata={"metric": metric, "targets_count": len(targets)}
        )
        
        # Sanitize winner name
        safe_winner = self._sanitize_user_input(str(winner))
        
        return f"""{query_section}

{results_section}

Winner: {safe_winner}
Chart visualization: {'Available' if has_chart else 'Not available'}

Please format this into a natural, conversational response following the guidelines."""
