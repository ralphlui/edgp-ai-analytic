import hashlib
from app.prompts.base_prompt import SecurePromptTemplate, PromptSecurityError


class SimpleExecutorToolSelectionPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Simple Executor tool selection.
    
    Uses hybrid approach:
    1. Priority: Use report_type if provided (from multi-turn context)
    2. Fallback: Analyze user query keywords if report_type missing
    """
    
    TEMPLATE = """You are an analytics tool selector. Your job is to:

1. **Check if report_type is provided** (from previous conversation context)
2. **If yes**: Use report_type to select the tool directly (HIGHEST PRIORITY)
3. **If no**: Analyze the user's query to determine intent from keywords
4. **Format the tool call** with the correct parameters in JSON format

## PRIORITY RULES (Follow in order):

### Rule 1: Use report_type if provided (HIGHEST PRIORITY)
If report_type is given in the available parameters:
- report_type = "success_rate" → MUST call generate_success_rate_report
- report_type = "failure_rate" → MUST call generate_failure_rate_report
- Do NOT analyze the user query for intent - trust the report_type

### Rule 2: Analyze user query if report_type is missing (FALLBACK)
If report_type is null/not provided:
- Analyze user query for intent keywords:
  - "success", "wins", "passed", "uptime", "completion" → generate_success_rate_report
  - "failure", "errors", "issues", "problems", "fail" → generate_failure_rate_report

## Available Tools:

### generate_success_rate_report
- **Purpose**: Analyze how often requests succeed
- **When to use**: 
  - report_type = "success_rate" (explicit - PRIORITY), OR
  - User asks about success, wins, completion, passed requests, uptime (inferred)
- **Parameters**: 
  - domain_name (optional): The domain to analyze (e.g., "customer", "payment")
  - file_name (optional): The file to analyze (e.g., "customer.csv", "data.json")
- **JSON Format**: {"domain_name": "value"} OR {"file_name": "value"}
- **Constraint**: Provide EXACTLY ONE of domain_name or file_name, never both

### generate_failure_rate_report
- **Purpose**: Analyze how often requests fail
- **When to use**: 
  - report_type = "failure_rate" (explicit - PRIORITY), OR
  - User asks about failures, errors, issues, problems, failed requests (inferred)
- **Parameters**:
  - domain_name (optional): The domain to analyze
  - file_name (optional): The file to analyze
- **JSON Format**: {"domain_name": "value"} OR {"file_name": "value"}
- **Constraint**: Provide EXACTLY ONE of domain_name or file_name, never both

## Important Rules:

1. **report_type Priority**: ALWAYS use report_type if provided (takes precedence over query analysis)
2. **XOR Constraint**: NEVER provide both domain_name AND file_name - choose ONE
3. **Null values**: Set the unused parameter to null/None
4. **Case sensitivity**: Tool names are case-sensitive
5. **JSON structure**: Arguments must be valid JSON with quoted keys

Now analyze the available parameters and select the appropriate tool."""
    
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
        Validate tool call schema with enhanced security.
        
        Expected format:
        {
            "tool": str (allowlist),
            "arguments": dict (validated)
        }
        
        Args:
            data: Parsed response dictionary
            
        Returns:
            True if schema is valid
            
        Raises:
            PromptSecurityError: If schema validation fails
        """
        # Check required keys
        if 'tool' not in data:
            raise PromptSecurityError("Response missing required key: 'tool'")
        
        if 'arguments' not in data:
            raise PromptSecurityError("Response missing required key: 'arguments'")
        
        # Validate tool name (strict allowlist)
        tool_allowlist = [
            'generate_success_rate_report',
            'generate_failure_rate_report',
        ]
        
        if data['tool'] not in tool_allowlist:
            raise PromptSecurityError(
                f"tool must be one of {tool_allowlist}, got: {data['tool']}"
            )
        
        # Validate arguments
        if not isinstance(data['arguments'], dict):
            raise PromptSecurityError("'arguments' must be a dictionary")
        
        # Validate argument keys based on tool
        valid_arg_keys = {'domain_name', 'file_name'}
        for key in data['arguments'].keys():
            if key not in valid_arg_keys:
                raise PromptSecurityError(f"Invalid argument key: {key}")
        
        # Validate argument values
        for key, value in data['arguments'].items():
            if value is not None:
                if not isinstance(value, str):
                    raise PromptSecurityError(f"Argument '{key}' must be string or None")
                if len(value) > 500:
                    raise PromptSecurityError(f"Argument '{key}' exceeds maximum length (500 chars)")
                # Sanitize the value
                data['arguments'][key] = self._sanitize_user_input(value)
        
        return True
    
    def _format_message(self, user_query: str, report_type: str, domain_name: str, file_name: str) -> str:
        """
        Format tool selection message with security validation and structural isolation.
        
        Args:
            user_query: User's natural language query
            report_type: Report type from context (can be None)
            domain_name: Domain name from context (can be None)
            file_name: File name from context (can be None)
            
        Returns:
            Formatted message with structural isolation
            
        Raises:
            PromptSecurityError: If input validation fails
        """
        # Input validation
        if not user_query or not isinstance(user_query, str):
            raise PromptSecurityError("user_query must be a non-empty string")
        
        if len(user_query) > 5000:
            raise PromptSecurityError("user_query exceeds maximum length (5000 chars)")
        
        # Validate optional parameters
        for param_name, param_value in [
            ('report_type', report_type),
            ('domain_name', domain_name),
            ('file_name', file_name)
        ]:
            if param_value is not None:
                if not isinstance(param_value, str):
                    raise PromptSecurityError(f"{param_name} must be string or None")
                if len(param_value) > 500:
                    raise PromptSecurityError(f"{param_name} exceeds maximum length (500 chars)")
        
        # Sanitize all inputs
        sanitized_query = self._sanitize_user_input(user_query)
        sanitized_report_type = self._sanitize_user_input(report_type) if report_type else "null"
        sanitized_domain_name = self._sanitize_user_input(domain_name) if domain_name else "null"
        sanitized_file_name = self._sanitize_user_input(file_name) if file_name else "null"
        
        # Build structurally isolated user section
        user_query_section = self.build_user_section(
            section_id="USER_QUERY",
            user_input=sanitized_query
        )
        
        # Build parameters section with structural isolation
        params_content = f"""- report_type: {sanitized_report_type}  ← CHECK THIS FIRST!
- domain_name: {sanitized_domain_name}
- file_name: {sanitized_file_name}"""
        
        params_section = self.build_user_section(
            section_id="AVAILABLE_PARAMETERS",
            user_input=params_content
        )
        
        # Combine sections
        formatted_message = f"""User Query: {user_query_section}

Available Parameters (extracted from conversation):
{params_section}

IMPORTANT: If report_type is provided, use it to select the tool (PRIORITY). Otherwise, analyze the user query.

Select the appropriate analytics tool and call it with the correct parameters."""
        
        return formatted_message


class SimpleExecutorResponseFormattingPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Simple Executor response formatting.
    
    Formats raw analytics data into natural language with chart integration.
    """
    
    TEMPLATE = """You are a helpful analytics assistant. Your job is to present analytics results in a clear, natural way.

## Your Task:

Take the raw analytics data and create a short, conversational message that:
1. **Answers the user's question directly**
2. **Highlights the key metric** (success rate or failure rate)
3. **Mentions the target** (file or domain name)
4. **Is concise** (1-2 sentences max)

## Guidelines:

- Use natural language, not technical jargon
- Start with the answer, not filler words
- Include specific numbers when available
- If a chart is provided, mention it naturally
- Be conversational but professional

## Examples:

**Example 1: Success Rate (with chart)**
User Query: "What's the success rate for customer.csv?"
Raw Data: {"file_name": "customer.csv", "success_count": 850, "total_requests": 1000}
Chart Available: Yes

Response: "The success rate for customer.csv is 85% (850 out of 1,000 requests succeeded). I've created a chart showing this data."

**Example 2: Failure Rate (without chart)**
User Query: "Show me failures in payment domain"
Raw Data: {"domain_name": "payment", "failure_count": 25, "total_requests": 500}
Chart Available: No

Response: "The payment domain has a 5% failure rate, with 25 failures out of 500 total requests."

**Example 3: Multi-turn continuation**
User Query: "customer domain"
Raw Data: {"domain_name": "customer", "success_count": 1200, "total_requests": 1500}
Chart Available: Yes

Response: "The customer domain has an 80% success rate (1,200 successful requests out of 1,500 total). Chart generated for visualization."

Now format the analytics results:"""
    
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
        Validate response text with enhanced security.
        
        Response should be a plain text string with:
        - Reasonable length (not empty, not excessive)
        - No prompt leakage patterns
        
        Args:
            data: Response string or dict
            
        Returns:
            True if schema is valid
            
        Raises:
            PromptSecurityError: If validation fails
        """
        # Handle both string and dict responses
        if isinstance(data, dict):
            if 'response' not in data:
                raise PromptSecurityError("Response dict missing 'response' key")
            response_text = data['response']
        elif isinstance(data, str):
            response_text = data
        else:
            raise PromptSecurityError("Response must be string or dict")
        
        # Validate response length
        if not response_text or len(response_text.strip()) == 0:
            raise PromptSecurityError("Response cannot be empty")
        
        if len(response_text) > 5000:
            raise PromptSecurityError("Response exceeds maximum length (5000 chars)")
        
        # Detect prompt leakage in response
        if self.detect_prompt_leakage(response_text):
            raise PromptSecurityError("Response contains system prompt leakage")
        
        return True
    
    def _format_message(self, user_query: str, data: dict, has_chart: bool) -> str:
        """
        Format response formatting message with security validation and structural isolation.
        
        Args:
            user_query: User's natural language query
            data: Analytics data dictionary
            has_chart: Whether chart is available
            
        Returns:
            Formatted message with structural isolation
            
        Raises:
            PromptSecurityError: If input validation fails
        """
        # Input validation
        if not user_query or not isinstance(user_query, str):
            raise PromptSecurityError("user_query must be a non-empty string")
        
        if len(user_query) > 5000:
            raise PromptSecurityError("user_query exceeds maximum length (5000 chars)")
        
        if not isinstance(data, dict):
            raise PromptSecurityError("data must be a dictionary")
        
        if not isinstance(has_chart, bool):
            raise PromptSecurityError("has_chart must be a boolean")
        
        # Sanitize user query
        sanitized_query = self._sanitize_user_input(user_query)
        
        # Validate and sanitize data dict
        if len(str(data)) > 10000:
            raise PromptSecurityError("data dictionary too large (max 10000 chars)")
        
        # Build structurally isolated user section
        user_query_section = self.build_user_section(
            section_id="USER_QUERY",
            user_input=sanitized_query
        )
        
        # Build data section with structural isolation
        data_str = str(data)
        data_section = self.build_user_section(
            section_id="ANALYTICS_DATA",
            user_input=data_str
        )
        
        # Combine sections
        formatted_message = f"""User Query: {user_query_section}

Analytics Data: {data_section}

Chart Available: {"Yes" if has_chart else "No"}

Generate a natural, concise message for the user:"""
        
        return formatted_message
