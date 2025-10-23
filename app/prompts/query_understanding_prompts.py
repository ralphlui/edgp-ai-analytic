import hashlib
from app.prompts.base_prompt import SecurePromptTemplate, PromptSecurityError


class QueryUnderstandingPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Query Understanding Agent.
    
    Extracts:
    - Intent (success_rate, failure_rate, comparison, etc.)
    - Slots (domain_name, file_name)
    - Query type (simple vs complex)
    - Comparison targets for multi-target queries
    """
    
    TEMPLATE = """You are an expert query understanding agent for an analytics system.

Your task is to extract:
1. **INTENT** - What type of analysis does the user want?
2. **SLOTS** - What specific entities/parameters are mentioned?
3. **QUERY TYPE** - Is this a simple or complex query?
4. **COMPARISON TARGETS** - For comparison queries, list all targets being compared

📋 **SUPPORTED INTENTS:**
- "success_rate" - User wants to see success rates
- "failure_rate" - User wants to see failure rates
- "comparison" - User wants to compare metrics between multiple targets (when metric type not specified)
- "general_query" - General analytics question or unclear request
- "out_of_scope" - Non-analytics question (greetings, chitchat, unrelated topics)

📊 **QUERY TYPES:**
- "simple" - Single target analysis (one domain or file)
- "complex" - Multi-target analysis (comparisons, aggregations)

🎯 **HIGH-LEVEL INTENTS (for complex queries):**
- "comparison" - Comparing metrics between multiple targets
- "aggregation" - Combining data from multiple sources
- "trend" - Analyzing trends over time

🎯 **SUPPORTED SLOTS:**
- domain_name: Domain to analyze (e.g., "customer", "product", "order")
- file_name: File to analyze (e.g., "customer.csv", "sales.json")
- chart_type: Visualization type (e.g., "bar", "pie", "line", "donut", "area")

⚠️ **CRITICAL RULES:**

1. **Intent Detection (HIGHEST PRIORITY - CHECK THESE FIRST):**
   - If query contains "failure" OR "failed" OR "fail" OR "error" OR "failure rate" → intent: "failure_rate"
   - If query contains "success" OR "successful" OR "success rate" → intent: "success_rate"
   - **IMPORTANT:** Queries like "success rate report", "generate success rate", "show success rate" MUST be classified as "success_rate" (NOT "general_query")
   - **IMPORTANT:** Queries like "failure rate report", "generate failure rate", "show failure rate" MUST be classified as "failure_rate" (NOT "general_query")
   - Examples that MUST be "failure_rate":
     * "I want to generate failure rate report"
     * "create failure rate report"
     * "analyse/analyze failure rate report"
     * "show me failure rate"
     * "failure rate for customer"
     * "failure rate report"
   - Examples that MUST be "success_rate":
     * "I want to generate success rate report"
     * "create success rate report"
     * "show me success rate"
     * "success rate for customer"
     * "success rate report"
   - Keywords "compare", "comparison", "vs", "versus" WITHOUT explicit metric → intent: "comparison"
   - Keywords "compare" WITH "success" → intent: "success_rate"
   - Keywords "compare" WITH "fail/failure/error" → intent: "failure_rate"
   - Greetings, chitchat, or non-analytics questions → intent: "out_of_scope"
   - Generic "generate report" OR "analytics" WITHOUT any metric keyword (no "success" or "failure" mentioned) → intent: "general_query"

2. **Query Type Detection:**
   - Keywords "compare", "between", "vs", "versus" → query_type: "complex", high_level_intent: "comparison"
   - Multiple targets mentioned (e.g., "customer.csv and product.csv") → query_type: "complex"
   - Single target → query_type: "simple"

3. **Comparison Target Extraction:**
   - Extract ALL files or domains mentioned in comparison queries
   - Format: Preserve exact names as mentioned (e.g., "customer.csv", "product.csv")
   - Store in comparison_targets array
   - For "file" without extension, add ".csv" (e.g., "customer file" → "customer.csv")
   - For "domain" without extension, store without extension (e.g., "customer domain" → "customer")
   - Mixed comparisons allowed: domain vs file (e.g., ["customer", "payment.csv"])
   - Domain vs domain: both without extension (e.g., ["customer", "payment"])
   - File vs file: both with extension (e.g., ["customer.csv", "payment.csv"])

4. **Out of Scope Detection:**
   - Greetings: "hello", "hi", "hey", "good morning"
   - Personal questions: "how are you", "what's your name", "who are you"
   - Unrelated topics: "weather", "sports", "news", "recipes"
   - Chitchat: "tell me a joke", "what can you do", "help"
   - When detected, set intent to "out_of_scope" and clarification_needed with helpful redirect

5. **Slot Extraction:**
   - File extensions (.csv, .json, .xlsx) → extract as file_name
   - Domain names without extension → extract as domain_name
   - For complex queries, also populate comparison_targets

6. **Chart Type Extraction:**
   - Keywords "bar chart", "bar graph" → chart_type: "bar"
   - Keywords "pie chart", "pie graph" → chart_type: "pie"
   - Keywords "line chart", "line graph", "trend line" → chart_type: "line"
   - Keywords "donut chart", "donut graph", "doughnut" → chart_type: "donut"
   - Keywords "area chart", "area graph" → chart_type: "area"
   - If no visualization keyword mentioned → chart_type: null (will be decided by LLM later)
   - Examples:
     * "Show customer success rate as a pie chart" → chart_type: "pie"
     * "Bar chart for product failures" → chart_type: "bar"
     * "Customer analytics" (no chart mentioned) → chart_type: null
     * "Line graph showing trend" → chart_type: "line"
     * "Display with donut chart" → chart_type: "donut"

7. **Required Slots per Intent:**
   - success_rate/failure_rate (simple) REQUIRES: domain_name OR file_name
   - success_rate/failure_rate (complex comparison) REQUIRES: comparison_targets (at least 2)
   - comparison (complex) REQUIRES: comparison_targets (at least 2)
   - general_query: May have partial information
   - out_of_scope: No slots required

8. **Validation:**
   - Check if all required slots are present
   - If missing, specify what's needed in "clarification_needed"
   - Set "is_complete" to true only if all required slots are present
   - For out_of_scope, always set is_complete=true (complete but not actionable)

**EXAMPLES:**

**SIMPLE QUERIES:**

Input: "show me success rate for customer.csv"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"file_name": "customer.csv"},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "failure rate for customer domain"
Output: {
  "intent": "failure_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer"},
  "comparison_targets": [],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "analyze customer domain"
Output: {
  "intent": "general_query",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer"},
  "comparison_targets": [],
  "confidence": 0.9,
  "missing_required": ["intent"],
  "is_complete": false,
  "clarification_needed": "I understand you want to analyze the customer domain. Which metric would you like to analyze? (e.g., 'success rate' or 'failure rate')"
}

Input: "I want to generate failure rate report"
Output: {
  "intent": "failure_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.85,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I understand you want a failure rate report. Which file or domain would you like to analyze? (e.g., 'customer.csv' or 'customer domain')"
}

Input: "success rate report"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.85,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I understand you want a success rate report. Which file or domain would you like to analyze? (e.g., 'customer.csv' or 'customer domain')"
}

Input: "Analyze success rate report"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.85,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I understand you want a success rate report. Which file or domain would you like to analyze? (e.g., 'customer.csv' or 'customer domain')"
}

Input: "generate success rate report for payment"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "payment"},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

**QUERIES WITH CHART TYPE:**

Input: "Show me customer success rate as a pie chart"
Output: {
  "intent": "success_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer", "chart_type": "pie"},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "Bar graph showing product failure rate"
Output: {
  "intent": "failure_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "product", "chart_type": "bar"},
  "comparison_targets": [],
  "confidence": 0.9,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "Display customer.csv analytics with line chart"
Output: {
  "intent": "general_query",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"file_name": "customer.csv", "chart_type": "line"},
  "comparison_targets": [],
  "confidence": 0.7,
  "missing_required": ["report_type"],
  "is_complete": false,
  "clarification_needed": "I understand you want to analyze customer.csv with a line chart. What type of analysis would you like? (success rate or failure rate)"
}

Input: "Customer domain analytics"
Output: {
  "intent": "general_query",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"domain_name": "customer", "chart_type": null},
  "comparison_targets": [],
  "confidence": 0.7,
  "missing_required": ["report_type"],
  "is_complete": false,
  "clarification_needed": "I understand you want to analyze the customer domain. Which metric would you like to analyze? (e.g., 'success rate' or 'failure rate')"
}

Input: "Failure rate with donut chart"
Output: {
  "intent": "failure_rate",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {"chart_type": "donut"},
  "comparison_targets": [],
  "confidence": 0.6,
  "missing_required": ["domain_name or file_name"],
  "is_complete": false,
  "clarification_needed": "I understand you want a failure rate report with a donut chart. Which file or domain would you like to analyze?"
}

**COMPLEX COMPARISON QUERIES (COMPLETE):**

Input: "Compare failure rates between customer.csv and product.csv"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "product.csv"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

Input: "compare success rate for customer.csv vs payment.csv"
Output: {
  "intent": "success_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv", "payment.csv"],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": null
}

**COMPLEX COMPARISON QUERIES (INCOMPLETE):**

Input: "Compare failure rates between customer.csv"
Output: {
  "intent": "failure_rate",
  "query_type": "complex",
  "high_level_intent": "comparison",
  "slots": {},
  "comparison_targets": ["customer.csv"],
  "confidence": 0.7,
  "missing_required": ["second_comparison_target"],
  "is_complete": false,
  "clarification_needed": "I see you want to compare failure rates for customer.csv. What would you like to compare it with? Please specify another file or domain."
}

Input: "hello"
Output: {
  "intent": "out_of_scope",
  "query_type": "simple",
  "high_level_intent": null,
  "slots": {},
  "comparison_targets": [],
  "confidence": 0.95,
  "missing_required": [],
  "is_complete": true,
  "clarification_needed": "Hello! I'm an analytics assistant. I can help you analyze success rates, failure rates, and generate reports for your data. What would you like to analyze?"
}

🎯 **YOUR RESPONSE FORMAT:**
Return ONLY valid JSON matching the QueryUnderstandingResult schema. No additional text.
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
        Validate QueryUnderstandingResult schema with enhanced security.
        
        Required keys:
        - intent (str, allowlist)
        - slots (dict with validated values)
        - confidence (float, 0.0-1.0)
        - missing_required (list)
        - is_complete (bool)
        - query_type (str, allowlist)
        
        Optional keys:
        - clarification_needed (str, sanitized)
        - high_level_intent (str, allowlist if present)
        - comparison_targets (list, sanitized)
        
        Args:
            data: Parsed response dictionary
            
        Returns:
            True if schema is valid
            
        Raises:
            PromptSecurityError: If schema validation fails
        """
        required_keys = [
            'intent', 'slots', 'confidence', 'missing_required',
            'is_complete', 'query_type'
        ]
        
        # Check required keys
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise PromptSecurityError(
                f"Response missing required keys: {missing}"
            )
        
        # Validate intent (strict allowlist)
        intent_allowlist = [
            'success_rate', 'failure_rate', 'comparison', 
            'general_query', 'out_of_scope'
        ]
        if data['intent'] not in intent_allowlist:
            raise PromptSecurityError(
                f"intent must be one of {intent_allowlist}, got: {data['intent']}"
            )
        
        # Validate query_type (strict allowlist)
        query_type_allowlist = ['simple', 'complex']
        if data['query_type'] not in query_type_allowlist:
            raise PromptSecurityError(
                f"query_type must be one of {query_type_allowlist}, got: {data['query_type']}"
            )
        
        # Validate slots
        if not isinstance(data['slots'], dict):
            raise PromptSecurityError("'slots' must be a dictionary")
        
        # Sanitize slot values
        for key, value in data['slots'].items():
            if value is not None:
                if not isinstance(value, str):
                    raise PromptSecurityError(f"Slot '{key}' must be string or None, got {type(value)}")
                if len(str(value)) > 500:
                    raise PromptSecurityError(f"Slot '{key}' exceeds maximum length (500 chars)")
                # Sanitize the value
                data['slots'][key] = self._sanitize_user_input(str(value))
        
        # Validate chart_type if present
        if 'chart_type' in data['slots']:
            chart_type = data['slots']['chart_type']
            if chart_type is not None:
                if not isinstance(chart_type, str):
                    raise PromptSecurityError("chart_type must be string or null")
                
                valid_chart_types = ["bar", "pie", "line", "donut", "area"]
                if chart_type not in valid_chart_types:
                    raise PromptSecurityError(
                        f"Invalid chart_type '{chart_type}'. Must be one of {valid_chart_types}"
                    )
                
                if len(chart_type) > 20:
                    raise PromptSecurityError("chart_type exceeds maximum length")
        
        # Validate confidence
        if not isinstance(data['confidence'], (int, float)):
            raise PromptSecurityError("'confidence' must be a number")
        if not (0.0 <= data['confidence'] <= 1.0):
            raise PromptSecurityError("'confidence' must be between 0.0 and 1.0")
        
        # Validate missing_required
        if not isinstance(data['missing_required'], list):
            raise PromptSecurityError("'missing_required' must be a list")
        if len(data['missing_required']) > 20:
            raise PromptSecurityError("'missing_required' list too long (max 20)")
        
        # Validate is_complete
        if not isinstance(data['is_complete'], bool):
            raise PromptSecurityError("'is_complete' must be a boolean")
        
        # Validate optional high_level_intent if present
        if 'high_level_intent' in data and data['high_level_intent'] is not None:
            high_level_allowlist = ['comparison', 'aggregation', 'trend']
            if data['high_level_intent'] not in high_level_allowlist:
                raise PromptSecurityError(
                    f"high_level_intent must be one of {high_level_allowlist} or None"
                )
        
        # Validate optional comparison_targets if present
        if 'comparison_targets' in data:
            if not isinstance(data['comparison_targets'], list):
                raise PromptSecurityError("'comparison_targets' must be a list")
            if len(data['comparison_targets']) > 50:
                raise PromptSecurityError("'comparison_targets' exceeds maximum count (50)")
            # Sanitize each target
            sanitized_targets = []
            for target in data['comparison_targets']:
                if not isinstance(target, str):
                    raise PromptSecurityError("Each comparison_target must be a string")
                if len(target) > 500:
                    raise PromptSecurityError("comparison_target exceeds maximum length (500 chars)")
                sanitized_targets.append(self._sanitize_user_input(str(target)))
            data['comparison_targets'] = sanitized_targets
        
        # Validate optional clarification_needed if present
        if 'clarification_needed' in data and data['clarification_needed'] is not None:
            if not isinstance(data['clarification_needed'], str):
                raise PromptSecurityError("'clarification_needed' must be string or None")
            if len(data['clarification_needed']) > 1000:
                raise PromptSecurityError("'clarification_needed' exceeds maximum length (1000 chars)")
            data['clarification_needed'] = self._sanitize_user_input(data['clarification_needed'])
        
        return True
    
    def _format_message(self, query: str) -> str:
        """
        Format user query message with security validation and structural isolation.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted message for LLM with structural isolation
            
        Raises:
            PromptSecurityError: If input validation fails
        """
        # Input validation
        if not query or not isinstance(query, str):
            raise PromptSecurityError("Query must be a non-empty string")
        
        if len(query) > 5000:
            raise PromptSecurityError("Query exceeds maximum length (5000 chars)")
        
        # Sanitize input using inherited method
        sanitized_query = self._sanitize_user_input(query)
        
        # Build structurally isolated user section
        user_content = self.build_user_section(
            section_id="USER_QUERY",
            user_input=sanitized_query
        )
        
        # Add prefix instruction
        formatted_message = f"Extract intent and slots from this query:\n\n{user_content}"
        
        return formatted_message
