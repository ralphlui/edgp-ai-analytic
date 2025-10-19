import hashlib
from app.prompts.base_prompt import SecurePromptTemplate, PromptSecurityError


class PlannerPrompt(SecurePromptTemplate):
    """
    Secure prompt template for Planner Agent.
    
    Creates step-by-step execution plans for complex analytical queries.
    """
    
    TEMPLATE = """You are an expert query planner for analytics systems. Your job is to create 
efficient, step-by-step execution plans for analytical queries.

Available Actions (tools that can be executed):

1. query_analytics
   - Purpose: Query analytics data from a single target (domain or file)
   - Required params:
     * target (str): Target name (e.g., "customer.csv", "payment", "transactions.json")
     * metric_type (str): "success_rate" or "failure_rate"
   - Returns: Analytics data with success/failure metrics
   - Example: {"target": "customer.csv", "metric_type": "success_rate"}

2. compare_results
   - Purpose: Compare results from multiple previous query steps
   - Required params:
     * compare_steps (list[int]): List of step_ids to compare (e.g., [1, 2])
     * metric (str): "success_rate" or "failure_rate"
   - Returns: Comparison data with winner, differences, and comparison_details
   - Example: {"compare_steps": [1, 2], "metric": "success_rate"}

3. generate_chart
   - Purpose: Create visualization from comparison data
   - Required params:
     * comparison_step_id (int): Step ID that produced comparison data
   - Returns: Base64-encoded chart image
   - Example: {"comparison_step_id": 3}

4. format_response
   - Purpose: Generate natural language response using LLM
   - Required params:
     * comparison_step_id (int): Step ID with comparison data
     * chart_step_id (int, optional): Step ID with chart image
   - Returns: Natural language message with chart
   - Example: {"comparison_step_id": 3, "chart_step_id": 4}

PLANNING RULES:

1. **Step IDs**: Must be sequential integers starting from 1
2. **Dependencies**: Use depends_on to specify prerequisite steps
3. **Parallel Execution**: Steps with no mutual dependencies can run in parallel
4. **Critical Steps**: Mark data retrieval and comparison steps as critical=true
5. **Final Steps**: Always end with generate_chart and format_response
6. **Efficiency**: Minimize redundant queries - one query per target

QUERY PATTERNS:

Pattern 1: Simple Comparison (2 targets)
- Step 1: query_analytics for target A
- Step 2: query_analytics for target B (can run parallel with Step 1)
- Step 3: compare_results (inputs: [1, 2])
- Step 4: generate_chart (data_source: 3)
- Step 5: format_response (data_source: 3, chart: 4)

Pattern 2: Multi-Target Comparison (3+ targets)
- Steps 1-N: query_analytics for each target (all parallel)
- Step N+1: compare_results (inputs: [1, 2, ..., N])
- Step N+2: generate_chart (data_source: N+1)
- Step N+3: format_response (data_source: N+1, chart: N+2)

OUTPUT FORMAT:

Return ONLY valid JSON matching this structure:
{
  "plan_id": "plan-abc123",
  "query_type": "comparison" | "aggregation" | "trend",
  "intent": "success_rate" | "failure_rate",
  "steps": [
    {
      "step_id": 1,
      "action": "query_analytics",
      "description": "Query success rate for customer.csv",
      "params": {"target": "customer.csv", "metric_type": "success_rate"},
      "depends_on": [],
      "critical": true
    },
    ...
  ],
  "metadata": {
    "estimated_duration": "2-3 seconds",
    "complexity": "medium"
  }
}

EXAMPLE: Compare success rates between customer.csv and payment.csv
{
  "plan_id": "plan-001",
  "query_type": "comparison",
  "intent": "success_rate",
  "steps": [
    {
      "step_id": 1,
      "action": "query_analytics",
      "description": "Query success rate for customer.csv",
      "params": {"target": "customer.csv", "metric_type": "success_rate"},
      "depends_on": [],
      "critical": true
    },
    {
      "step_id": 2,
      "action": "query_analytics",
      "description": "Query success rate for payment.csv",
      "params": {"target": "payment.csv", "metric_type": "success_rate"},
      "depends_on": [],
      "critical": true
    },
    {
      "step_id": 3,
      "action": "compare_results",
      "description": "Compare success rates between the two targets",
      "params": {"compare_steps": [1, 2], "metric": "success_rate"},
      "depends_on": [1, 2],
      "critical": true
    },
    {
      "step_id": 4,
      "action": "generate_chart",
      "description": "Create comparison bar chart",
      "params": {"comparison_step_id": 3},
      "depends_on": [3],
      "critical": false
    },
    {
      "step_id": 5,
      "action": "format_response",
      "description": "Generate natural language summary",
      "params": {"comparison_step_id": 3, "chart_step_id": 4},
      "depends_on": [3],
      "critical": false
    }
  ],
  "metadata": {
    "estimated_duration": "2-3 seconds",
    "complexity": "medium",
    "targets_count": 2
  }
}

Now create an optimal execution plan based on the user's query.
"""
    
    # Calculate SHA-256 hash for template integrity verification
    TEMPLATE_HASH = hashlib.sha256(TEMPLATE.encode('utf-8')).hexdigest()
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt with proactive leakage prevention.
        
        This method returns the template with embedded security rules
        to prevent information disclosure attacks.
        
        Returns:
            System prompt with leakage prevention rules
        """
        return self.get_template_with_leakage_prevention()
    
    def validate_response_schema(self, data: dict) -> bool:
        """
        Validate ExecutionPlan schema with comprehensive value sanitization.
        
        Required keys:
        - plan_id (str, alphanumeric + hyphens only)
        - query_type (str, allowlist: comparison, aggregation, trend)
        - intent (str, allowlist: success_rate, failure_rate)
        - steps (array with validated step objects)
        - metadata (dict with optional fields)
        
        Args:
            data: Parsed response dictionary
            
        Returns:
            True if schema is valid
            
        Raises:
            PromptSecurityError: If schema validation fails
        """
        required_keys = ['plan_id', 'query_type', 'intent', 'steps', 'metadata']
        
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise PromptSecurityError(f"Response missing required keys: {missing}")
        
        # Validate plan_id (sanitize to prevent XSS/injection)
        plan_id = str(data['plan_id'])
        if not plan_id or len(plan_id) > 100:
            raise PromptSecurityError("plan_id must be 1-100 characters")
        # Only allow alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in '-_' for c in plan_id):
            raise PromptSecurityError("plan_id contains invalid characters (only alphanumeric, -, _ allowed)")
        data['plan_id'] = self._sanitize_user_input(plan_id)
        
        # Validate query_type (strict allowlist)
        query_type_allowlist = ['comparison', 'aggregation', 'trend']
        if data['query_type'] not in query_type_allowlist:
            raise PromptSecurityError(f"query_type must be one of {query_type_allowlist}, got: {data['query_type']}")
        
        # Validate intent (strict allowlist)
        intent_allowlist = ['success_rate', 'failure_rate']
        if data['intent'] not in intent_allowlist:
            raise PromptSecurityError(f"intent must be one of {intent_allowlist}, got: {data['intent']}")
        
        # Validate steps array
        if not isinstance(data['steps'], list):
            raise PromptSecurityError("'steps' must be an array")
        
        if len(data['steps']) == 0:
            raise PromptSecurityError("'steps' cannot be empty")
        
        if len(data['steps']) > 20:
            raise PromptSecurityError("'steps' cannot exceed 20 (too complex)")
        
        # Validate each step
        action_allowlist = ['query_analytics', 'compare_results', 'generate_chart', 'format_response']
        seen_step_ids = set()
        
        for idx, step in enumerate(data['steps']):
            step_required = ['step_id', 'action', 'description', 'params', 'depends_on', 'critical']
            step_missing = [key for key in step_required if key not in step]
            if step_missing:
                raise PromptSecurityError(f"Step {idx} missing required keys: {step_missing}")
            
            # Validate step_id (must be positive integer, unique)
            try:
                step_id = int(step['step_id'])
                if step_id < 1:
                    raise PromptSecurityError(f"Step {idx}: step_id must be positive, got {step_id}")
                if step_id in seen_step_ids:
                    raise PromptSecurityError(f"Step {idx}: duplicate step_id {step_id}")
                seen_step_ids.add(step_id)
            except (ValueError, TypeError):
                raise PromptSecurityError(f"Step {idx}: step_id must be an integer, got {type(step['step_id'])}")
            
            # Validate action (strict allowlist)
            if step['action'] not in action_allowlist:
                raise PromptSecurityError(f"Step {idx}: action must be one of {action_allowlist}, got: {step['action']}")
            
            # Validate description (sanitize)
            description = str(step['description'])
            if len(description) > 500:
                raise PromptSecurityError(f"Step {idx}: description too long (max 500 chars)")
            step['description'] = self._sanitize_user_input(description)
            
            # Validate params (must be dict)
            if not isinstance(step['params'], dict):
                raise PromptSecurityError(f"Step {idx}: params must be a dict")
            
            # Validate depends_on (must be list of integers)
            if not isinstance(step['depends_on'], list):
                raise PromptSecurityError(f"Step {idx}: depends_on must be a list")
            for dep_id in step['depends_on']:
                try:
                    dep_int = int(dep_id)
                    if dep_int < 1:
                        raise PromptSecurityError(f"Step {idx}: depends_on contains invalid step_id {dep_int}")
                except (ValueError, TypeError):
                    raise PromptSecurityError(f"Step {idx}: depends_on must contain integers")
            
            # Validate critical (must be boolean)
            if not isinstance(step['critical'], bool):
                raise PromptSecurityError(f"Step {idx}: critical must be boolean")
        
        # Validate metadata (must be dict)
        if not isinstance(data['metadata'], dict):
            raise PromptSecurityError("'metadata' must be a dict")
        
        return True
    
    def _format_message(self, user_query: str, intent: str, query_type: str, comparison_targets: list) -> str:
        """
        Format planner user message with secure input handling.
        
        Uses structural isolation (build_user_section) to prevent prompt injection
        and semantic attacks. All user inputs are sanitized automatically.
        
        Args:
            user_query: Original user question (will be sanitized)
            intent: Extracted intent (will be validated against allowlist)
            query_type: Query classification (will be validated against allowlist)
            comparison_targets: List of targets to compare (will be sanitized)
            
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
        
        # Validate intent against allowlist
        intent_allowlist = ['success_rate', 'failure_rate']
        if intent not in intent_allowlist:
            raise PromptSecurityError(f"intent must be one of {intent_allowlist}, got: {intent}")
        
        # Validate query_type against allowlist
        query_type_allowlist = ['comparison', 'aggregation', 'trend']
        if query_type not in query_type_allowlist:
            raise PromptSecurityError(f"query_type must be one of {query_type_allowlist}, got: {query_type}")
        
        # Validate comparison_targets
        if comparison_targets is not None:
            if not isinstance(comparison_targets, list):
                raise PromptSecurityError("comparison_targets must be a list or None")
            if len(comparison_targets) > 50:
                raise PromptSecurityError("comparison_targets exceeds maximum count (50)")
        
        # Build user query section with structural isolation
        query_section = self.build_user_section(
            section_id="USER_QUERY",
            user_input=user_query,
            header="User Request",
            metadata={
                "source": "query_understanding_agent",
                "timestamp": "runtime",
                "sanitized": True
            }
        )
        
        # Build extracted parameters section with structural isolation
        # Note: intent and query_type are validated against allowlists, but we still sanitize for defense-in-depth
        targets_str = str(comparison_targets) if comparison_targets else "None"
        params_content = f"""Intent: {intent}
Query Type: {query_type}
Comparison Targets: {targets_str}"""
        
        params_section = self.build_user_section(
            section_id="EXTRACTED_PARAMS",
            user_input=params_content,
            header="Extracted Parameters",
            metadata={
                "intent": intent,
                "query_type": query_type,
                "targets_count": len(comparison_targets) if comparison_targets else 0
            }
        )
        
        # Construct final message with security boundaries
        return f"""Create an execution plan for this analytical query:

{query_section}

{params_section}

**Requirements**:
1. Create optimal plan with minimal steps
2. Enable parallel execution where possible (steps with no dependencies)
3. Include chart generation for visualization
4. End with natural language response formatting
5. Return valid JSON only

Generate the execution plan now:
"""
