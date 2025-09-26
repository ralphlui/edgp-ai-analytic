import boto3
import logging
from typing import Dict, Any, Optional
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

class DatabaseService:
    logger = logging.getLogger("database_service")

    async def get_success_rate_by_file_name(self, file_name: Optional[str] = None, org_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the percentage of success and fail records for a given file_id or file_name in the tracker table.

        If `file_id` is not provided, `file_name` will be used to look up the file id via `get_file_id_by_name`.
        If `org_id` is provided, results will be filtered to that organization.
        If start_date or end_date are provided, results will be filtered by created_date range.

        Returns a dict with chart data for visualization and includes the resolved `file_id` and `org_id` for debugging.
        """
        # Initialize file_id at the beginning, outside try block
        file_id = None
        
        try:
            # Resolve file_id from file_name when needed
            if file_name:
                file_id = await self.get_file_id_by_name(file_name)
                self.logger.info("Resolved file_name '%s' -> file_id '%s'", file_name, file_id)

            if not file_id:
                self.logger.error("Could not resolve file_id from file_name: %s", file_name)
                return {
                    "success": False,
                    "error": f"Could not resolve file_id from file_name: {file_name}",
                    "chart_data": [],
                    "row_count": 0,
                    "file_name": file_name,
                    "org_id": org_id
                }

            # Build filter expression
            filter_expr = boto3.dynamodb.conditions.Attr('file_id').eq(file_id)
            self.logger.info("Base filter: file_id=%s", file_id)
            
            # Always filter out records with empty/null organization_id
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').exists()
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').ne('')
            self.logger.info("Added filters to exclude empty/null organization_id")
            
            if org_id:
                # Log the org_id being used for filtering
                self.logger.info("Applying organization filter with org_id: %s", org_id)
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
                self.logger.debug("Complete filter: file_id=%s AND organization_id=%s (excluding empty orgs)", file_id, org_id)
            else:
                self.logger.warning("No org_id provided - but will still exclude records with empty organization_id")
            
            # Add date range filtering
            if start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
                self.logger.debug("Adding start_date filter: %s", start_date)
            if end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)
                self.logger.debug("Adding end_date filter: %s", end_date)

            self.logger.info("Executing DynamoDB scan with filters...")
            response = self.tracker_table.scan(
                FilterExpression=filter_expr
            )

            items = response.get('Items', [])
            total = len(items)
            self.logger.info("DynamoDB returned %d items for file_id=%s org_id=%s", total, file_id, org_id)
            
            # Log organization IDs of returned items to debug filtering
            if items:
                org_ids_found = []
                for item in items:
                    item_org_id = item.get('organization_id', 'NO_ORG_ID')
                    if item_org_id not in org_ids_found:
                        org_ids_found.append(item_org_id)
                self.logger.info("Organization IDs in results: %s", org_ids_found)
                
                if org_id and len(org_ids_found) > 1:
                    self.logger.error("ERROR: Multiple organizations found when filtering by org_id=%s: %s", org_id, org_ids_found)
            
            if total == 0:
                # Build descriptive message based on filters applied
                message_parts = ["No records found for this file"]
                if start_date and end_date:
                    if start_date == end_date:
                        message_parts.append(f"on {start_date}")
                    else:
                        message_parts.append(f"between {start_date} and {end_date}")
                elif start_date:
                    message_parts.append(f"from {start_date} onwards")
                elif end_date:
                    message_parts.append(f"until {end_date}")
                
                return {
                    "success": True,
                    "chart_data": [],
                    "row_count": 0,
                    "message": " ".join(message_parts) + ".",
                    "file_id": file_id,
                    "file_name": file_name,
                    "org_id": org_id,
                    "date_filter": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }

            success_count = 0
            fail_count = 0
            status_values = []
            unknown_status_count = 0
            
            for item in items:
                status = item.get('final_status', None)
                item_org_id = item.get('organization_id', 'NO_ORG_ID')
                
                if status is not None:
                    status_clean = str(status).strip().lower()
                    status_values.append(f"{status_clean}(org:{item_org_id})")
                    
                    if status_clean == 'success':
                        success_count += 1
                    elif status_clean == 'fail':
                        fail_count += 1
                    else:
                        unknown_status_count += 1
                        self.logger.warning("Unknown status found: '%s' for org_id=%s", status_clean, item_org_id)
                else:
                    unknown_status_count += 1
                    self.logger.warning("Item with missing final_status for org_id=%s", item_org_id)

            self.logger.info("Status breakdown: Success=%d, Fail=%d, Unknown=%d", success_count, fail_count, unknown_status_count)
            self.logger.info("Detailed status values: %s", status_values)
            
            # Calculate percentages
            if total > 0:
                success_rate = round((success_count / total) * 100, 2)
                fail_rate = round((fail_count / total) * 100, 2)
            else:
                success_rate = fail_rate = 0
            self.logger.info("Computed success=%s%% (%d) fail=%s%% (%d) out of total=%d for file_id=%s org_id=%s", 
                           success_rate, success_count, fail_rate, fail_count, total, file_id, org_id)
            chart_data = self.format_chart_data(success_rate, success_count, fail_rate, fail_count)

            return {
                "success": True,
                "chart_data": chart_data,
                "row_count": total,
                "chart_type": "bar",
                "file_id": file_id,
                "file_name": file_name,
                "org_id": org_id
            }
        except Exception as e:
            self.logger.exception("Error in get_success_rate_by_file_name for file_name: %s", file_name)
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "row_count": 0,
                "file_id": file_id,
                "file_name": file_name,
                "org_id": org_id
            }
        

    async def get_success_rate_by_domain_name(self, domain_name: Optional[str] = None, org_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the percentage of success and fail records for a given domain_name in the tracker table.

        If `org_id` is provided, results will be filtered to that organization.
        If start_date or end_date are provided, results will be filtered by created_date range.

        Returns a dict with chart data for visualization and includes the domain_name and org_id for debugging.
        """
        try:
            if not domain_name:
                self.logger.error("Domain name is required")
                return {
                    "success": False,
                    "error": "Domain name is required",
                    "chart_data": [],
                    "row_count": 0,
                    "domain_name": domain_name,
                    "org_id": org_id
                }

            # Build filter expression
            filter_expr = boto3.dynamodb.conditions.Attr('domain_name').eq(domain_name)
            self.logger.info("Base filter: domain_name=%s", domain_name)
            
            # Always filter out records with empty/null organization_id
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').exists()
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').ne('')
            self.logger.info("Added filters to exclude empty/null organization_id")
            
            if org_id:
                # Log the org_id being used for filtering
                self.logger.info("Applying organization filter with org_id: %s", org_id)
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
                self.logger.debug("Complete filter: domain_name=%s AND organization_id=%s (excluding empty orgs)", domain_name, org_id)
            else:
                self.logger.warning("No org_id provided - but will still exclude records with empty organization_id")
            
            # Add date range filtering
            if start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
                self.logger.debug("Adding start_date filter: %s", start_date)
            if end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)
                self.logger.debug("Adding end_date filter: %s", end_date)

            self.logger.info("Executing DynamoDB scan with filters...")
            response = self.tracker_table.scan(
                FilterExpression=filter_expr
            )

            items = response.get('Items', [])
            total = len(items)
            self.logger.info("DynamoDB returned %d items for domain_name=%s org_id=%s", total, domain_name, org_id)
            
            # Log organization IDs of returned items to debug filtering
            if items:
                org_ids_found = []
                for item in items:
                    item_org_id = item.get('organization_id', 'NO_ORG_ID')
                    if item_org_id not in org_ids_found:
                        org_ids_found.append(item_org_id)
                self.logger.info("Organization IDs in results: %s", org_ids_found)
                
                if org_id and len(org_ids_found) > 1:
                    self.logger.error("ERROR: Multiple organizations found when filtering by org_id=%s: %s", org_id, org_ids_found)
            
            if total == 0:
                # Build descriptive message based on filters applied
                message_parts = ["No records found for this domain"]
                if start_date and end_date:
                    if start_date == end_date:
                        message_parts.append(f"on {start_date}")
                    else:
                        message_parts.append(f"between {start_date} and {end_date}")
                elif start_date:
                    message_parts.append(f"from {start_date} onwards")
                elif end_date:
                    message_parts.append(f"until {end_date}")
                
                return {
                    "success": True,
                    "chart_data": [],
                    "row_count": 0,
                    "message": " ".join(message_parts) + ".",
                    "domain_name": domain_name,
                    "org_id": org_id,
                    "date_filter": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }

            success_count = 0
            fail_count = 0
            status_values = []
            unknown_status_count = 0
            
            for item in items:
                status = item.get('final_status', None)
                item_org_id = item.get('organization_id', 'NO_ORG_ID')
                
                if status is not None:
                    status_clean = str(status).strip().lower()
                    status_values.append(f"{status_clean}(org:{item_org_id})")
                    
                    if status_clean == 'success':
                        success_count += 1
                    elif status_clean == 'fail':
                        fail_count += 1
                    else:
                        unknown_status_count += 1
                        self.logger.warning("Unknown status found: '%s' for org_id=%s", status_clean, item_org_id)
                else:
                    unknown_status_count += 1
                    self.logger.warning("Item with missing final_status for org_id=%s", item_org_id)

            self.logger.info("Status breakdown: Success=%d, Fail=%d, Unknown=%d", success_count, fail_count, unknown_status_count)
            self.logger.info("Detailed status values: %s", status_values)
            
            # Calculate percentages
            if total > 0:
                success_rate = round((success_count / total) * 100, 2)
                fail_rate = round((fail_count / total) * 100, 2)
            else:
                success_rate = fail_rate = 0
            self.logger.info("Computed success=%s%% (%d) fail=%s%% (%d) out of total=%d for domain_name=%s org_id=%s", 
                           success_rate, success_count, fail_rate, fail_count, total, domain_name, org_id)
            chart_data = self.format_chart_data(success_rate, success_count, fail_rate, fail_count)

            return {
                "success": True,
                "chart_data": chart_data,
                "row_count": total,
                "chart_type": "bar",
                "domain_name": domain_name,
                "org_id": org_id
            }
        except Exception as e:
            self.logger.exception("Error in get_success_rate_by_domain_name for domain_name: %s", domain_name)
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "row_count": 0,
                "domain_name": domain_name,
                "org_id": org_id
            }


    async def get_file_id_by_name(self, file_name: str) -> Optional[str]:
        """
        Retrieve file_id from header table using file_name.
        Requires a GSI on file_name in the header table.
        """
        try:
            response = self.header_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('file_name').eq(file_name)
            )
            items = response.get('Items', [])
            if items:
                return items[0].get('id')
            return None
        except Exception as e:
            self.logger.exception("Error in get_file_id_by_name for file_name: %s", file_name)
            return None
        
    @staticmethod
    def format_chart_data(success_rate, success_count, fail_rate, fail_count):
        """
        Helper to format chart data for LLM chart creation.
        """
        return [
            {"status": "success", "percentage": success_rate, "count": success_count},
            {"status": "fail", "percentage": fail_rate, "count": fail_count}
        ]
        
    def _parse_boolean_field(self, value):
        """
        Parse a boolean field that might be stored as string or boolean.
        
        Args:
            value: The value to parse (can be boolean, string, or None)
            
        Returns:
            bool or None: True for true values, False for false values, None for invalid/missing values
        """
        if value is None:
            value = 'failed'
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            # Handle string representations
            lower_value = value.lower().strip()
            if lower_value in ['true', '1', 'yes', 'pass', 'passed']:
                return True
            elif lower_value in ['false', '0', 'no', 'fail', 'failed']:
                return False
            else:
                # Invalid string value
                return None
        
        # Handle numeric values (1 = True, 0 = False)
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            elif value == 0:
                return False
            else:
                return None
        
        # Unknown type
        return None
        
    async def get_rule_failure_rate(self, file_name: Optional[str] = None, org_id: Optional[str] = None,
                                  start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the failure rate percentage for rule validation (rule_status=false) for a specific file.
        
        Args:
            file_name: Name of the file to analyze
            org_id: Organization ID for filtering
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            
        Returns:
            Dict with failure rate percentage, counts, and chart data
        """
        return await self._calculate_failure_rate(file_name, org_id, start_date, end_date, "rule_status")
    
    async def get_data_quality_failure_rate(self, file_name: Optional[str] = None, org_id: Optional[str] = None,
                                          start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the failure rate percentage for data quality validation (dataquality_status=false) for a specific file.
        
        Args:
            file_name: Name of the file to analyze
            org_id: Organization ID for filtering
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            
        Returns:
            Dict with failure rate percentage, counts, and chart data
        """
        return await self._calculate_failure_rate(file_name, org_id, start_date, end_date, "dataquality_status")
    
    async def _calculate_failure_rate(self, file_name: Optional[str] = None, org_id: Optional[str] = None,
                                    start_date: Optional[str] = None, end_date: Optional[str] = None,
                                    status_field: str = "rule_status") -> Dict[str, Any]:
        """
        Internal method to calculate failure rate for a specific status field.
        
        Args:
            file_name: Name of the file to analyze
            org_id: Organization ID for filtering
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            status_field: The field to analyze ('rule_status' or 'dataquality_status')
            
        Returns:
            Dict with failure rate percentage, counts, and chart data
        """
        file_id = None
        
        try:
            # Resolve file_id from file_name
            if file_name:
                file_id = await self.get_file_id_by_name(file_name)
                self.logger.info("Resolved file_name '%s' -> file_id '%s'", file_name, file_id)

            if not file_id:
                self.logger.error("Could not resolve file_id from file_name: %s", file_name)
                return {
                    "success": False,
                    "error": f"Could not resolve file_id from file_name: {file_name}",
                    "failure_rate": 0,
                    "chart_data": [],
                    "total_records": 0,
                    "file_name": file_name,
                    "status_field": status_field
                }

            # Build filter expression
            filter_expr = boto3.dynamodb.conditions.Attr('file_id').eq(file_id)
            
            # Always filter out records with empty/null organization_id
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').exists()
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').ne('')
            
            # Add org_id filter if provided
            if org_id:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
                self.logger.info("Added org_id filter: %s", org_id)

            # Add date filtering if provided
            if start_date and end_date:
                if start_date == end_date:
                    filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').begins_with(start_date)
                else:
                    filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').between(start_date, end_date)
            elif start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
            elif end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)

            # Query DynamoDB using the existing tracker table connection
            response = self.tracker_table.scan(FilterExpression=filter_expr)

            items = response.get('Items', [])
            total = len(items)
            self.logger.info("DynamoDB returned %d items for %s failure rate analysis", total, status_field)

            if total == 0:
                return {
                    "success": True,
                    "failure_rate": 0,
                    "passed_count": 0,
                    "failed_count": 0,
                    "total_records": 0,
                    "chart_data": [],
                    "file_name": file_name,
                    "status_field": status_field,
                    "message": f"No records found for {status_field} analysis."
                }

            # Count passed and failed records
            passed_count = 0
            failed_count = 0
            
            for item in items:
                status_value = item.get(status_field)
                
                if status_value == "success":
                    passed_count += 1
                else:
                    failed_count += 1
                # Skip records where the status field is None/missing or has other values

            # Calculate failure rate
            total_with_status = passed_count + failed_count
            if total_with_status > 0:
                failure_rate = round((failed_count / total_with_status) * 100, 2)
                success_rate = round((passed_count / total_with_status) * 100, 2)
            else:
                failure_rate = 0
                success_rate = 0

            # Create chart data in the expected format
            chart_data = []
            if success_rate > 0:
                chart_data.append({
                    "status": "success",
                    "count": passed_count,
                    "percentage": success_rate
                })
            if failure_rate > 0:
                chart_data.append({
                    "status": "fail",
                    "count": failed_count,
                    "percentage": failure_rate
                })

            self.logger.info("%s analysis: %d/%d records failed (%.2f%% failure rate)", 
                           status_field, failed_count, total_with_status, failure_rate)

            return {
                "success": True,
                "failure_rate": failure_rate,
                "success_rate": success_rate,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "total_records": total,
                "records_with_status": total_with_status,
                "chart_data": chart_data,
                "chart_type": "bar",
                "file_id": file_id,
                "file_name": file_name,
                "org_id": org_id,
                "status_field": status_field,
                "can_generate_report": True,
                "report_summary": {
                    "failure_percentage": failure_rate,
                    "success_percentage": success_rate,
                    "analysis_focus": f"{status_field.replace('_', ' ').title()} Failure Rate"
                }
            }

        except Exception as e:
            self.logger.exception("Error in _calculate_failure_rate for file_name: %s, status_field: %s", file_name, status_field)
            return {
                "success": False,
                "error": str(e),
                "failure_rate": 0,
                "chart_data": [],
                "total_records": 0,
                "file_id": file_id,
                "file_name": file_name,
                "status_field": status_field
            }

    async def get_domain_analytics_by_field(self, domain_name: str, group_by_field: str, org_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics for any domain grouped by any field from tracker table.
        
        This is a flexible method that can analyze any domain (customer, product, order, etc.)
        and group results by any field (country, category, status, region, etc.).
        
        Args:
            domain_name: Domain to analyze (e.g., 'customer', 'product', 'vendor', 'location')
            group_by_field: Field to group by (e.g., 'country', 'category', 'region')  
            org_id: Organization ID for filtering
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            
        Returns:
            Dict with analytics data grouped by the specified field
        """
        try:
            # Build filter expression for the specified domain
            filter_expr = boto3.dynamodb.conditions.Attr('domain_name').eq(domain_name)
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
            
            self.logger.info("Querying %s analytics grouped by %s for org_id: %s", domain_name, group_by_field, org_id)
            
            # Add date filtering if provided
            if start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
                self.logger.debug("Adding start_date filter: %s", start_date)
            if end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)
                self.logger.debug("Adding end_date filter: %s", end_date)

            # Query DynamoDB
            response = self.tracker_table.scan(FilterExpression=filter_expr)
            items = response.get('Items', [])
            
            self.logger.info("Found %d %s records", len(items), domain_name)
            
            if len(items) == 0:
                # Build descriptive message based on filters applied
                message_parts = [f"No {domain_name} records found"]
                if start_date and end_date:
                    if start_date == end_date:
                        message_parts.append(f"on {start_date}")
                    else:
                        message_parts.append(f"between {start_date} and {end_date}")
                elif start_date:
                    message_parts.append(f"from {start_date} onwards")
                elif end_date:
                    message_parts.append(f"until {end_date}")
                
                return {
                    "success": True,
                    "chart_data": [],
                    "total_records": 0,
                    "groups_found": 0,
                    "domain_name": domain_name,
                    "group_by_field": group_by_field,
                    "message": " ".join(message_parts) + ".",
                    "org_id": org_id,
                    "date_filter": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
            
            # Count records by the specified grouping field
            field_counts = {}
            total_records = 0
            
            # Define possible field name variations to try
            field_variations = [
                group_by_field,  # exact match
                f"{domain_name}_{group_by_field}",  # customer_country, product_category
                f"{group_by_field}_name",  # country_name, category_name
                f"{group_by_field}s",  # countries, categories (plural)
                group_by_field.replace('_', ''),  # remove underscores
                group_by_field.replace('-', '_'),  # replace dashes with underscores
            ]
            
            for item in items:
                # Try different possible field names to find the grouping value
                group_value = None
                field_used = None
                
                for field_variant in field_variations:
                    if field_variant in item and item[field_variant]:
                        group_value = str(item[field_variant]).strip()
                        field_used = field_variant
                        break
                
                if not group_value:
                    # If no grouping field found, use "Unknown"
                    group_value = "Unknown"
                    field_used = "default"
                
                field_counts[group_value] = field_counts.get(group_value, 0) + 1
                total_records += 1
            
            # Convert to chart data format
            chart_data = []
            for group_name, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = round((count / total_records) * 100, 2) if total_records > 0 else 0
                chart_data.append({
                    group_by_field: group_name,  # e.g., "country": "Singapore"
                    f"{domain_name}_count": count,  # e.g., "customer_count": 5
                    "percentage": percentage
                })
            
            self.logger.info("Analytics complete: %d %s records grouped into %d %s groups", 
                           total_records, domain_name, len(field_counts), group_by_field)
            
            return {
                "success": True,
                "chart_data": chart_data,
                "total_records": total_records,
                "groups_found": len(field_counts),
                "domain_name": domain_name,
                "group_by_field": group_by_field,
                "chart_type": "bar",
                "org_id": org_id,
                "date_filter": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "field_mapping": {
                    "attempted_fields": field_variations,
                    "successful_matches": len([item for item in items if any(field in item for field in field_variations)])
                }
            }
            
        except Exception as e:
            self.logger.exception("Error getting %s analytics by %s", domain_name, group_by_field)
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "total_records": 0,
                "domain_name": domain_name,
                "group_by_field": group_by_field,
                "org_id": org_id
            }
        
    def __init__(self, tracker_table_name: str = "MasterDataTaskTrackerSIT", header_table_name: str = "MasterDataHeaderSIT"):
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        self.tracker_table = self.dynamodb.Table(tracker_table_name)
        self.header_table = self.dynamodb.Table(header_table_name)

# Initialize database service
db_service = DatabaseService()    