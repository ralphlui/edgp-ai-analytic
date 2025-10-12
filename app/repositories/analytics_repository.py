"""
Analytics Repository for querying analytics data from DynamoDB.

This repository handles all database queries for success/failure rate analytics
by domain or file name.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import boto3
from boto3.dynamodb.conditions import Key, Attr
from app.config import DYNAMODB_TRACKER_TABLE_NAME

logger = logging.getLogger("analytic_agent")


class AnalyticsRepository:
    """
    Repository for analytics data queries from DynamoDB.
    
    Table Schema:
    - Partition Key: domain_name or file_name (depending on GSI)
    - Sort Key: timestamp
    - Attributes: status ('success' | 'failure'), event_id, user_id, etc.
    
    Required GSIs:
    - domain_name-timestamp-index: For domain-based queries
    - file_name-timestamp-index: For file-based queries
    """
    
    def __init__(self, table_name: str = "analytics_events"):
        """
        Initialize the analytics repository.
        
        Args:
            table_name: Name of the DynamoDB table
        """
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)
        self.table_name = table_name
        logger.info(f"📊 Initialized AnalyticsRepository with table: {table_name}")
    
    def get_success_rate_by_domain(
        self,
        domain_name: str
    ) -> Dict[str, Any]:
        """
        Calculate success rate for a specific domain.
        
        Args:
            domain_name: The domain to analyze (e.g., "customer", "product", "location", "vendor")
        
        Returns:
            Dictionary containing:
            - total_requests: Total number of requests
            - successful_requests: Number of successful requests
            - failed_requests: Number of failed requests
            - success_rate: Percentage of successful requests (0-100)
            - target_type: "domain"
            - target_value: The domain name
        """
        logger.info(f"📈 Querying success rate for domain: {domain_name}")
        
        
        # Query DynamoDB using GSI
        items = self._query_by_domain(domain_name)
        
        # Calculate metrics
        metrics = self._calculate_metrics(items)
        
        result = {
            "total_requests": metrics['total'],
            "successful_requests": metrics['successful'],
            "failed_requests": metrics['failed'],
            "success_rate": metrics['success_rate'],
            "target_type": "domain",
            "target_value": domain_name
        }
        
        logger.info(
            f"✅ Domain query complete - Total: {metrics['total']}, "
            f"Success Rate: {metrics['success_rate']}%"
        )
        
        return result
    
    def get_success_rate_by_file(
        self,
        file_name: str,
    ) -> Dict[str, Any]:
        """
        Calculate success rate for a specific file.
        
        Args:
            file_name: The file to analyze (e.g., "customer.csv")
        Returns:
            Dictionary containing analytics metrics (same structure as domain query)
        """
        logger.info(f"📈 Querying success rate for file: {file_name}")
        
        
        # Query DynamoDB using GSI
        items = self._query_by_file(file_name)

        # Calculate metrics
        metrics = self._calculate_metrics(items)
        
        result = {
            "total_requests": metrics['total'],
            "successful_requests": metrics['successful'],
            "failed_requests": metrics['failed'],
            "success_rate": metrics['success_rate'],
            "target_type": "file",
            "target_value": file_name
        }
        
        logger.info(
            f"✅ File query complete - Total: {metrics['total']}, "
            f"Success Rate: {metrics['success_rate']}%"
        )
        
        return result
    
    def get_failure_rate_by_domain(
        self,
        domain_name: str
    ) -> Dict[str, Any]:
        """
        Calculate failure rate for a specific domain.
        
        Args:
            domain_name: The domain to analyze
        Returns:
            Dictionary containing failure rate metrics
        """
        logger.info(f"📉 Querying failure rate for domain: {domain_name}")
        
        # Get success rate data first
        success_data = self.get_success_rate_by_domain(domain_name)
        
        # Calculate failure rate (inverse of success rate)
        failure_rate = 100.0 - success_data["success_rate"]
        
        result = {
            "total_requests": success_data["total_requests"],
            "successful_requests": success_data["successful_requests"],
            "failed_requests": success_data["failed_requests"],
            "failure_rate": round(failure_rate, 2),
            "target_type": "domain",
            "target_value": domain_name
        }
        
        logger.info(f"✅ Domain failure rate: {failure_rate}%")
        
        return result
    
    def get_failure_rate_by_file(
        self,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Calculate failure rate for a specific file.
        
        Args:
            file_name: The file to analyze
        
        Returns:
            Dictionary containing failure rate metrics
        """
        logger.info(f"📉 Querying failure rate for file: {file_name}")
        
        # Get success rate data first
        success_data = self.get_success_rate_by_file(file_name)
        
        # Calculate failure rate
        failure_rate = 100.0 - success_data["success_rate"]
        
        result = {
            "total_requests": success_data["total_requests"],
            "successful_requests": success_data["successful_requests"],
            "failed_requests": success_data["failed_requests"],
            "failure_rate": round(failure_rate, 2),
            "target_type": "file",
            "target_value": file_name
        }
        
        logger.info(f"✅ File failure rate: {failure_rate}%")
        
        return result
    
    # Private helper methods
    
    def _query_by_domain(
        self,
        domain_name: str
    ) -> List[Dict[str, Any]]:
        """
        Scan DynamoDB for events by domain name (no GSI required).
        
        Args:
            domain_name: Domain to query
        Returns:
            List of items from DynamoDB
        """  
        logger.info(f"🔍 Scanning DynamoDB table '{self.table_name}' for domain: {domain_name}")
        items = []
        
        try:
            # Use scan with filter expression (no GSI required)
            logger.info(f"📊 Using SCAN with FilterExpression (no GSI)")
            response = self.table.scan(
                FilterExpression=Attr('domain_name').eq(domain_name)
            )
            
            items.extend(response.get('Items', []))
            logger.info(f"📦 First page retrieved: {len(items)} items")
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                logger.info(f"📄 Fetching next page (current items: {len(items)})")
                response = self.table.scan(
                    FilterExpression=Attr('domain_name').eq(domain_name),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            logger.info(f"✅ Retrieved {len(items)} total items from DynamoDB")
            
            # Debug: Log first item if available
            if items:
                first_item = items[0]
                logger.info(f"🔍 Sample item fields: {list(first_item.keys())}")
                logger.info(f"🔍 Sample item final_status: {first_item.get('final_status')}")
            else:
                logger.warning(f"⚠️ No items found for domain: {domain_name}")
            
        except Exception as e:
            logger.exception(f"❌ Error scanning DynamoDB: {e}")
            logger.error(f"❌ Table: {self.table_name}, Domain: {domain_name}")
            # Return empty list on error
            return []
        
        return items
    
    def _query_by_file(
        self,
        file_name: str
    ) -> List[Dict[str, Any]]:
        """
        Scan DynamoDB for events by file name (no GSI required).
        
        Args:
            file_name: File to query
        Returns:
            List of items from DynamoDB
        """
        logger.info(f"🔍 Scanning DynamoDB table '{self.table_name}' for file: {file_name}")

        items = []
        
        try:
            # Use scan with filter expression (no GSI required)
            logger.info(f"📊 Using SCAN with FilterExpression (no GSI)")
            response = self.table.scan(
                FilterExpression=Attr('file_name').eq(file_name)
            )
            
            items.extend(response.get('Items', []))
            logger.info(f"📦 First page retrieved: {len(items)} items")
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                logger.info(f"📄 Fetching next page (current items: {len(items)})")
                response = self.table.scan(
                    FilterExpression=Attr('file_name').eq(file_name),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            logger.info(f"✅ Retrieved {len(items)} total items from DynamoDB")
            
            # Debug: Log first item if available
            if items:
                first_item = items[0]
                logger.info(f"🔍 Sample item fields: {list(first_item.keys())}")
                logger.info(f"🔍 Sample item final_status: {first_item.get('final_status')}")
            else:
                logger.warning(f"⚠️ No items found for file: {file_name}")
            
        except Exception as e:
            logger.exception(f"❌ Error scanning DynamoDB: {e}")
            logger.error(f"❌ Table: {self.table_name}, File: {file_name}")
            # Return empty list on error
            return []
        
        return items
    
    def _calculate_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from items.
        
        Args:
            items: List of DynamoDB items
        
        Returns:
            Dictionary with total, successful, failed counts and success_rate
        """
        total_requests = len(items)
        
        # Debug: Log what statuses we're seeing
        if items:
            statuses = [item.get('final_status') for item in items[:5]]  # First 5
            logger.info(f"🔍 Sample final_status values: {statuses}")
        
        # Use final_status field (not status)
        successful_requests = sum(1 for item in items if item.get('final_status') == 'success')
        failed_requests = total_requests - successful_requests
        
        logger.info(f"📊 Metrics - Total: {total_requests}, Success: {successful_requests}, Failed: {failed_requests}")
        
        # Calculate success rate
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'total': total_requests,
            'successful': successful_requests,
            'failed': failed_requests,
            'success_rate': round(success_rate, 2)
        }
    
    def _build_time_series(
        self,
        items: List[Dict[str, Any]],
        metric_type: str = 'success_rate'
    ) -> List[Dict[str, Any]]:
        """
        Build time series data for charting.
        
        Args:
            items: List of DynamoDB items
            metric_type: Type of metric ('success_rate' or 'failure_rate')
        
        Returns:
            List of daily statistics [{date, success_rate, total_requests}]
        """
        # Group by date
        daily_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        for item in items:
            # Extract date from timestamp (YYYY-MM-DD)
            timestamp = item.get('timestamp', '')
            date = timestamp[:10] if timestamp else None
            
            if not date:
                continue
            
            # Use final_status field (not status)
            final_status = item.get('final_status')
            if final_status == 'success':
                daily_stats[date]['success'] += 1
            elif final_status == 'failure':
                daily_stats[date]['failure'] += 1
        
        # Build time series list
        time_series = []
        for date in sorted(daily_stats.keys()):
            stats = daily_stats[date]
            total = stats['success'] + stats['failure']
            
            if total > 0:
                success_rate = (stats['success'] / total * 100)
            else:
                success_rate = 0.0
            
            time_series.append({
                "date": date,
                "success_rate": round(success_rate, 2),
                "total_requests": total
            })
        
        return time_series


    def debug_scan_sample(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        DEBUG METHOD: Scan table and return sample items to inspect structure.
        
        Args:
            limit: Number of items to return
        
        Returns:
            List of sample items
        """
        logger.info(f"🔍 DEBUG: Scanning table '{self.table_name}' for sample items...")
        
        try:
            response = self.table.scan(Limit=limit)
            items = response.get('Items', [])
            
            logger.info(f"📦 Found {len(items)} sample items")
            
            if items:
                first_item = items[0]
                logger.info(f"🔍 Sample item fields: {list(first_item.keys())}")
                logger.info(f"🔍 Sample domain_name values: {[item.get('domain_name') for item in items]}")
                logger.info(f"🔍 Sample final_status values: {[item.get('final_status') for item in items]}")
                logger.info(f"🔍 Sample status values (if exists): {[item.get('status') for item in items]}")
            
            return items
            
        except Exception as e:
            logger.exception(f"❌ Error scanning table: {e}")
            return []


def get_analytics_repository(table_name: str = None) -> AnalyticsRepository:
    """
    Factory function to get an analytics repository instance.
    
    Args:
        table_name: Name of the DynamoDB table. If None, uses DYNAMODB_TRACKER_TABLE_NAME from config.
    
    Returns:
        AnalyticsRepository instance
    """
    if table_name is None:
        table_name = DYNAMODB_TRACKER_TABLE_NAME
        logger.info(f"📋 Using table from config: {table_name}")
    
    return AnalyticsRepository(table_name)
