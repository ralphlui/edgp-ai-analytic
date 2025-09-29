"""
Message formatting utilities for consistent error and response formatting.
"""
from typing import Dict, Any, List
from .sanitization import sanitize_text_input, sanitize_filename, sanitize_numeric_value


def format_error_message(error_type: str, user_message: str, technical_details: str = "") -> str:
    """Format error messages with consistent structure and user-friendly language."""
    formatted = f"❌ **{error_type}**\n\n{user_message}"
    if technical_details:
        formatted += f"\n\n🔧 *Technical Details:* {technical_details}"
    formatted += "\n\n💡 *Suggestion:* Try rephrasing your query or contact support if the issue persists."
    return formatted


def format_basic_message(chart_data: List[Dict], file_name: str, row_count: int,
                        chart_type: str, report_type: str, date_filter_used: Dict[str, str],
                        original_chart_data: List[Dict] = None) -> str:
    """Enhanced fallback message formatter with intelligent insights."""
    
    # Debug logging to help troubleshoot
    import logging
    logger = logging.getLogger(__name__)

    # Use original_chart_data to understand the full picture
    all_data = original_chart_data if original_chart_data else chart_data

    if not chart_data:
        # Check if we have data in the original that matches our report type
        if all_data and report_type != "both":
            # Count successes and failures in original data
            success_count = sum(1 for item in all_data if item.get('status', '').lower() in ['success', 'passed', 'pass', 'ok', 'successful'])
            failure_count = sum(1 for item in all_data if item.get('status', '').lower() in ['fail', 'failure', 'failed', 'error', 'issue', 'problem'])
            total_count = len(all_data)
            
            if report_type == "failure" and failure_count > 0:
                failure_rate = (failure_count / total_count) * 100 if total_count > 0 else 0
                base_msg = f"Failure analysis complete: {failure_count} out of {total_count} records failed ({failure_rate:.1f}% failure rate)"
                if file_name:
                    base_msg += f" in file: {sanitize_filename(file_name)}"
                return base_msg
            elif report_type == "success" and success_count > 0:
                success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
                base_msg = f"Success analysis complete: {success_count} out of {total_count} records succeeded ({success_rate:.1f}% success rate)"
                if file_name:
                    base_msg += f" in file: {sanitize_filename(file_name)}"
                return base_msg
        elif all_data and report_type == "both":
            # For "both" report type, check if we have any data at all
            success_count = sum(1 for item in all_data if item.get('status', '').lower() in ['success', 'passed', 'pass', 'ok', 'successful'])
            failure_count = sum(1 for item in all_data if item.get('status', '').lower() in ['fail', 'failure', 'failed', 'error', 'issue', 'problem'])
            total_count = len(all_data)
            
            if total_count > 0:
                success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
                failure_rate = (failure_count / total_count) * 100 if total_count > 0 else 0
                base_msg = f"Analysis complete: {success_count} successful, {failure_count} failed out of {total_count} total records ({success_rate:.1f}% success, {failure_rate:.1f}% failure)"
                if file_name:
                    base_msg += f" in file: {sanitize_filename(file_name)}"
                return base_msg
        
        # Fallback for when no relevant data is found
        base_msg = "I'm an analytics assistant focused on data analysis, reporting, and visualizations. I'm not able to help with this specific request, but I'd be happy to assist with any data-related questions you have!"
        return base_msg

    success_data = next((item for item in chart_data if item.get('status', '').lower() == 'success'), None)
    fail_data = next((item for item in chart_data if item.get('status', '').lower() == 'fail'), None)

    message_parts = []

    # File and date context
    if file_name:
        message_parts.append(f"Analysis of {sanitize_filename(file_name)} complete.")
    if date_filter_used:
        if date_filter_used.get("start_date") and date_filter_used.get("end_date"):
            if date_filter_used["start_date"] == date_filter_used["end_date"]:
                message_parts.append(f"Data from {sanitize_text_input(date_filter_used['start_date'], 20)}.")
            else:
                message_parts.append(f"Data from {sanitize_text_input(date_filter_used['start_date'], 20)} to {sanitize_text_input(date_filter_used['end_date'], 20)}.")
    if row_count > 0:
        message_parts.append(f"Processed {sanitize_numeric_value(row_count)} records.")

    # Intelligent insights based on data patterns
    if report_type == "success" and success_data:
        percentage = success_data['percentage']
        if percentage >= 95:
            message_parts.append(f"🎉 Exceptional success rate: {sanitize_numeric_value(percentage)}% - outstanding performance!")
        elif percentage >= 90:
            message_parts.append(f"✅ Excellent success rate: {sanitize_numeric_value(percentage)}% - very good results.")
        elif percentage >= 80:
            message_parts.append(f"👍 Good success rate: {sanitize_numeric_value(percentage)}% - solid performance.")
        else:
            message_parts.append(f"⚠️ Success rate: {sanitize_numeric_value(percentage)}% - room for improvement.")

    elif report_type == "failure" and fail_data:
        percentage = fail_data['percentage']
        if percentage == 0:
            message_parts.append(f"🎉 Perfect! Zero failure rate - all records processed successfully.")
        elif percentage <= 5:
            message_parts.append(f"✅ Excellent! Very low failure rate: {sanitize_numeric_value(percentage)}%.")
        elif percentage <= 10:
            message_parts.append(f"⚠️ Moderate failure rate: {sanitize_numeric_value(percentage)}% - worth investigating.")
        else:
            message_parts.append(f"🚨 High failure rate detected: {sanitize_numeric_value(percentage)}% - requires attention.")

    elif report_type == "both":
        if success_data and fail_data:
            success_pct = success_data['percentage']
            fail_pct = fail_data['percentage']

            # Overall assessment
            if fail_pct == 0:
                message_parts.append(f"🎉 Perfect performance! 100% success rate with zero failures.")
            elif success_pct >= 90:
                message_parts.append(f"✅ Strong performance: {sanitize_numeric_value(success_pct)}% success, {sanitize_numeric_value(fail_pct)}% failure.")
            elif success_pct >= 80:
                message_parts.append(f"👍 Good performance: {sanitize_numeric_value(success_pct)}% success, {sanitize_numeric_value(fail_pct)}% failure.")
            else:
                message_parts.append(f"⚠️ Needs improvement: {sanitize_numeric_value(success_pct)}% success, {sanitize_numeric_value(fail_pct)}% failure.")

    # Chart information
    chart_descriptions = {
        "bar": "bar chart",
        "pie": "pie chart",
        "donut": "donut chart",
        "line": "line chart",
        "stacked": "stacked bar chart"
    }
    chart_desc = chart_descriptions.get(chart_type, f"{sanitize_text_input(chart_type, 20)} chart")
    message_parts.append(f"Generated {sanitize_text_input(report_type, 20)}-focused {chart_desc} for visualization.")

    return " ".join(message_parts)