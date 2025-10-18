"""
Chart generation service for analytics visualizations.

Generates base64-encoded charts from analytics data using matplotlib.
"""
import logging
import base64
from io import BytesIO
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

logger = logging.getLogger("analytic_agent")


class AnalyticsChartGenerator:
    """
    Generate charts from analytics data.
    
    All charts are returned as base64-encoded PNG images for easy embedding
    in web responses or API payloads.
    """
    
    def __init__(self):
        """Initialize chart generator with default settings."""
        self.figure_size = (10, 6)
        self.dpi = 100
        self.color_success = '#10b981'  # Green
        self.color_failure = '#ef4444'  # Red
        self.color_neutral = '#6b7280'  # Gray
    
    def generate_success_failure_bar_chart(
        self,
        data: dict,
        chart_type: str
    ) -> Optional[str]:
        """
        Generate a bar chart showing success vs failure counts.
        
        Creates a horizontal or vertical bar chart comparing successful
        and failed requests with the success/failure rate prominently displayed.
        
        Args:
            data: Analytics data with keys:
                - target_type: "domain" or "file"
                - target_value: actual domain/file name
                - total_requests: total number of requests
                - successful_requests: number of successful requests
                - failed_requests: number of failed requests
                - success_rate or failure_rate: percentage
            chart_type: "success_rate" or "failure_rate" (for title)
        
        Returns:
            Base64-encoded PNG image string, or None if data is invalid
        
        Example:
            >>> generator = AnalyticsChartGenerator()
            >>> data = {
            ...     "target_type": "domain",
            ...     "target_value": "customer",
            ...     "total_requests": 1000,
            ...     "successful_requests": 950,
            ...     "failed_requests": 50,
            ...     "success_rate": 95.0
            ... }
            >>> base64_image = generator.generate_success_failure_bar_chart(data)
        """
        try:
            # Extract data
            target_type = data.get("target_type", "target")
            target_value = data.get("target_value", "Unknown")
            total_requests = data.get("total_requests", 0)
            successful_requests = data.get("successful_requests", 0)
            failed_requests = data.get("failed_requests", 0)
            
            # Get the appropriate rate
            if chart_type == "success_rate":
                rate = data.get("success_rate", 0)
                rate_label = "Success Rate"
            else:
                rate = data.get("failure_rate", 0)
                rate_label = "Failure Rate"
            
            # Validate data
            if total_requests == 0:
                logger.warning("No data to chart (total_requests = 0)")
                return None
            
            logger.info(f"Generating bar chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Filter chart data based on chart_type
            if chart_type == "success_rate":
                # Show only successful requests
                categories = ['Successful']
                values = [successful_requests]
                colors = [self.color_success]
                logger.info(f"Showing success-only chart: {successful_requests} successful requests")
            elif chart_type == "failure_rate":
                # Show only failed requests
                categories = ['Failed']
                values = [failed_requests]
                colors = [self.color_failure]
                logger.info(f"Showing failure-only chart: {failed_requests} failed requests")
            else:
                # Show both (default)
                categories = ['Successful', 'Failed']
                values = [successful_requests, failed_requests]
                colors = [self.color_success, self.color_failure]
                logger.info(f"Showing combined chart: {successful_requests} success + {failed_requests} failed")
            
            # Create vertical bar chart
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{value:,}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold'
                )
            
            # Add percentage labels inside bars (if tall enough)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if height > (max(values) * 0.1):  # Only if bar is tall enough
                    percentage = (value / total_requests) * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,
                        f'{percentage:.1f}%',
                        ha='center',
                        va='center',
                        fontsize=11,
                        color='white',
                        fontweight='bold'
                    )
            
            # Customize chart
            ax.set_ylabel('Number of Requests', fontsize=12, fontweight='bold')
            ax.set_title(
                f'{rate_label} for {target_type.title()} "{target_value}"\n'
                f'{rate:.2f}% | Total: {total_requests:,} requests',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            # Add grid for better readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
            # Format y-axis to show comma-separated numbers
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            
            # Add a horizontal line for total requests reference
            ax.axhline(y=total_requests, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(
                1.02, total_requests, f'Total: {total_requests:,}',
                transform=ax.get_yaxis_transform(),
                fontsize=9,
                va='center',
                color='gray'
            )
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Clean up
            buffer.close()
            plt.close(fig)
            
            logger.info(f"Chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"Error generating chart: {e}")
            plt.close('all')  # Clean up any open figures
            return None
    
    def generate_pie_chart(self, data: dict) -> Optional[str]:
        """
        Generate a pie chart showing success vs failure distribution.
        
        Args:
            data: Analytics data (same format as bar chart)
        
        Returns:
            Base64-encoded PNG image string, or None if data is invalid
        """
        try:
            # Extract data
            target_type = data.get("target_type", "target")
            target_value = data.get("target_value", "Unknown")
            total_requests = data.get("total_requests", 0)
            successful_requests = data.get("successful_requests", 0)
            failed_requests = data.get("failed_requests", 0)
            success_rate = data.get("success_rate", 0)
            
            # Validate data
            if total_requests == 0:
                logger.warning("No data to chart (total_requests = 0)")
                return None
            
            logger.info(f"Generating pie chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
            
            # Data for pie chart
            sizes = [successful_requests, failed_requests]
            labels = [f'Successful\n{successful_requests:,}', f'Failed\n{failed_requests:,}']
            colors = [self.color_success, self.color_failure]
            explode = (0.05, 0.05)  # Slightly separate both slices
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # Make percentage text white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_fontweight('bold')
            
            # Equal aspect ratio ensures circular pie
            ax.axis('equal')
            
            # Title
            plt.title(
                f'Success/Failure Distribution\n{target_type.title()}: "{target_value}"\n'
                f'Total Requests: {total_requests:,}',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Clean up
            buffer.close()
            plt.close(fig)
            
            logger.info(f"Pie chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"Error generating pie chart: {e}")
            plt.close('all')
            return None


# Convenience function
def generate_analytics_chart(data: dict, chart_type: str = "success_rate", style: str = "bar") -> Optional[str]:
    """
    Generate analytics chart with specified style.
    
    Args:
        data: Analytics data dictionary
        chart_type: "success_rate" or "failure_rate"
        style: "bar" or "pie"
    
    Returns:
        Base64-encoded PNG image string
    
    Example:
        >>> chart = generate_analytics_chart(data, chart_type="success_rate", style="bar")
        >>> print(f"data:image/png;base64,{chart}")
    """
    generator = AnalyticsChartGenerator()
    
    if style == "pie":
        return generator.generate_pie_chart(data)
    else:
        return generator.generate_success_failure_bar_chart(data, chart_type)


def generate_comparison_chart(comparison_data: dict) -> Optional[str]:
    """
    Generate a comparison chart for multiple targets.
    
    Creates a side-by-side bar chart comparing metrics across multiple targets.
    
    Args:
        comparison_data: Comparison data with structure:
            {
                "targets": ["customer.csv", "payment.csv"],
                "metric": "success_rate" or "failure_rate",
                "winner": "customer.csv",
                "comparison_details": [
                    {
                        "target": "customer.csv",
                        "metric_value": 95.5,
                        "total_requests": 1000,
                        "successful_requests": 955,
                        "failed_requests": 45
                    },
                    ...
                ]
            }
    
    Returns:
        Base64-encoded PNG image string, or None if data is invalid
    
    Example:
        >>> chart = generate_comparison_chart(comparison_data)
        >>> print(f"Chart size: {len(chart)} bytes")
    """
    try:
        # Extract data
        targets = comparison_data.get("targets", [])
        metric = comparison_data.get("metric", "success_rate")
        winner = comparison_data.get("winner")
        details = comparison_data.get("comparison_details", [])
        
        if not targets or not details:
            logger.warning("No comparison data to chart")
            return None
        
        logger.info(f"Generating comparison chart for {len(targets)} targets")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Extract data for chart
        target_names = [d["target"] for d in details]
        metric_values = [d["metric_value"] for d in details]
        
        # Set colors - winner gets green, others get blue
        colors = ['#10b981' if d["target"] == winner else '#3b82f6' for d in details]
        
        # Create bar chart
        bars = ax.bar(target_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar, value, detail in zip(bars, metric_values, details):
            height = bar.get_height()
            
            # Add percentage on top
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{value:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
            
            # Add request counts inside bar (if tall enough)
            if height > 10:
                total = detail["total_requests"]
                successful = detail["successful_requests"]
                failed = detail["failed_requests"]
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    f'Total: {total:,}\nSuccess: {successful:,}\nFailed: {failed:,}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='white',
                    fontweight='bold'
                )
        
        # Add winner indicator
        if winner:
            winner_idx = target_names.index(winner)
            ax.text(
                winner_idx,
                metric_values[winner_idx] + 5,
                '🏆 Winner',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#10b981'
            )
        
        # Customize chart
        metric_label = metric.replace('_', ' ').title()
        ax.set_ylabel(f'{metric_label} (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Targets', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{metric_label} Comparison\n'
            f'Comparing {len(targets)} targets',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Set y-axis range from 0 to 100 (percentage)
        ax.set_ylim(0, max(metric_values) * 1.2)
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Rotate x-axis labels if many targets
        if len(targets) > 3:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Clean up
        buffer.close()
        plt.close(fig)
        
        logger.info(f"Comparison chart generated successfully ({len(image_base64)} bytes)")
        
        return image_base64
        
    except Exception as e:
        logger.exception(f"Error generating comparison chart: {e}")
        plt.close('all')
        return None
