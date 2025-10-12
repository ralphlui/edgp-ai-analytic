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
        chart_type: str = "success_rate"
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
                logger.warning("⚠️ No data to chart (total_requests = 0)")
                return None
            
            logger.info(f"📊 Generating bar chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Data for chart
            categories = ['Successful', 'Failed']
            values = [successful_requests, failed_requests]
            colors = [self.color_success, self.color_failure]
            
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
            
            logger.info(f"✅ Chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"❌ Error generating chart: {e}")
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
                logger.warning("⚠️ No data to chart (total_requests = 0)")
                return None
            
            logger.info(f"📊 Generating pie chart for {target_type} '{target_value}'")
            
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
            
            logger.info(f"✅ Pie chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"❌ Error generating pie chart: {e}")
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
