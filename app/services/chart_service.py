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
    
    def generate_donut_chart(self, data: dict) -> Optional[str]:
        """
        Generate a donut chart (pie chart with a hole in the center).
        
        Similar to pie chart but with center cutout for modern look.
        Great for emphasizing a single percentage.
        
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
            
            logger.info(f"Generating donut chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
            
            # Data for donut chart
            sizes = [successful_requests, failed_requests]
            labels = [f'Successful\n{successful_requests:,}', f'Failed\n{failed_requests:,}']
            colors = [self.color_success, self.color_failure]
            explode = (0.05, 0.05)
            
            # Create donut chart (pie with wedgeprops)
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                wedgeprops=dict(width=0.5),  # This creates the donut hole
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # Make percentage text white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_fontweight('bold')
            
            # Add center circle for emphasis
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            
            # Equal aspect ratio ensures circular donut
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
            
            logger.info(f"Donut chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"Error generating donut chart: {e}")
            plt.close('all')
            return None
    
    def generate_line_chart(self, data: dict, chart_type: str) -> Optional[str]:
        """
        Generate a line chart showing trend over time.
        
        Note: Current data structure doesn't include time series, so this
        creates a simple two-point line chart as placeholder.
        For true time series, data structure would need timestamps.
        
        Args:
            data: Analytics data (same format as bar chart)
            chart_type: "success_rate" or "failure_rate"
        
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
            
            logger.info(f"Generating line chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # For now, create a simple representation with current values
            # In a real scenario, this would plot time series data
            categories = ['Start', 'Current']
            values = [0, rate]
            
            # Create line chart
            ax.plot(categories, values, marker='o', linewidth=2, markersize=10,
                   color=self.color_success if chart_type == "success_rate" else self.color_failure)
            
            # Fill area under the line
            ax.fill_between(range(len(categories)), values, alpha=0.3,
                           color=self.color_success if chart_type == "success_rate" else self.color_failure)
            
            # Add value labels
            for i, (cat, val) in enumerate(zip(categories, values)):
                ax.text(i, val, f'{val:.1f}%', ha='center', va='bottom',
                       fontsize=12, fontweight='bold')
            
            # Customize chart
            ax.set_ylabel(f'{rate_label} (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Timeline', fontsize=12, fontweight='bold')
            ax.set_title(
                f'{rate_label} Trend for {target_type.title()} "{target_value}"\n'
                f'Current: {rate:.2f}% | Total: {total_requests:,} requests',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            # Set y-axis range
            ax.set_ylim(0, max(100, rate * 1.2))
            
            # Add grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.xaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
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
            
            logger.info(f"Line chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"Error generating line chart: {e}")
            plt.close('all')
            return None
    
    def generate_area_chart(self, data: dict, chart_type: str) -> Optional[str]:
        """
        Generate an area chart showing cumulative values.
        
        Similar to line chart but with emphasized filled area.
        Good for showing volume/magnitude.
        
        Args:
            data: Analytics data (same format as bar chart)
            chart_type: "success_rate" or "failure_rate"
        
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
            
            # Get the appropriate rate
            if chart_type == "success_rate":
                rate = data.get("success_rate", 0)
                rate_label = "Success Rate"
                color = self.color_success
            else:
                rate = data.get("failure_rate", 0)
                rate_label = "Failure Rate"
                color = self.color_failure
            
            # Validate data
            if total_requests == 0:
                logger.warning("No data to chart (total_requests = 0)")
                return None
            
            logger.info(f"Generating area chart for {target_type} '{target_value}'")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Create area representation
            categories = ['Start', 'Current']
            values = [0, rate]
            
            # Create area chart
            ax.fill_between(range(len(categories)), values, alpha=0.7, color=color, linewidth=0)
            ax.plot(categories, values, marker='o', linewidth=3, markersize=12, color=color)
            
            # Add value labels
            for i, (cat, val) in enumerate(zip(categories, values)):
                ax.text(i, val, f'{val:.1f}%', ha='center', va='bottom',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, linewidth=2))
            
            # Customize chart
            ax.set_ylabel(f'{rate_label} (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Timeline', fontsize=12, fontweight='bold')
            ax.set_title(
                f'{rate_label} Volume for {target_type.title()} "{target_value}"\n'
                f'Current: {rate:.2f}% | Total: {total_requests:,} requests',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            # Set y-axis range
            ax.set_ylim(0, max(100, rate * 1.2))
            
            # Add grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.xaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
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
            
            logger.info(f"Area chart generated successfully ({len(image_base64)} bytes)")
            
            return image_base64
            
        except Exception as e:
            logger.exception(f"Error generating area chart: {e}")
            plt.close('all')
            return None


# Convenience function
def generate_analytics_chart(data: dict, chart_type: str = "success_rate", style: str = "bar") -> Optional[str]:
    """
    Generate analytics chart with specified style.
    
    Args:
        data: Analytics data dictionary
        chart_type: "success_rate" or "failure_rate"
        style: "bar", "pie", "line", "donut", or "area"
    
    Returns:
        Base64-encoded PNG image string
    
    Example:
        >>> chart = generate_analytics_chart(data, chart_type="success_rate", style="bar")
        >>> print(f"data:image/png;base64,{chart}")
    """
    generator = AnalyticsChartGenerator()
    
    # Route to appropriate chart generation method based on style
    if style == "pie":
        return generator.generate_pie_chart(data)
    elif style == "donut":
        return generator.generate_donut_chart(data)
    elif style == "line":
        return generator.generate_line_chart(data, chart_type)
    elif style == "area":
        return generator.generate_area_chart(data, chart_type)
    elif style == "bar":
        return generator.generate_success_failure_bar_chart(data, chart_type)
    else:
        # Unknown style, fallback to bar
        logger.warning(f"Unknown chart style '{style}', falling back to bar chart")
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


async def get_chart_type_recommendation(
    user_query: str,
    report_type: str,
    data: dict
) -> str:
    """
    Use LLM to recommend the best chart type based on query semantics and data.
    
    Args:
        user_query: Original user query
        report_type: "success_rate" or "failure_rate"
        data: Analytics data dictionary
    
    Returns:
        One of: "bar", "pie", "line", "donut", "area"
    """
    from langchain_openai import ChatOpenAI
    from app.config import OPENAI_API_KEY
    
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0
        )
        
        prompt = f"""You are a data visualization expert. Recommend the BEST chart type for this analytics query.

USER QUERY: "{user_query}"
REPORT TYPE: {report_type}
DATA SUMMARY:
- Total requests: {data.get('total_requests', 0)}
- Success rate: {data.get('success_rate', 0)}%
- Failure rate: {data.get('failure_rate', 0)}%

AVAILABLE CHART TYPES:
- bar: Best for comparing categories, showing counts
- pie: Best for showing proportions/percentages, part-to-whole relationships
- line: Best for trends over time, continuous data
- donut: Best for highlighting a single percentage, circular proportion
- area: Best for cumulative trends, volume over time

RULES:
1. Consider the user's intent from their query
2. Match chart type to data characteristics
3. Choose the most intuitive visualization
4. Return ONLY the chart type name (bar/pie/line/donut/area)

RECOMMENDED CHART TYPE:"""
        
        response = await llm.ainvoke(prompt)
        chart_type = response.content.strip().lower()
        
        # Validate response
        valid_types = ["bar", "pie", "line", "donut", "area"]
        if chart_type not in valid_types:
            logger.warning(f"LLM returned invalid chart type: '{chart_type}', using rule-based fallback")
            return apply_chart_type_rules(user_query, report_type, data)
        
        logger.info(f"LLM recommended chart type: {chart_type}")
        return chart_type
        
    except Exception as e:
        logger.warning(f"LLM chart recommendation failed: {e}, using rule-based fallback")
        return apply_chart_type_rules(user_query, report_type, data)


def apply_chart_type_rules(
    user_query: str,
    report_type: str,
    data: dict
) -> str:
    """
    Rule-based chart type selection as fallback when LLM is unavailable.
    
    Args:
        user_query: Original user query
        report_type: "success_rate" or "failure_rate"
        data: Analytics data dictionary
    
    Returns:
        One of: "bar", "pie", "line", "donut", "area"
    """
    query_lower = user_query.lower()
    
    # Rule 1: Time/trend keywords → line chart
    if any(keyword in query_lower for keyword in ["trend", "over time", "timeline", "track", "history"]):
        logger.info("Rule-based: Selected 'line' (trend keywords detected)")
        return "line"
    
    # Rule 2: Proportion/percentage keywords → pie chart
    if any(keyword in query_lower for keyword in ["proportion", "percentage", "breakdown", "distribution", "share"]):
        logger.info("Rule-based: Selected 'pie' (proportion keywords detected)")
        return "pie"
    
    # Rule 3: High failure rate → donut (visual emphasis)
    failure_rate = data.get("failure_rate", 0)
    if report_type == "failure_rate" and failure_rate > 80:
        logger.info(f"Rule-based: Selected 'donut' (high failure rate: {failure_rate}%)")
        return "donut"
    
    # Rule 4: High success rate → donut (visual emphasis)
    success_rate = data.get("success_rate", 0)
    if report_type == "success_rate" and success_rate > 80:
        logger.info(f"Rule-based: Selected 'donut' (high success rate: {success_rate}%)")
        return "donut"
    
    # Rule 5: Large dataset → area chart
    total = data.get("total_requests", 0)
    if total > 100:
        logger.info(f"Rule-based: Selected 'area' (large dataset: {total} requests)")
        return "area"
    
    # Default: bar chart (works for most cases)
    logger.info("Rule-based: Selected 'bar' (default)")
    return "bar"
