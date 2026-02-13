"""HTML report generation with interactive charts."""

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate interactive HTML reports with Plotly charts."""

    def __init__(self, template_dir: str = "./templates"):
        self.template_dir = Path(template_dir)
        self.template_file = self.template_dir / "report.html.j2"

    def generate(
        self,
        results: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Generate an HTML report.

        Args:
            results: Benchmark results dictionary
            output_path: Path to save the HTML file

        Returns:
            Path to the generated file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        metrics = results.get("metrics", {})
        sentences = results.get("sentences", [])

        bleu_distribution = [s.get("bleu", 0) for s in sentences]
        chrf_distribution = [s.get("chrf", 0) for s in sentences]
        meteor_distribution = [s.get("meteor", 0) for s in sentences]

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "BLEU Score Distribution",
                "chrF Score Distribution",
                "METEOR Score Distribution",
                "BLEU vs chrF Correlation",
                "BLEU vs METEOR Correlation",
                "Metrics Summary",
            ),
        )

        fig.add_trace(
            go.Histogram(
                x=bleu_distribution,
                nbinsx=20,
                name="BLEU",
                marker_color="#2196F3",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=chrf_distribution,
                nbinsx=20,
                name="chrF",
                marker_color="#4CAF50",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                x=meteor_distribution,
                nbinsx=20,
                name="METEOR",
                marker_color="#FF9800",
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=bleu_distribution,
                y=chrf_distribution,
                mode="markers",
                name="Sentences",
                marker=dict(size=8, opacity=0.7),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=bleu_distribution,
                y=meteor_distribution,
                mode="markers",
                name="Sentences",
                marker=dict(size=8, opacity=0.7),
            ),
            row=2,
            col=2,
        )

        metrics_names = ["BLEU", "chrF", "METEOR"]
        metrics_values = [
            metrics.get("bleu", 0),
            metrics.get("chrf", 0),
            metrics.get("meteor", 0),
        ]

        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                name="Score",
                marker_color=["#2196F3", "#4CAF50", "#FF9800"],
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title_text=f"Translation Benchmark Report: {results.get('model', 'Unknown Model')}",
            height=900,
            showlegend=False,
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPUS-LLM-Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .meta-item {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .meta-item label {{
            display: block;
            color: #6c757d;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .meta-item value {{
            display: block;
            font-size: 1.2em;
            font-weight: 600;
            color: #343a40;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 30px;
            padding: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }}
        .metric-card h3 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 3em;
            font-weight: 700;
        }}
        .metric-card .suffix {{
            font-size: 1em;
            opacity: 0.8;
        }}
        .charts-section {{
            padding: 40px;
            background: #f8f9fa;
        }}
        .charts-section h2 {{
            margin-bottom: 30px;
            color: #343a40;
        }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .table-section {{
            padding: 40px;
        }}
        .table-section h2 {{
            margin-bottom: 20px;
            color: #343a40;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .footer {{
            background: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OPUS-LLM-Benchmark Report</h1>
            <p>Translation Quality Evaluation</p>
        </div>

        <div class="metadata">
            <div class="meta-item">
                <label>Model</label>
                <value>{results.get("model", "Unknown")}</value>
            </div>
            <div class="meta-item">
                <label>Corpus</label>
                <value>{results.get("corpus", "Unknown")}</value>
            </div>
            <div class="meta-item">
                <label>Language Pair</label>
                <value>{results.get("source", "")} → {results.get("target", "")}</value>
            </div>
            <div class="meta-item">
                <label>Samples</label>
                <value>{results.get("num_samples", 0)}</value>
            </div>
            <div class="meta-item">
                <label>Timestamp</label>
                <value>{results.get("timestamp", datetime.now().isoformat())}</value>
            </div>
            <div class="meta-item">
                <label>Execution Time</label>
                <value>{results.get("execution_time", "N/A")}</value>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>BLEU Score</h3>
                <div class="value">{metrics.get("bleu", 0):.2f}</div>
                <div class="suffix">/ 100</div>
            </div>
            <div class="metric-card">
                <h3>chrF++ Score</h3>
                <div class="value">{metrics.get("chrf", 0):.2f}</div>
                <div class="suffix">/ 100</div>
            </div>
            <div class="metric-card">
                <h3>METEOR Score</h3>
                <div class="value">{metrics.get("meteor", 0):.2f}</div>
                <div class="suffix">/ 100</div>
            </div>
        </div>

        <div class="charts-section">
            <h2>Score Distributions</h2>
            <div class="chart-container">
                {chart_html}
            </div>
        </div>

        <div class="table-section">
            <h2>Per-Sentence Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Source</th>
                        <th>Reference</th>
                        <th>Hypothesis</th>
                        <th>BLEU</th>
                        <th>chrF</th>
                        <th>METEOR</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_table_rows(sentences)}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by OPUS-LLM-Benchmark v0.1.0</p>
            <p>Timestamp: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>"""

        output_file.write_text(html_content, encoding="utf-8")
        logger.info(f"HTML report saved to: {output_file}")

        return str(output_file)

    def _generate_table_rows(self, sentences: List[Dict]) -> str:
        """Generate HTML table rows for sentences."""
        rows = []
        for i, sent in enumerate(sentences):
            row = f"""
            <tr>
                <td>{i + 1}</td>
                <td>{self._escape_html(sent.get("source", ""))}</td>
                <td>{self._escape_html(sent.get("reference", ""))}</td>
                <td>{self._escape_html(sent.get("hypothesis", ""))}</td>
                <td>{sent.get("bleu", 0):.2f}</td>
                <td>{sent.get("chrf", 0):.2f}</td>
                <td>{sent.get("meteor", 0):.2f}</td>
            </tr>
            """
            rows.append(row)
        return "\n".join(rows)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
