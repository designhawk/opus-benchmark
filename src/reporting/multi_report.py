"""Multi-language HTML report generator for OPUS-LLM-Benchmark.

Generates interactive HTML reports for multi-language translation benchmarks.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.languages import TARGET_LANGUAGES


class MultiReportGenerator:
    """Generate HTML reports for multi-language benchmarks."""

    def __init__(self):
        pass

    def generate(
        self,
        results: Dict[str, Any],
        output_file: str,
        model: str = "arcee-ai/trinity-large-preview:free",
        samples_per_lang: int = 10,
        corpus: str = "wikimedia",
    ):
        """Generate a multi-language HTML report."""
        languages_data = results.get("results", {})
        summary = results.get("summary", {})
        rankings = results.get("rankings", {})

        ranking_bleu = rankings.get("bleu", [])
        ranking_chrf = rankings.get("chrf", [])

        html_content = self._generate_html(
            languages_data=languages_data,
            summary=summary,
            ranking_bleu=ranking_bleu,
            ranking_chrf=ranking_chrf,
            model=model,
            samples=samples_per_lang,
            corpus=corpus,
        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(output_path)

    def _generate_html(
        self,
        languages_data: Dict[str, Dict],
        summary: Dict[str, Any],
        ranking_bleu: List[tuple],
        ranking_chrf: List[tuple],
        model: str,
        samples: int,
        corpus: str,
    ) -> str:
        """Generate HTML content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        ranking_rows = ""
        for i, (lang, bleu, samps) in enumerate(ranking_bleu, 1):
            lang_info = TARGET_LANGUAGES.get(lang, {"name": lang})
            chrf_score = next((l[1] for l in ranking_chrf if l[0] == lang), 0)
            ranking_rows += f"""
            <tr>
                <td>{i}</td>
                <td><span class="lang-code">{lang.upper()}</span> {lang_info.get("name", lang)}</td>
                <td class="score">{bleu:.1f}</td>
                <td class="score">{chrf_score:.1f}</td>
                <td>{samps}</td>
            </tr>
            """

        detail_sections = ""
        for lang in sorted(languages_data.keys()):
            data = languages_data[lang]
            lang_info = TARGET_LANGUAGES.get(lang, {"name": lang})
            metrics = data.get("metrics", {})
            sentences = data.get("sentences", [])

            sentence_rows = ""
            for j, sent in enumerate(sentences[:samples], 1):
                bleu = sent.get("bleu", 0)
                chrf = sent.get("chrf", 0)
                src = sent.get("source", "")[:200]
                ref = sent.get("reference", "")[:200]
                hyp = sent.get("hypothesis", "")[:200]

                sentence_rows += f"""
                <div class="sentence-pair">
                    <div class="sentence-header">
                        <span class="badge">{j}</span>
                        <span class="metrics">BLEU: {bleu:.1f} | chrF: {chrf:.1f}</span>
                    </div>
                    <div class="sentence-content">
                        <div class="source"><strong>EN:</strong> {self._escape_html(src)}</div>
                        <div class="reference"><strong>REF:</strong> {self._escape_html(ref)}</div>
                        <div class="hypothesis"><strong>HYP:</strong> {self._escape_html(hyp)}</div>
                    </div>
                </div>
                """

            detail_sections += f"""
            <div class="language-detail" id="lang-{lang}">
                <h2><span class="lang-code">{lang.upper()}</span> {lang_info.get("name", lang)}</h2>
                <div class="metrics-summary">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get("bleu", 0):.1f}</div>
                        <div class="metric-label">BLEU</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get("chrf", 0):.1f}</div>
                        <div class="metric-label">chrF++</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get("samples", 0)}</div>
                        <div class="metric-label">Samples</div>
                    </div>
                </div>
                <h3>Sample Translations</h3>
                <div class="sentences">{sentence_rows}</div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Language Translation Benchmark</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --text-light: #64748b;
            --border: #e2e8f0;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}

        .header h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}

        .header-meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .summary-card {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}

        .summary-card .label {{
            color: var(--text-light);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}

        .card {{
            background: var(--card);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            overflow: hidden;
        }}

        .card-header {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }}

        .card-body {{
            padding: 1.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--text-light);
        }}

        tr:hover {{
            background: var(--bg);
        }}

        .score {{
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }}

        .lang-code {{
            background: var(--primary);
            color: white;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}

        .top-performer {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        }}

        .language-detail {{
            margin-top: 2rem;
        }}

        .language-detail h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }}

        .metrics-summary {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: var(--bg);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--primary);
        }}

        .metric-label {{
            font-size: 0.75rem;
            color: var(--text-light);
        }}

        .sentence-pair {{
            background: var(--bg);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}

        .sentence-header {{
            background: var(--border);
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .badge {{
            background: var(--primary);
            color: white;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .metrics {{
            font-size: 0.875rem;
            color: var(--text-light);
        }}

        .sentence-content {{
            padding: 1rem;
        }}

        .source, .reference, .hypothesis {{
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}

        .hypothesis {{
            background: #fef3c7;
            padding: 0.5rem;
            border-radius: 4px;
        }}

        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-light);
            font-size: 0.875rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            .summary-cards {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Language Translation Benchmark</h1>
        <div class="header-meta">
            Generated: {timestamp} | Model: {model} | Corpus: {corpus} | Samples: {samples}/language
        </div>
    </div>

    <div class="container">
        <div class="summary-cards">
            <div class="summary-card">
                <div class="value">{summary.get("num_languages", len(languages_data))}</div>
                <div class="label">Languages</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get("bleu", {}).get("mean", 0):.1f}</div>
                <div class="label">Avg BLEU</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get("chrf", {}).get("mean", 0):.1f}</div>
                <div class="label">Avg chrF++</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get("bleu", {}).get("max", 0):.1f}</div>
                <div class="label">Best BLEU</div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Language Rankings (by BLEU score)</div>
            <div class="card-body" style="padding: 0;">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 50px;">Rank</th>
                            <th>Language</th>
                            <th style="width: 100px;">BLEU</th>
                            <th style="width: 100px;">chrF++</th>
                            <th style="width: 100px;">Samples</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ranking_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Detailed Results by Language</div>
            <div class="card-body">
                {detail_sections if detail_sections else "<p>No detailed results available.</p>"}
            </div>
        </div>
    </div>

    <div class="footer">
        Generated by OPUS-LLM-Benchmark
    </div>
</body>
</html>
"""
        return html

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def generate_from_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        output_file: str,
        model: str = "arcee-ai/trinity-large-preview:free",
    ):
        """Generate report from checkpoint data."""
        languages = {}
        for lang, data in checkpoint_data.get("languages", {}).items():
            if data.get("status") == "completed" and "results" in data:
                sentences = data.get("results", [])
                languages[lang] = {
                    "source": "en",
                    "target": lang,
                    "samples": data.get("samples", 0),
                    "metrics": data.get("metrics", {}),
                    "sentences": sentences,
                }

        results = {
            "results": languages,
            "summary": self._calculate_summary(languages),
            "rankings": self._calculate_rankings(languages),
        }

        return self.generate(results, output_file, model)

    def _calculate_summary(self, languages: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not languages:
            return {}

        bleu_scores = [d.get("metrics", {}).get("bleu", 0) for d in languages.values()]
        chrf_scores = [d.get("metrics", {}).get("chrf", 0) for d in languages.values()]

        import statistics

        return {
            "num_languages": len(languages),
            "bleu": {
                "mean": round(statistics.mean(bleu_scores), 2),
                "median": round(statistics.median(bleu_scores), 2),
                "stdev": round(statistics.stdev(bleu_scores), 2)
                if len(bleu_scores) > 1
                else 0,
                "min": round(min(bleu_scores), 2),
                "max": round(max(bleu_scores), 2),
            },
            "chrf": {
                "mean": round(statistics.mean(chrf_scores), 2),
                "median": round(statistics.median(chrf_scores), 2),
                "stdev": round(statistics.stdev(chrf_scores), 2)
                if len(chrf_scores) > 1
                else 0,
                "min": round(min(chrf_scores), 2),
                "max": round(max(chrf_scores), 2),
            },
        }

    def _calculate_rankings(self, languages: Dict[str, Dict]) -> Dict[str, List]:
        """Calculate rankings."""
        bleu_ranking = sorted(
            [
                (lang, d.get("metrics", {}).get("bleu", 0), d.get("samples", 0))
                for lang, d in languages.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        chrf_ranking = sorted(
            [
                (lang, d.get("metrics", {}).get("chrf", 0), d.get("samples", 0))
                for lang, d in languages.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return {"bleu": bleu_ranking, "chrf": chrf_ranking}
