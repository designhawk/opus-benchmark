"""Main CLI entry point for OPUS-LLM-Benchmark."""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import SIMPLE

from utils.config import Config
from llm.translator import LLMTranslator
from evaluation.metrics import TranslationEvaluator
from reporting.html_report import HTMLReportGenerator
from config.languages import TARGET_LANGUAGES, DEFAULT_LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console(force_terminal=True)


def load_parallel_sentences(
    file_path: Path,
) -> Tuple[List[str], List[str]]:
    """Load parallel sentences from a file.

    File format: source[TAB]target (one pair per line)
    """
    if not file_path.exists():
        return [], []

    source_sentences = []
    target_sentences = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    source_sentences.append(parts[0])
                    target_sentences.append(parts[1])

    return source_sentences, target_sentences


def get_available_language_pairs(
    data_dir: Path,
    source: str = "en",
) -> List[str]:
    """Get available language pairs from data directory.

    Scans for files matching {source}-{target}.txt pattern.
    """
    if not data_dir.exists():
        return []

    pairs = []
    for file_path in data_dir.glob(f"{source}-*.txt"):
        # Extract target language from filename (e.g., en-de.txt -> de)
        stem = file_path.stem
        if stem.startswith(f"{source}-"):
            target = stem[len(source) + 1 :]
            pairs.append(target)

    return sorted(pairs)


def run_benchmark(
    source_sentences: List[str],
    target_sentences: List[str],
    target: str,
    num_samples: int,
    model: str,
    source: str = "en",
    corpus_name: str = "tatoeba",
):
    """Run the translation benchmark."""
    config = Config()
    if not config.validate():
        sys.exit(1)

    api_key = config.get_api_key()

    # Sample sentences if needed
    if len(source_sentences) > num_samples:
        import random

        indices = random.sample(range(len(source_sentences)), num_samples)
        source_sentences = [source_sentences[i] for i in indices]
        target_sentences = [target_sentences[i] for i in indices]

    console.print(
        Panel(
            Text(
                f"Starting Benchmark\n\n"
                f"Corpus: {corpus_name}\n"
                f"Languages: {source.upper()} -> {target.upper()}\n"
                f"Samples: {len(source_sentences)}\n"
                f"Model: {model}",
                style="cyan",
            ),
            title="OPUS-LLM-Benchmark",
            expand=False,
        )
    )

    start_time = time.time()

    try:
        translator = LLMTranslator(api_key, model)

        console.print("\n[bold cyan]Step 1: Translating...[/bold cyan]")
        translations = translator.translate_batch(
            texts=source_sentences,
            source_lang=source,
            target_lang=target,
            num_samples=num_samples,
        )

        evaluator = TranslationEvaluator()

        console.print("\n[bold cyan]Step 2: Evaluating...[/bold cyan]")
        references = [[ref] for ref in target_sentences]
        hypotheses = [translations.get(i, "") for i in range(len(source_sentences))]

        metrics = evaluator.evaluate(references, hypotheses)
        sentence_metrics = evaluator.evaluate_per_sentence(references, hypotheses)

        execution_time = time.time() - start_time

        console.print("\n[bold cyan]Results[/bold cyan]")
        results_table = Table()
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Score", style="green")
        results_table.add_row("BLEU", f"{metrics['bleu']:.2f}")
        results_table.add_row("chrF++", f"{metrics['chrf']:.2f}")
        results_table.add_row("METEOR", f"{metrics['meteor']:.2f}")
        console.print(results_table)

        results: Dict[str, Any] = {
            "model": model,
            "corpus": corpus_name,
            "source": source,
            "target": target,
            "num_samples": len(source_sentences),
            "metrics": metrics,
            "sentences": [
                {
                    "source": source_sentences[i],
                    "reference": target_sentences[i],
                    "hypothesis": translations.get(i, ""),
                    "bleu": sentence_metrics[i].get("bleu", 0),
                    "chrf": sentence_metrics[i].get("chrf", 0),
                }
                for i in range(len(source_sentences))
            ],
            "timestamp": datetime.now().isoformat(),
            "execution_time": f"{execution_time:.2f}s",
        }

        report_generator = HTMLReportGenerator()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_dir = Path.cwd() / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = (
            output_dir
            / f"benchmark_{source.upper()}-{target.upper()}_{corpus_name}_{timestamp}.html"
        )
        report_generator.generate(results, str(report_file))

        console.print(f"\n[green]Report saved to: {report_file}[/green]")
        console.print(f"[green]Execution time: {execution_time:.2f}s[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Benchmark failed")
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0", prog_name="opus-benchmark")
def main():
    """OPUS-LLM-Benchmark: Translate and evaluate LLM translations using OPUS corpora."""
    pass


@main.group()
def config():
    """Configure application settings."""
    pass


@config.command("set-api-key")
def set_api_key():
    """Set OpenRouter API key."""
    config = Config()
    if not config.get_api_key():
        console.print("Enter your OpenRouter API key:")
        api_key = click.prompt("API Key", type=str, hide_input=True)
        config.set_api_key(api_key)
    else:
        console.print("[yellow]API key already set[/yellow]")
        if click.confirm("Do you want to update it?"):
            api_key = click.prompt("New API Key", type=str, hide_input=True)
            config.set_api_key(api_key)


@config.command("show")
def show_config():
    """Show current configuration."""
    config = Config()
    config.show()


@config.command("set-model")
@click.argument("model", type=str)
def set_default_model(model: str):
    """Set default model."""
    config = Config()
    config.set_default_model(model)


@config.command("use-free")
def use_free_models():
    """Switch to OpenRouter free tier models."""
    config = Config()

    console.print("[cyan]Switching to OpenRouter FREE tier...[/cyan]\n")

    free_models = [
        (
            "arcee-ai/trinity-large-preview:free",
            "Auto-select best free model (recommended)",
        ),
        ("google/gemma-3-27b-it:free", "Google Gemma 3 27B - Good for translation"),
        ("qwen/qwen-2.5-72b-instruct:free", "Qwen 2.5 72B - Excellent multilingual"),
        ("deepseek/deepseek-chat:free", "DeepSeek Chat - Strong reasoning"),
        ("arcee-ai/trinity-large-preview:free", "Llama 3.3 70B - Reliable"),
        ("nvidia/llama-3.1-nemotron-70b-instruct:free", "NVIDIA Nemotron 70B - Fast"),
    ]

    console.print("Available FREE models:")
    console.print("=" * 60)
    for i, (model_id, description) in enumerate(free_models, 1):
        console.print(f"{i}. [cyan]{model_id}[/cyan]")
        console.print(f"   {description}")
        console.print()

    choice = click.prompt("Select model (1-6)", type=int, default=1)

    if 1 <= choice <= len(free_models):
        selected_model = free_models[choice - 1][0]
        config.set_default_model(selected_model)
        console.print(f"[green]Default model set to: {selected_model}[/green]")
        console.print("[dim]You can now run benchmarks without incurring costs![/dim]")
    else:
        console.print("[red]Invalid selection[/red]")


@config.command("interactive")
def interactive_setup():
    """Run interactive configuration."""
    config = Config()
    config.interactive_setup()


@main.group()
def list():
    """List available resources."""
    pass


@list.command("targets")
@click.option(
    "--source",
    default="en",
    help="Source language code",
)
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing parallel corpus files",
)
def list_targets(source: str, data_dir: str):
    """List available target languages for a source."""
    data_path = Path(data_dir)
    available_pairs = get_available_language_pairs(data_path, source)

    table = Table(title=f"Available Targets for {source.upper()}")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="green")

    # If no files found, show available from config
    if not available_pairs:
        console.print(f"[yellow]No corpus files found in {data_dir}[/yellow]")
        console.print("[dim]Available languages in config:[/dim]")
        for target in sorted(TARGET_LANGUAGES.keys()):
            if target != source:
                lang_info = TARGET_LANGUAGES[target]
                table.add_row(target.upper(), lang_info["name"])
    else:
        for target in available_pairs:
            lang_name = TARGET_LANGUAGES.get(target, {}).get("name", target)
            table.add_row(target.upper(), lang_name)

    console.print(table)


@main.command()
@click.option(
    "--target",
    required=True,
    help="Target language code (e.g., de, fr, ja)",
)
@click.option(
    "--samples",
    default=10,
    type=int,
    help="Number of samples",
)
@click.option(
    "--model",
    default="arcee-ai/trinity-large-preview:free",
    help="OpenRouter model",
)
@click.option(
    "--source",
    default="en",
    help="Source language code",
)
@click.option(
    "--file",
    "corpus_file",
    default=None,
    help="Path to corpus file (auto-detected if not specified)",
)
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing parallel corpus files",
)
def run(
    target: str,
    samples: int,
    model: str,
    source: str,
    corpus_file: Optional[str],
    data_dir: str,
):
    """Run a translation benchmark."""
    data_path = Path(data_dir)

    # Determine corpus file path
    if corpus_file:
        file_path = Path(corpus_file)
    else:
        # Auto-detect from data/tatoeba/en-{target}.txt
        file_path = data_path / f"{source}-{target}.txt"

    # Load sentences
    console.print(f"[cyan]Loading sentences from {file_path}...[/cyan]")
    source_sentences, target_sentences = load_parallel_sentences(file_path)

    if not source_sentences or not target_sentences:
        console.print(f"[red]No sentences found in {file_path}[/red]")
        console.print(
            f"[dim]Place corpus files in {data_dir}/ or use --file to specify a path[/dim]"
        )
        console.print("[dim]Expected format: source ||| target (one per line)[/dim]")
        sys.exit(1)

    console.print(f"[green]Loaded {len(source_sentences)} sentence pairs[/green]")

    run_benchmark(
        source_sentences=source_sentences,
        target_sentences=target_sentences,
        target=target,
        num_samples=samples,
        model=model,
        source=source,
    )


@main.command()
@click.option(
    "--input",
    help="Input results file or directory",
)
@click.option(
    "--output",
    help="Output HTML file",
)
def report(input: Optional[str], output: Optional[str]):
    """Generate HTML report from results."""
    if not input:
        reports_dir = Path.cwd() / "reports"
        if reports_dir.exists():
            reports = list(reports_dir.glob("benchmark_*.html"))
            if reports:
                input = str(reports[-1])
            else:
                console.print("[red]No reports found in ./reports[/red]")
                sys.exit(1)
        else:
            console.print("[red]No reports directory found[/red]")
            sys.exit(1)

    if output:
        output_file = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = Path.cwd() / "reports" / f"report_{timestamp}.html"

    report_generator = HTMLReportGenerator()

    if input.endswith(".json"):
        with open(input) as f:
            results = json.load(f)
        report_file = report_generator.generate(results, str(output_file))
    elif input.endswith(".html"):
        console.print("[yellow]HTML report already exists, copying...[/yellow]")
        import shutil

        shutil.copy(input, output_file)
        report_file = output_file
    else:
        console.print("[red]Unknown input format[/red]")
        sys.exit(1)

    console.print(f"[green]Report saved to: {report_file}[/green]")


@main.command()
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing ZIP files",
)
def convert(data_dir: str):
    """Convert ZIP files to parallel corpus format."""
    from data.zip_converter import convert_all

    data_path = Path(data_dir)
    results = convert_all(data_path)

    if results:
        console.print(f"\n[green]Successfully converted {len(results)} file(s)[/green]")
    else:
        console.print("[yellow]No files converted[/yellow]")


@main.command("run-multi")
@click.option(
    "--samples",
    default=10,
    type=int,
    help="Samples per language to translate",
)
@click.option(
    "--model",
    default="arcee-ai/trinity-large-preview:free",
    help="OpenRouter model (default: arcee-ai/trinity-large-preview:free - uses best free model)",
)
@click.option(
    "--langs",
    default=None,
    help="Comma-separated language codes (default: all available)",
)
@click.option(
    "--source",
    default="en",
    help="Source language code",
)
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory with parallel corpus files (default: ./data/tatoeba)",
)
@click.option(
    "--output",
    default=None,
    help="Output HTML file (auto-generated if not specified)",
)
def run_multi(
    samples: int,
    model: str,
    langs: Optional[str],
    source: str,
    data_dir: str,
    output: Optional[str],
):
    """Run translation benchmark across multiple languages."""
    config = Config()
    if not config.validate():
        sys.exit(1)

    api_key = config.get_api_key()

    # Get available language pairs from data directory
    data_path = Path(data_dir)
    available_pairs = get_available_language_pairs(data_path, source)

    if not available_pairs:
        console.print(f"[red]No corpus files found in {data_dir}[/red]")
        console.print(f"[dim]Expected files: {source}-{{target}}.txt[/dim]")
        sys.exit(1)

    # Determine which languages to process
    if langs:
        languages = [l.strip().lower() for l in langs.split(",")]
        # Filter to only available pairs
        languages = [lang for lang in languages if lang in available_pairs]
        if not languages:
            console.print(f"[red]None of the specified languages are available[/red]")
            console.print(f"[dim]Available: {', '.join(available_pairs)}[/dim]")
            sys.exit(1)
    else:
        languages = available_pairs.copy()

    console.print(
        Panel.fit(
            f"Multi-Language Translation Benchmark\n\n"
            f"Languages: {len(languages)}\n"
            f"Samples: {samples}/language\n"
            f"Model: {model}\n"
            f"Source: {source.upper()}",
            style="cyan bold",
        )
    )

    start_time = time.time()
    translator = LLMTranslator(api_key, model)
    evaluator = TranslationEvaluator()

    all_results = []

    for i, target in enumerate(languages, 1):
        lang_info = TARGET_LANGUAGES.get(target, {"name": target})
        console.print(
            f"\n[{i}/{len(languages)}] Processing {lang_info['name']} ({target.upper()})..."
        )

        # Load sentences
        file_path = data_path / f"{source}-{target}.txt"
        source_sentences, target_sentences = load_parallel_sentences(file_path)

        if not source_sentences or not target_sentences:
            console.print(f"  [yellow]No data available for {target.upper()}[/yellow]")
            continue

        # Sample if needed
        if len(source_sentences) > samples:
            import random

            indices = random.sample(range(len(source_sentences)), samples)
            source_sentences = [source_sentences[idx] for idx in indices]
            target_sentences = [target_sentences[idx] for idx in indices]

        console.print(
            f"  [cyan]Translating {len(source_sentences)} sentences...[/cyan]"
        )

        translations = translator.translate_batch(
            texts=source_sentences,
            source_lang=source,
            target_lang=target,
            num_samples=samples,
        )

        if not translations:
            console.print(f"  [red]Translation failed for {target.upper()}[/red]")
            continue

        console.print(f"  [cyan]Evaluating...[/cyan]")

        references = [[ref] for ref in target_sentences]
        hypotheses = [translations.get(j, "") for j in range(len(source_sentences))]

        metrics = evaluator.evaluate(references, hypotheses)
        sentence_metrics = evaluator.evaluate_per_sentence(references, hypotheses)

        all_results.append(
            {
                "target": target,
                "name": lang_info["name"],
                "metrics": metrics,
                "sentences": [
                    {
                        "source": source_sentences[j],
                        "reference": target_sentences[j],
                        "hypothesis": translations.get(j, ""),
                        "bleu": sentence_metrics[j].get("bleu", 0),
                        "chrf": sentence_metrics[j].get("chrf", 0),
                    }
                    for j in range(len(source_sentences))
                ],
            }
        )

        console.print(
            f"  [green]Done: BLEU={metrics['bleu']:.1f}, chrF={metrics['chrf']:.1f}[/green]"
        )

    execution_time = time.time() - start_time

    console.print("\n[bold cyan]Benchmark Complete![/bold cyan]")

    # Generate summary
    if all_results:
        total_bleu = sum(r["metrics"]["bleu"] for r in all_results)
        avg_bleu = total_bleu / len(all_results)

        summary_table = Table(title="Summary")
        summary_table.add_column("Language", style="cyan")
        summary_table.add_column("BLEU", style="green")
        summary_table.add_column("chrF++", style="green")

        for result in all_results:
            summary_table.add_row(
                result["name"],
                f"{result['metrics']['bleu']:.2f}",
                f"{result['metrics']['chrf']:.2f}",
            )

        console.print(summary_table)
        console.print(f"\n[cyan]Average BLEU: {avg_bleu:.2f}[/cyan]")

    # Generate report
    if output:
        output_file = output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"./reports/multi-benchmark_{timestamp}.html"

    # Build results dict for report
    results = {
        "model": model,
        "source": source,
        "targets": [r["target"] for r in all_results],
        "samples": samples,
        "all_results": all_results,
        "execution_time": f"{execution_time:.2f}s",
    }

    report_gen = HTMLReportGenerator()
    report_gen.generate(results, output_file)

    console.print(f"\n[green]Report saved to: {output_file}[/green]")
    console.print(f"[green]Execution time: {execution_time:.2f}s[/green]")


if __name__ == "__main__":
    main()
