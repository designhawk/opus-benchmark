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
from reporting.multi_report import MultiReportGenerator
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
                    "meteor": sentence_metrics[i].get("meteor", 0),
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


@list.command("duplicates")
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing parallel corpus files",
)
def list_duplicates(data_dir):
    """Check for duplicate sentences in corpus files."""
    import os
    data_dir = str(data_dir)
    
    if not os.path.exists(data_dir):
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        return

    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        console.print(f"[yellow]No .txt files found in {data_dir}[/yellow]")
        return

    table = Table(title="Duplicate Sentences Check")
    table.add_column("File", style="cyan")
    table.add_column("Languages", style="green")
    table.add_column("Total", justify="right", style="yellow")
    table.add_column("Unique", justify="right", style="green")
    table.add_column("Duplicates", justify="right", style="red")

    total_dupes = 0
    total_unique = 0
    total_sentences = 0

    for filename in sorted(txt_files):
        file_path = os.path.join(data_dir, filename)
        
        sources = set()
        duplicates = 0
        total = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    parts = line.strip().split("\t")
                    if len(parts) >= 1:
                        src = parts[0]
                        if src in sources:
                            duplicates += 1
                        else:
                            sources.add(src)

        total_dupes += duplicates
        total_unique += len(sources)
        total_sentences += total

        stem = os.path.splitext(filename)[0]
        if "-" in stem:
            parts = stem.split("-")
            if len(parts) == 2:
                src, tgt = parts
                src_name = TARGET_LANGUAGES.get(src, {}).get("name", src.upper())
                tgt_name = TARGET_LANGUAGES.get(tgt, {}).get("name", tgt.upper())
                lang_pair = f"{src_name} -> {tgt_name}"
            else:
                lang_pair = stem
        else:
            lang_pair = stem

        table.add_row(
            filename,
            lang_pair,
            f"{total:,}",
            f"{len(sources):,}",
            f"{duplicates:,}",
        )

    table.add_row(
        "",
        "",
        f"{total_sentences:,}",
        f"{total_unique:,}",
        f"{total_dupes:,}",
        style="bold",
    )

    console.print(table)


@main.command("clean")
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing parallel corpus files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be removed without actually removing",
)
def clean(data_dir, dry_run):
    """Remove duplicate sentences from corpus files."""
    import os
    data_dir = str(data_dir)
    
    if not os.path.exists(data_dir):
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        return

    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        console.print(f"[yellow]No .txt files found in {data_dir}[/yellow]")
        return

    if dry_run:
        console.print("[yellow]DRY RUN - No files will be modified[/yellow]\n")

    total_removed = 0
    total_kept = 0

    for filename in sorted(txt_files):
        file_path = os.path.join(data_dir, filename)
        
        sources = set()
        unique_lines = []
        duplicates = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split("\t")
                    src = parts[0] if parts else ""
                    if src in sources:
                        duplicates += 1
                    else:
                        sources.add(src)
                        unique_lines.append(line.strip())

        total_removed += duplicates
        total_kept += len(unique_lines)

        if duplicates > 0:
            action = "Would remove" if dry_run else "Removed"
            console.print(f"  {action} {duplicates:,} duplicates from {filename} (kept {len(unique_lines):,})")
            
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    for line in unique_lines:
                        f.write(line + "\n")

    console.print(f"\n[green]Total: {total_kept:,} unique sentences kept[/green]")
    if total_removed > 0:
        if dry_run:
            console.print(f"[yellow]Would remove {total_removed:,} duplicates[/yellow]")
        else:
            console.print(f"[green]Removed {total_removed:,} duplicates[/green]")


@list.command("files")
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing parallel corpus files",
)
def list_files(data_dir):
    """List corpus files with their sizes and sentence counts."""
    import os
    data_dir = str(data_dir)
    
    if not os.path.exists(data_dir):
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        return

    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        console.print(f"[yellow]No .txt files found in {data_dir}[/yellow]")
        return

    table = Table(title="Corpus Files")
    table.add_column("File", style="cyan")
    table.add_column("Languages", style="green")
    table.add_column("Sentences", justify="right", style="yellow")
    table.add_column("Size", justify="right", style="magenta")

    total_sentences = 0
    total_size = 0

    for filename in sorted(txt_files):
        file_path = os.path.join(data_dir, filename)
        stem = os.path.splitext(filename)[0]
        
        if "-" in stem:
            parts = stem.split("-")
            if len(parts) == 2:
                src, tgt = parts
                src_name = TARGET_LANGUAGES.get(src, {}).get("name", src.upper())
                tgt_name = TARGET_LANGUAGES.get(tgt, {}).get("name", tgt.upper())
                lang_pair = f"{src_name} -> {tgt_name}"
            else:
                lang_pair = stem
        else:
            lang_pair = stem

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sentence_count = sum(1 for line in f if line.strip())
        except Exception:
            sentence_count = 0

        size_bytes = os.path.getsize(file_path)
        total_size += size_bytes
        total_sentences += sentence_count

        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

        table.add_row(
            filename,
            lang_pair,
            f"{sentence_count:,}",
            size_str,
        )

    total_size_str = (
        f"{total_size / (1024 * 1024):.1f} MB"
        if total_size >= 1024 * 1024
        else f"{total_size / 1024:.1f} KB"
    )
    table.add_row(
        "",
        "",
        f"{total_sentences:,}",
        total_size_str,
        style="bold",
    )

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
    help="OpenRouter model (default: arcee-ai/trinity-large-preview:free)",
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
    languages_data = {}
    for r in all_results:
        languages_data[r["target"]] = {
            "metrics": r["metrics"],
            "sentences": r["sentences"],
            "samples": len(r["sentences"]),
        }

    # Calculate summary and rankings
    import statistics
    bleu_scores = [d["metrics"]["bleu"] for d in languages_data.values()]
    chrf_scores = [d["metrics"]["chrf"] for d in languages_data.values()]

    summary = {
        "num_languages": len(languages_data),
        "bleu": {
            "mean": statistics.mean(bleu_scores) if bleu_scores else 0,
            "median": statistics.median(bleu_scores) if bleu_scores else 0,
            "min": min(bleu_scores) if bleu_scores else 0,
            "max": max(bleu_scores) if bleu_scores else 0,
        },
        "chrf": {
            "mean": statistics.mean(chrf_scores) if chrf_scores else 0,
            "median": statistics.median(chrf_scores) if chrf_scores else 0,
            "min": min(chrf_scores) if chrf_scores else 0,
            "max": max(chrf_scores) if chrf_scores else 0,
        },
    }

    bleu_ranking = sorted(
        [(lang, d["metrics"]["bleu"], d["samples"]) for lang, d in languages_data.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    chrf_ranking = sorted(
        [(lang, d["metrics"]["chrf"], d["samples"]) for lang, d in languages_data.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    report_gen = MultiReportGenerator()
    report_gen.generate(
        results={
            "results": languages_data,
            "summary": summary,
            "rankings": {"bleu": bleu_ranking, "chrf": chrf_ranking},
        },
        output_file=output_file,
        model=model,
        samples_per_lang=samples,
        corpus="wikimedia",
    )

    console.print(f"\n[green]Report saved to: {output_file}[/green]")
    console.print(f"[green]Execution time: {execution_time:.2f}s[/green]")


if __name__ == "__main__":
    main()
