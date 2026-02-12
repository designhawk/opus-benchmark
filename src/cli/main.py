"""Main CLI entry point for OPUS-LLM-Benchmark."""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.box import SIMPLE

from utils.config import Config
from data.corpus import OPUSCorpusDownloader
from llm.translator import LLMTranslator
from evaluation.metrics import TranslationEvaluator
from reporting.html_report import HTMLReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console(force_terminal=True)


def show_corpus_menu(
    corpora: list,
    title: str,
    source: str,
    target: str,
    show_all: bool = False,
) -> Optional[Dict[str, str]]:
    """Display corpus selection menu."""
    if not corpora:
        console.print("[red]No corpora found matching criteria.[/red]")
        return None

    console.print()
    count_info = (
        f"({len(corpora)} corpora)" if show_all else f"({len(corpora)} corpora < 1GB)"
    )
    panel_title = f"Select Corpus for {source.upper()} -> {target.upper()} {count_info}"

    console.print(
        Panel.fit(
            panel_title, style="cyan bold", subtitle="Choose from available options"
        )
    )
    console.print()

    for i, corp in enumerate(corpora[:10], 1):
        name = corp.get("name", "Unknown")
        version = corp.get("version", "")
        size = corp.get("size_formatted", "?")
        pairs = corp.get("alignment_pairs", 0)

        try:
            pairs_str = f"{int(pairs):,}"
        except (ValueError, TypeError):
            pairs_str = str(pairs)

        row = f"  [{i}] {name:<20} {size:<10}  ({pairs_str} pairs)"
        if version:
            if not version.startswith("v"):
                version = "v" + version
            row += f"  {version}"
        console.print(row)

    console.print()

    options = []
    for i in range(1, min(len(corpora), 10) + 1):
        options.append(str(i))

    if len(corpora) > 10:
        options.append("N")

    options.extend(["A", "S", "X"])

    menu_text = "  [1-10] Select corpus"
    if len(corpora) > 10:
        menu_text += "  [N] Next page"
    menu_text += "  [A] Show ALL corpora"
    menu_text += "  [S] Search by name"
    menu_text += "  [X] Cancel"

    console.print(menu_text)
    console.print()

    while True:
        choice = Prompt.ask("[cyan]Selection[/cyan]", choices=options, default="1")

        if choice.upper() == "X":
            console.print("[yellow]Cancelled.[/yellow]")
            sys.exit(0)

        if choice.upper() == "A":
            return {"id": "all", "name": "Show All"}

        if choice.upper() == "S":
            search_query = Prompt.ask("[cyan]Search corpus name[/cyan]", default="")
            if search_query:
                return {
                    "id": f"search:{search_query}",
                    "name": f"Search: {search_query}",
                }
            continue

        if choice.upper() == "N" and len(corpora) > 10:
            return show_corpus_menu(corpora[10:], title, source, target, show_all=True)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(corpora) and idx < 10:
                selected = corpora[idx]
                return {"id": selected["name"], "name": selected["name"]}
        except ValueError:
            pass

        console.print("[red]Invalid selection. Try again.[/red]")


def select_corpus_interactive(target: str, source: str = "en") -> Dict[str, str]:
    """Show interactive corpus selection menu."""
    downloader = OPUSCorpusDownloader()

    console.print(
        f"\n[cyan]Fetching corpora for {source.upper()} -> {target.upper()}...[/cyan]\n"
    )

    small_corpora = downloader.get_small_corpora(source, target, max_mb=1000)
    all_corpora = downloader.get_all_corpora_for_pair(source, target)

    if not small_corpora:
        console.print("[red]No small corpora found. Showing all available...[/red]")
        small_corpora = all_corpora

    result = show_corpus_menu(small_corpora, "", source, target, show_all=False)

    if result is None:
        sys.exit(1)

    if result["id"] == "all":
        result = show_corpus_menu(all_corpora, "", source, target, show_all=True)
        if result is None or result["id"] in ["all", "X"]:
            sys.exit(1)

    if result["id"].startswith("search:"):
        search_query = result["id"].split(":", 1)[1]
        search_results = downloader.search_corpora(source, target, search_query)
        result = show_corpus_menu(search_results, "", source, target, show_all=True)
        if result is None or result["id"] in ["all", "X"]:
            sys.exit(1)

    return result


def run_benchmark(
    corpus_choice: Dict[str, str],
    target: str,
    num_samples: int,
    model: str,
    source: str = "en",
):
    """Run the translation benchmark."""
    config = Config()
    if not config.validate():
        sys.exit(1)

    api_key = config.get_api_key()

    corpus_id = corpus_choice["id"]
    corpus_name = corpus_choice["name"]

    if corpus_id == "auto":
        corpus_display = "Auto (smallest available)"
    else:
        corpus_display = corpus_name

    console.print(
        Panel(
            Text(
                f"Starting Benchmark\n\n"
                f"Corpus: {corpus_display}\n"
                f"Languages: {source.upper()} -> {target.upper()}\n"
                f"Samples: {num_samples}\n"
                f"Model: {model}",
                style="cyan",
            ),
            title="OPUS-LLM-Benchmark",
            expand=False,
        )
    )

    start_time = time.time()

    try:
        downloader = OPUSCorpusDownloader()

        console.print("\n[bold cyan]Step 1: Downloading corpus...[/bold cyan]")
        result = downloader.download_with_fallback(
            preferred_corpus=corpus_id,
            source=source,
            target=target,
            num_samples=num_samples,
        )

        source_sentences, target_sentences = result["data"]
        used_corpus = result["corpus_used"]

        if result["used_fallback"]:
            console.print(
                f"[yellow]Note: Fell back to {used_corpus} (preferred: {corpus_name})[/yellow]"
            )

        if not source_sentences:
            console.print("[red]Error: No sentences downloaded[/red]")
            sys.exit(1)

        console.print(
            f"[green]Downloaded {len(source_sentences)} sentence pairs from {used_corpus}[/green]"
        )

        translator = LLMTranslator(api_key, model)

        console.print("\n[bold cyan]Step 2: Translating...[/bold cyan]")
        translations = translator.translate_batch(
            texts=source_sentences,
            source_lang=source,
            target_lang=target,
            num_samples=num_samples,
        )

        evaluator = TranslationEvaluator()

        console.print("\n[bold cyan]Step 3: Evaluating...[/bold cyan]")
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
            "corpus": used_corpus,
            "source": source,
            "target": target,
            "num_samples": num_samples,
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
            / f"benchmark_{source.upper()}-{target.upper()}_{used_corpus}_{timestamp}.html"
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

    # List of recommended free models
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


@list.command("corpora")
def list_corpora():
    """List available corpora."""
    downloader = OPUSCorpusDownloader()

    console.print("\n[cyan]Fetching corpora...[/cyan]\n")

    all_corpora = downloader.get_all_corpora_for_pair("en", "de")
    small_corpora = [c for c in all_corpora if c["size_bytes"] <= 1000 * 1024 * 1024]

    console.print(f"[cyan]Total corpora for EN-DE: {len(all_corpora)}[/cyan]")
    console.print(f"[green]Small corpora (< 1GB): {len(small_corpora)}[/green]\n")

    table = Table(title="Small Corpora (< 1GB)", box=SIMPLE)
    table.add_column("Corpus", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Pairs", style="white")
    table.add_column("Version", style="dim")

    for corp in small_corpora[:20]:
        pairs = corp.get("alignment_pairs", 0)
        try:
            pairs_str = f"{int(pairs):,}"
        except (ValueError, TypeError):
            pairs_str = str(pairs)
        table.add_row(
            corp["name"],
            corp["size_formatted"],
            pairs_str,
            corp.get("version", ""),
        )

    console.print(table)


@list.command("targets")
@click.option("--source", default="en", help="Source language code")
@click.option("--corpus", help="Filter by corpus")
def list_targets(source: str, corpus: Optional[str]):
    """List available target languages for a source."""
    downloader = OPUSCorpusDownloader()

    language_names = {
        "ar": "Arabic",
        "bg": "Bulgarian",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "es": "Spanish",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "he": "Hebrew",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "nb": "Norwegian",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "sv": "Swedish",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
        "zh": "Chinese",
    }

    if corpus:
        pairs = downloader.get_available_pairs(corpus, source)
        table = Table(title=f"Available Targets in {corpus} ({source.upper()})")
        table.add_column("Code", style="cyan")
        table.add_column("Language", style="green")
        for target in sorted(pairs):
            name = language_names.get(target.lower(), target)
            table.add_row(target.upper(), name)
        console.print(table)
    else:
        console.print("[cyan]Fetching available targets...[/cyan]\n")

        all_corpora = downloader.get_all_corpora_for_pair(source, "de")
        unique_targets = set()
        for corp in all_corpora:
            if corp.get("target"):
                unique_targets.add(corp["target"].upper())

        table = Table(title=f"Available Targets for {source.upper()}")
        table.add_column("Code", style="cyan")
        table.add_column("Language", style="green")

        for target in sorted(unique_targets):
            name = language_names.get(target.lower(), target)
            table.add_row(target.upper(), name)

        console.print(table)


@list.command("pairs")
@click.option("--source", default="en", help="Source language")
def list_pairs(source: str):
    """List available language pairs."""
    downloader = OPUSCorpusDownloader()

    console.print(f"\n[cyan]Fetching corpora for {source.upper()}...[/cyan]\n")

    all_corpora = downloader.get_all_corpora_for_pair(source, "de")

    corpora_info = {}
    for corp in all_corpora:
        name = corp.get("name", "Unknown")
        if name not in corpora_info:
            corpora_info[name] = corp

    small_corpora = [
        c for c in corpora_info.values() if c["size_bytes"] <= 1000 * 1024 * 1024
    ]

    table = Table(title=f"Available Pairs for {source.upper()} (small corpora < 1GB)")
    table.add_column("Corpus", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Pairs", style="white")

    for corp in small_corpora[:20]:
        pairs = corp.get("alignment_pairs", 0)
        try:
            pairs_str = f"{int(pairs):,}"
        except (ValueError, TypeError):
            pairs_str = str(pairs)
        table.add_row(
            corp["name"],
            corp["size_formatted"],
            pairs_str,
        )

    console.print(table)


@main.command()
@click.option("--target", required=True, help="Target language code")
@click.option(
    "--samples",
    default=10,
    type=click.Choice(["10", "100", "500", "1000"]),
    help="Number of samples",
)
@click.option(
    "--model",
    default="arcee-ai/trinity-large-preview:free",
    help="OpenRouter model",
)
def run(target: str, samples: str, model: str):
    """Run a translation benchmark with interactive corpus selection."""
    corpus_choice = select_corpus_interactive(target)
    run_benchmark(corpus_choice, target, int(samples), model)


@main.command()
@click.option("--input", help="Input results file or directory")
@click.option("--output", help="Output HTML file")
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


@main.group()
def download():
    """Download parallel corpora."""
    pass


@download.command("tatoeba")
@click.option(
    "--langs",
    default=None,
    help="Comma-separated language codes",
)
@click.option(
    "--all",
    is_flag=True,
    help="Download all available Tatoeba languages",
)
@click.option(
    "--max",
    type=int,
    default=None,
    help="Download only N smallest corpora (by sentence count)",
)
@click.option(
    "--parallel",
    default=5,
    type=int,
    help="Number of parallel downloads",
)
@click.option(
    "--output",
    default="./data/tatoeba",
    help="Output directory",
)
@click.option(
    "--no-external",
    is_flag=True,
    help="Don't use external downloaders",
)
def download_tatoeba(
    langs: Optional[str],
    all: bool,
    max: Optional[int],
    parallel: int,
    output: str,
    no_external: bool,
):
    """Download Tatoeba corpus (sorted by sentence count, smallest first)."""
    from data.tatoeba_downloader import TatoebaDownloader, get_tatoeba_languages

    if all:
        languages = get_tatoeba_languages()
        console.print(f"Found {len(languages)} languages with Tatoeba corpus")
    elif langs:
        languages = [l.strip().lower() for l in langs.split(",")]
    else:
        # Default to top 10 smallest
        languages = get_tatoeba_languages()[:10]
        console.print(
            f"No languages specified, downloading {len(languages)} smallest corpora"
        )

    downloader = TatoebaDownloader(
        output_dir=output,
        max_parallel=parallel,
        use_external=not no_external,
    )

    results = downloader.download_languages(languages, max_languages=max)

    if results["success"]:
        console.print("\n[green]OK All downloads completed successfully![/green]")
    else:
        failed = [
            lang for lang, success in results["downloaded"].items() if not success
        ]
        if failed:
            console.print(f"\n[yellow]⚠ Failed downloads: {', '.join(failed)}[/yellow]")


@download.command("fast")
@click.option(
    "--langs",
    default=None,
    help="Comma-separated language codes (default: de,fr,ja)",
)
@click.option(
    "--all",
    is_flag=True,
    help="Download all 32 languages",
)
@click.option(
    "--parallel",
    default=5,
    type=int,
    help="Number of parallel downloads",
)
@click.option(
    "--output",
    default="./data/wikimedia",
    help="Output directory",
)
@click.option(
    "--no-external",
    is_flag=True,
    help="Don't use external downloaders (aria2c, wget, curl)",
)
def download_fast(
    langs: Optional[str], all: bool, parallel: int, output: str, no_external: bool
):
    """Fast parallel download using aria2c/wget/curl (recommended)."""
    from data.fast_downloader import FastDownloader
    from config.languages import DEFAULT_LANGUAGES

    if all:
        languages = DEFAULT_LANGUAGES
    elif langs:
        languages = [l.strip() for l in langs.split(",")]
    else:
        languages = ["de", "fr", "ja"]

    downloader = FastDownloader(
        output_dir=output,
        max_parallel=parallel,
        use_external=not no_external,
    )

    results = downloader.download_languages(languages)

    if results["success"]:
        console.print("\n[green]OK All downloads completed successfully![/green]")
    else:
        failed = [
            lang for lang, success in results["downloaded"].items() if not success
        ]
        if failed:
            console.print(f"\n[yellow]⚠ Failed downloads: {', '.join(failed)}[/yellow]")


@download.command("multi-lang")
@click.option(
    "--samples",
    default=10,
    type=int,
    help="Samples per language to download",
)
@click.option(
    "--corpus",
    default="wikimedia",
    help="Corpus name to download from",
)
@click.option(
    "--langs",
    default=None,
    help="Comma-separated language codes (default: all 32)",
)
@click.option(
    "--output",
    default="./data/wikimedia",
    help="Output directory for downloaded files",
)
def download_multi_lang(samples: int, corpus: str, langs: Optional[str], output: str):
    """Download parallel corpora for multiple languages (legacy)."""
    from data.multi_downloader import MultiLangDownloader

    languages = None
    if langs:
        languages = [l.strip() for l in langs.split(",")]

    downloader = MultiLangDownloader(output_dir=output)
    results = downloader.download_all_languages(
        corpus=corpus,
        languages=languages,
        samples_per_lang=samples,
    )


@download.command("cleanup")
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory containing dataset files",
)
def download_cleanup(data_dir: str):
    """Remove duplicate sentences from all datasets."""
    from data.tatoeba_downloader import cleanup_all_datasets

    results = cleanup_all_datasets(data_dir)

    total_original = sum(orig for orig, _ in results.values())
    total_unique = sum(uniq for _, uniq in results.values())
    total_removed = total_original - total_unique

    if total_removed > 0:
        console.print(
            f"\n[green]Successfully removed {total_removed:,} duplicate sentences total[/green]"
        )
        console.print(f"Total sentences: {total_original:,} -> {total_unique:,}")
    else:
        console.print("\n[dim]No duplicates found in any dataset[/dim]")


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
    help="Comma-separated language codes (default: all 32)",
)
@click.option(
    "--corpus",
    default="tatoeba",
    help="Corpus name (default: tatoeba)",
)
@click.option(
    "--data-dir",
    default="./data/tatoeba",
    help="Directory with downloaded parallel corpora (default: ./data/tatoeba)",
)
@click.option(
    "--checkpoint",
    default="./data/wikimedia/checkpoint.json",
    help="Checkpoint file for resume",
)
@click.option(
    "--output",
    default=None,
    help="Output HTML file (auto-generated if not specified)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from checkpoint",
)
def run_multi(
    samples: int,
    model: str,
    langs: Optional[str],
    corpus: str,
    data_dir: str,
    checkpoint: str,
    output: Optional[str],
    resume: bool,
):
    """Run translation benchmark across multiple languages."""
    import time
    from pathlib import Path as FilePath
    from config.languages import TARGET_LANGUAGES, DEFAULT_LANGUAGES
    from data.multi_downloader import MultiLangDownloader
    from llm.translator import LLMTranslator
    from evaluation.multi_evaluator import MultiLangEvaluator, CheckpointManager
    from reporting.multi_report import MultiReportGenerator
    from utils.config import Config

    config = Config()
    if not config.validate():
        sys.exit(1)

    api_key = config.get_api_key()

    languages = None
    if langs:
        languages = [l.strip().lower() for l in langs.split(",")]
    else:
        languages = DEFAULT_LANGUAGES.copy()

    console.print(
        Panel.fit(
            f"Multi-Language Translation Benchmark\n\n"
            f"Languages: {len(languages)}\n"
            f"Samples: {samples}/language (random selection)\n"
            f"Model: {model}\n"
            f"Resume: {'Yes' if resume else 'No'}",
            style="cyan bold",
        )
    )

    start_time = time.time()

    downloader = MultiLangDownloader(output_dir=data_dir)
    translator = LLMTranslator(api_key, model)
    evaluator = MultiLangEvaluator()
    report_gen = MultiReportGenerator()

    checkpoint_mgr = CheckpointManager(checkpoint_file=checkpoint)

    if resume and checkpoint_mgr.data.get("status") == "in_progress":
        console.print("[yellow]Resuming from checkpoint...[/yellow]")
        pending_langs = checkpoint_mgr.get_pending_languages()
        if pending_langs:
            console.print(f"Pending languages: {len(pending_langs)}")
        languages = pending_langs if pending_langs else languages
    else:
        config_dict = {
            "model": model,
            "samples": samples,
            "corpus": corpus,
        }
        checkpoint_mgr.init_benchmark(config_dict, languages)

    completed = 0
    total = len(languages)

    for i, target in enumerate(languages, 1):
        lang_info = TARGET_LANGUAGES.get(target, {"name": target})
        console.print(
            f"\n[{i}/{total}] Processing {lang_info['name']} ({target.upper()})..."
        )

        checkpoint_mgr.update_language(target, "in_progress", samples=samples)

        source_lines, target_lines = downloader.load_language_pair(
            target, max_samples=samples
        )

        if not source_lines or not target_lines:
            console.print(f"  [yellow]No data available for {target.upper()}[/yellow]")
            checkpoint_mgr.update_language(target, "failed", samples=0)
            continue

        console.print(f"  [cyan]Translating {len(source_lines)} sentences...[/cyan]")

        translations = translator.translate_batch(
            texts=source_lines,
            source_lang="en",
            target_lang=target,
            num_samples=samples,
        )

        if not translations:
            console.print(f"  [red]Translation failed for {target.upper()}[/red]")
            checkpoint_mgr.update_language(target, "failed", samples=len(source_lines))
            continue

        console.print(f"  [cyan]Evaluating...[/cyan]")

        references = [[ref] for ref in target_lines]
        hypotheses = [translations.get(j, "") for j in range(len(source_lines))]

        result = evaluator.evaluate_language(
            target_lang=target,
            references=references,
            hypotheses=hypotheses,
        )

        sentence_data = []
        for j in range(len(source_lines)):
            sent_metrics = (
                result["sentence_metrics"][j]
                if j < len(result["sentence_metrics"])
                else {}
            )
            sentence_data.append(
                {
                    "source": source_lines[j],
                    "reference": target_lines[j],
                    "hypothesis": translations.get(j, ""),
                    "bleu": sent_metrics.get("bleu", 0),
                    "chrf": sent_metrics.get("chrf", 0),
                }
            )

        checkpoint_mgr.update_language(
            target,
            "completed",
            metrics=result["metrics"],
            samples=len(source_lines),
            result_data=sentence_data,
        )

        completed += 1
        console.print(
            f"  [green]Done: BLEU={result['metrics']['bleu']:.1f}, chrF={result['metrics']['chrf']:.1f}[/green]"
        )

    execution_time = time.time() - start_time

    checkpoint_mgr.mark_complete()

    console.print("\n[bold cyan]Benchmark Complete![/bold cyan]")

    results = evaluator.to_dict()
    results["config"] = {
        "model": model,
        "samples": samples,
        "corpus": corpus,
        "languages": len(languages),
        "execution_time": f"{execution_time:.2f}s",
    }

    if output:
        output_file = output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"./reports/multi-benchmark_{timestamp}.html"

    report_gen.generate(
        results=results,
        output_file=output_file,
        model=model,
        samples_per_lang=samples,
        corpus=corpus,
    )

    console.print(f"\n[green]Report saved to: {output_file}[/green]")
    console.print(f"[green]Execution time: {execution_time:.2f}s[/green]")


if __name__ == "__main__":
    main()
