# OPUS-LLM-Benchmark

A CLI tool to benchmark LLM translations using local parallel corpora via OpenRouter API.

## Features

- **Multiple Language Support**: Translate from English to 30+ target languages
- **Flexible Model Selection**: Use OpenRouter's free models or select specific models
- **Multiple Sample Sizes**: Configure any number of sentences
- **Comprehensive Metrics**: BLEU, chrF++, and METEOR scoring
- **Interactive Reports**: HTML reports with charts and visualizations (auto-generated after runs)
- **Rich Terminal Output**: Real-time progress and results display
- **Automatic Deduplication**: Duplicate sentences removed during conversion

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opus-benchmark.git
cd opus-benchmark

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Configure via .env file

Create a `.env` file in the project root:

```bash
# Required: OpenRouter API key
OPENROUTER_API_KEY=your_api_key_here

# Optional: Default model (defaults to arcee-ai/trinity-large-preview:free)
DEFAULT_MODEL=openai/gpt-4o-mini
```

Get your API key from [https://openrouter.ai/](https://openrouter.ai/)

### Prepare your data

```bash
# Drop ZIP files in data/tatoeba/
# ZIP should contain: Tatoeba.cs-en.cs, Tatoeba.cs-en.en, etc.

# Convert ZIP to TXT format (also removes duplicates)
python -m src.cli.main convert
```

### Run a benchmark

```bash
# Run single language benchmark
python -m src.cli.main run --target de

# Run multi-language benchmark (all pairs in data/tatoeba/)
python -m src.cli.main run-multi

# Specify custom model
python -m src.cli.main run --target de --model openai/gpt-4o-mini

# Specify sample size
python -m src.cli.main run --target de --samples 100
```

### View available data

```bash
# List corpus files with sizes and sentence counts
python -m src.cli.main list files

# List available target languages
python -m src.cli.main list targets

# Check for duplicate sentences
python -m src.cli.main list duplicates
```

## Commands

### convert

Convert ZIP files to parallel corpus format. Automatically removes duplicate sentences.

```bash
python -m src.cli.main convert

# Specify data directory
python -m src.cli.main convert --data-dir ./data/tatoeba
```

### list

List available resources.

```bash
# Show corpus files with sizes and sentence counts
python -m src.cli.main list files

# Show available target languages
python -m src.cli.main list targets

# Check for duplicate sentences
python -m src.cli.main list duplicates
```

### clean

Remove duplicate sentences from existing corpus files.

```bash
# Preview what would be removed
python -m src.cli.main clean --dry-run

# Actually remove duplicates
python -m src.cli.main clean
```

### run

Run a translation benchmark for a single language pair.

```bash
python -m src.cli.main run --target de --samples 10
```

Options:
- `--target` (required): Target language code (e.g., de, fr, ja)
- `--samples`: Number of sentences (default: 10)
- `--model`: OpenRouter model (default: arcee-ai/trinity-large-preview:free)
- `--source`: Source language code (default: en)
- `--data-dir`: Directory containing corpus files (default: ./data/tatoeba)

### run-multi

Run translation benchmarks across multiple languages.

```bash
python -m src.cli.main run-multi --samples 10 --langs de,fr,es
```

Options:
- `--samples`: Samples per language (default: 10)
- `--model`: OpenRouter model (default: arcee-ai/trinity-large-preview:free)
- `--langs`: Comma-separated language codes (default: all available)
- `--source`: Source language code (default: en)
- `--data-dir`: Directory with corpus files (default: ./data/tatoeba)
- `--output`: Output HTML file

## Supported Languages

Source: English (EN)

Targets:
- AR, BG, CS, DA, DE, EL, ES, ET, FI, FR
- HE, HU, ID, IT, JA, KO, LT, LV, NB, NL
- PL, PT, RO, RU, SK, SL, SV, TH, TR, UK
- VI, ZH

## Data Format

Place ZIP files containing OPUS parallel corpora in `data/tatoeba/`. Each ZIP should contain:
- `Tatoeba.langpair.source` (e.g., `Tatoeba.cs-en.cs`)
- `Tatoeba.langpair.target` (e.g., `Tatoeba.cs-en.en`)

The `convert` command transforms these into simple TXT files (one sentence per line) and removes duplicate sentences.

## Metrics

OPUS-Benchmark evaluates translations using three metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| BLEU | 0-100 | N-gram overlap with reference |
| chrF++ | 0-100 | Character n-gram F-score |
| METEOR | 0-100 | Semantic similarity with synonym matching |

For detailed information on how each metric works, see [SCORING.md](SCORING.md).

## Reports

HTML reports are automatically generated after each benchmark run and saved to the `reports/` directory.

- Single benchmark: `reports/benchmark_{source}-{target}_{corpus}_{timestamp}.html`
- Multi benchmark: `reports/multi-benchmark_{timestamp}.html`

Reports include:
- BLEU/chrF/METEOR score distributions
- Correlation charts
- Per-sentence results table with all metrics
- Language rankings (for multi-benchmark)

## Environment Variables

Configure via `.env` file:

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
DEFAULT_MODEL=arcee-ai/trinity-large-preview:free
```

## Sample Size & Time

| Samples | Approx. Time (Free Model) |
|---------|--------------------------|
| 10 | ~30 seconds |
| 50 | ~2 minutes |
| 100 | ~5 minutes |

## Documentation

- [API Guide](API.md) - Customize translation providers (DeepL, Google Translate, etc.)
- [Scoring Explained](SCORING.md) - How BLEU, chrF++, METEOR work
- [Developer Guide](DEVELOPER.md) - Contributing and extending the project

## License

MIT License
