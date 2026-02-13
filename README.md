# OPUS-LLM-Benchmark

A CLI tool to benchmark LLM translations using local parallel corpora via OpenRouter API.

## Features

- **Multiple Language Support**: Translate from English to 32 target languages
- **Flexible Model Selection**: Use OpenRouter's free router or select specific models
- **Multiple Sample Sizes**: 10, 100, 500, or 1000 sentences
- **Comprehensive Metrics**: BLEU, chrF++, and METEOR scoring
- **Interactive Reports**: HTML reports with charts and visualizations
- **Rich Terminal Output**: Real-time progress and results display

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opus-benchmark.git
cd opus-benchmark

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Set your API key
```bash
export OPENROUTER_API_KEY=your_api_key_here

# Or set it interactively
opus-benchmark config set-api-key

# Python equivalent
python -m src.cli.main config set-api-key
```

### Prepare your data
```bash
# Drop ZIP files in data/tatoeba/
# ZIP should contain: Tatoeba.cs-en.cs, Tatoeba.cs-en.en, etc.

# Convert ZIP to TXT format
opus-benchmark convert

# Python equivalent
python -m src.cli.main convert
```

### Run a benchmark
```bash
# Run benchmark (uses data/tatoeba/en-de.txt)
opus-benchmark run --target de

# Multi-language benchmark (all pairs in data/tatoeba/)
opus-benchmark run-multi

# Python equivalents
python -m src.cli.main run --target de
python -m src.cli.main run-multi
```

## Commands

### Convert
```bash
# Convert ZIP files to parallel corpus format
opus-benchmark convert

# Python equivalent
python -m src.cli.main convert
```

### Configuration
```bash
# Set API key
opus-benchmark config set-api-key

# View configuration
opus-benchmark config show

# Set default model
opus-benchmark config set-model openai/gpt-4o-mini

# Python equivalents
python -m src.cli.main config set-api-key
python -m src.cli.main config show
python -m src.cli.main config set-model openai/gpt-4o-mini
```

### Benchmark
```bash
# Run benchmark with options
opus-benchmark run \
    --target de \
    --source en \
    --samples 100 \
    --model openrouter

# Run all language pairs
opus-benchmark run-multi

# Python equivalents
python -m src.cli.main run \
    --target de \
    --source en \
    --samples 100 \
    --model openrouter

python -m src.cli.main run-multi
```

### Reporting
```bash
# Generate HTML report
opus-benchmark report

# Compare models
opus-benchmark compare \
    --model-a openrouter \
    --model-b openai/gpt-4o-mini \
    --pairs en-de en-fr \
    --samples 10

# Python equivalents
python -m src.cli.main report

python -m src.cli.main compare \
    --model-a openrouter \
    --model-b openai/gpt-4o-mini \
    --pairs en-de en-fr \
    --samples 10
```

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

The `convert` command transforms these into simple TXT files (one sentence per line) for benchmarking.

## Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| BLEU | 0-100 | N-gram overlap with reference |
| chrF++ | 0-100 | Character n-gram F-score |
| METEOR | 0-100 | Semantic similarity with synonym matching |

## Sample Size Options

| Size | Time (Free) | Use Case |
|------|------------|----------|
| 10 | ~30s | Smoke test |
| 100 | ~5min | Development |
| 500 | ~25min | Standard evaluation |
| 1000 | ~50min | Full benchmark |

## Configuration

Configuration is stored in `~/.config/opus-benchmark/config.yaml`:

```yaml
api_key: your_openrouter_api_key
default_model: openrouter
```

## License

MIT License
