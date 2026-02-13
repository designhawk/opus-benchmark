# Developer Guide

This guide helps contributors understand the OPUS-Benchmark codebase and how to extend it.

## Project Structure

```
opus-benchmark/
├── src/
│   ├── api/
│   │   └── openrouter.py       # OpenRouter API client
│   ├── cli/
│   │   └── main.py            # CLI command definitions
│   ├── config/
│   │   └── languages.py       # Language code mappings
│   ├── data/
│   │   ├── __init__.py
│   │   └── zip_converter.py   # ZIP to TXT conversion
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # BLEU, chrF++, METEOR
│   │   └── multi_evaluator.py # Multi-language evaluation
│   ├── llm/
│   │   ├── __init__.py
│   │   └── translator.py       # Translation orchestration
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── html_report.py     # Single-language reports
│   │   └── multi_report.py    # Multi-language reports
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration management
├── data/
│   └── tatoeba/               # Parallel corpus files
├── reports/                   # Generated HTML reports
├── templates/                 # Jinja2 templates (if any)
├── .env.example               # Environment template
├── pyproject.toml            # Project config
├── requirements.txt           # Dependencies
└── README.md                  # Main documentation
```

## Key Components

### 1. API Layer (`src/api/`)

**Purpose**: Handle communication with translation providers.

**Files**:
- `openrouter.py` - OpenRouter API client

**Interface**:
```python
class TranslationClient:
    def __init__(self, api_key: str): ...
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str = None
    ) -> str: ...
```

**Adding a new provider**: See [API.md](API.md)

---

### 2. Translation Layer (`src/llm/`)

**Purpose**: Orchestrate translation workflow.

**Files**:
- `translator.py` - Main translation coordinator

**Key Class**: `LLMTranslator`

```python
class LLMTranslator:
    def __init__(self, api_key: str, model: str = "...", client_class=None):
        self.client = OpenRouterClient(api_key)  # or custom client
        self.model = model

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_samples: int
    ) -> Dict[int, str]: ...
```

**Flow**:
1. Receive list of source sentences
2. Call API client for each sentence
3. Apply rate limiting
4. Return translations with indices

---

### 3. Evaluation Layer (`src/evaluation/`)

**Purpose**: Calculate translation quality metrics.

**Files**:
- `metrics.py` - BLEU, chrF++, METEOR implementations
- `multi_evaluator.py` - Multi-language evaluation

**Key Class**: `TranslationEvaluator`

```python
class TranslationEvaluator:
    def evaluate(
        self,
        references: List[List[str]],  # [["ref1"], ["ref2"]]
        hypotheses: List[str]          # ["hyp1", "hyp2"]
    ) -> Dict[str, float]:             # {"bleu": 45.2, "chrf": 67.8, "meteor": 72.1}

    def evaluate_per_sentence(
        self,
        references: List[List[str]],
        hypotheses: List[str]
    ) -> List[Dict[str, float]]:      # [{"bleu": 45, "chrf": 67, "meteor": 72}, ...]
```

**Metrics**:
- **BLEU**: N-gram precision (sacrebleu)
- **chrF++**: Character n-gram F-score (sacrebleu)
- **METEOR**: Semantic similarity (NLTK)

**Adding a new metric**:
1. Add method to `TranslationEvaluator` class
2. Update `evaluate()` to include new metric
3. Update CLI and reports to display it

---

### 4. CLI Layer (`src/cli/`)

**Purpose**: Command-line interface using Click.

**File**: `main.py`

**Commands**:
| Command | Description |
|---------|-------------|
| `run` | Single language benchmark |
| `run-multi` | Multi-language benchmark |
| `convert` | ZIP to TXT conversion |
| `list files` | List corpus files |
| `list targets` | List available languages |
| `list duplicates` | Check for duplicates |
| `clean` | Remove duplicates |

**Adding a new command**:

```python
@main.command("my-command")
@click.option("--option", default="value")
def my_command(option):
    """Description of command."""
    # Your code here
    pass
```

**Command structure**:
1. Parse arguments with Click
2. Load data
3. Process (translate/evaluate)
4. Output results

---

### 5. Reporting Layer (`src/reporting/`)

**Purpose**: Generate HTML reports.

**Files**:
- `html_report.py` - Single-language report generator
- `multi_report.py` - Multi-language report generator

**Key Class**: `HTMLReportGenerator`

```python
class HTMLReportGenerator:
    def generate(
        self,
        results: Dict[str, Any],
        output_file: str
    ) -> str:  # Returns path to generated file
```

**Adding a new chart to reports**:
1. Add Plotly chart code to `_generate_charts()`
2. Embed in HTML template section
3. Pass required data in `results` dict

---

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/opus-benchmark.git
cd opus-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API key

# Test installation
python -m src.cli.main --help
```

### Running Tests

Currently, there are no formal tests. To manually test:

```bash
# Test single run
python -m src.cli.main run --target de --samples 5

# Test multi run
python -m src.cli.main run-multi --langs de,fr --samples 3

# Test data conversion
python -m src.cli.main convert

# Test list commands
python -m src.cli.main list files
python -m src.cli.main list targets
```

### Code Style

- **Python**: Follow PEP 8
- **Imports**: Use absolute imports from `src/`
- **Types**: Use type hints where helpful
- **Logging**: Use `logging.getLogger(__name__)` for module logging

**Example**:
```python
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MyClass:
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info(f"Initialized with key: {api_key[:8]}...")

    def process(self, data: List[str]) -> Dict[str, int]:
        """Process input data."""
        results = {}
        for item in data:
            results[item] = len(item)
        logger.debug(f"Processed {len(data)} items")
        return results
```

---

## Common Development Tasks

### Adding a New Language

1. Add to `src/config/languages.py`:
```python
TARGET_LANGUAGES = {
    "new": {"name": "New Language", "code": "new"},
    # ...
}
```

2. Download corpus:
   - Find on OPUS (https://opus.nlpl.eu/)
   - Place in `data/tatoeba/`
   - Run `python -m src.cli.main convert`

### Adding a New Metric

1. Implement in `src/evaluation/metrics.py`:
```python
def new_metric(self, references, hypotheses):
    # Implementation
    return score
```

2. Update `evaluate()` to include it:
```python
return {
    "bleu": self.bleu(...),
    "chrf": self.chrf(...),
    "meteor": self.meteor(...),
    "new_metric": self.new_metric(...),
}
```

3. Update CLI display in `src/cli/main.py`
4. Update reports in `src/reporting/`

### Modifying the Report

HTML reports use Plotly for charts. To modify:

1. Edit `src/reporting/html_report.py`
2. Look for Plotly figure construction
3. Add/modify traces:
```python
fig.add_trace(go.Bar(x=data, y=values))
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test API Independently

```python
from src.api.openrouter import OpenRouterClient

client = OpenRouterClient("YOUR_API_KEY")
result = client.translate("Hello", "en", "de")
print(result)
```

### Check Data Loading

```python
from src.cli.main import load_parallel_sentences

src, tgt = load_parallel_sentences("data/tatoeba/en-de.txt")
print(f"Loaded {len(src)} sentences")
print(f"Sample: {src[0]} -> {tgt[0]}")
```

---

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md (create if missing)
3. Commit with tag:
```bash
git tag -a v0.x.x -m "Release v0.x.x"
git push origin master --tags
```

---

## Getting Help

- Open an issue: https://github.com/designhawk/opus-benchmark/issues
- Check existing issues before creating new ones
- Provide environment details, error messages, and steps to reproduce
