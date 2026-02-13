# OPUS-LLM-Benchmark Project Summary

## Project Overview
Multi-language translation benchmarking tool using OPUS parallel corpora and OpenRouter API.

## Features Implemented

### 1. Multi-Language Support
- **30+ target languages** from English
- Tatoeba corpus integration
- Automatic duplicate removal during conversion
- Random sentence sampling (not sequential)

### 2. Translation & Evaluation
- OpenRouter API integration (free tier support)
- Rate limiting with automatic retry
- BLEU, chrF++, METEOR metrics
- Random sentence selection for unbiased evaluation

### 3. Reporting
- Interactive HTML reports with Plotly charts
- Score distributions and correlations
- Per-sentence results table
- Language rankings (for multi-benchmark)
- Auto-generated after each run
- All three metrics displayed (BLEU, chrF++, METEOR)

### 4. CLI Commands
- `convert` - Convert ZIP files to TXT format (with deduplication)
- `clean` - Remove duplicate sentences from existing files
- `list files` - Show corpus files with sizes and sentence counts
- `list targets` - Show available target languages
- `list duplicates` - Check for duplicate sentences
- `run` - Run single language benchmark
- `run-multi` - Run multi-language benchmark

## Project Structure

```
opus-benchmark/
├── src/
│   ├── api/              # API clients (OpenRouter)
│   ├── cli/              # Command-line interface
│   ├── config/           # Language configurations
│   ├── data/             # Data converters and loaders
│   ├── evaluation/       # Metrics calculation
│   ├── llm/              # Translation orchestration
│   ├── reporting/       # HTML report generation
│   └── utils/            # Utility functions
├── data/
│   └── tatoeba/         # Parallel corpora files
├── reports/              # Generated HTML reports
├── API.md               # Custom API integration guide
├── SCORING.md           # Metrics explanation
├── DEVELOPER.md         # Contributor guide
├── .env                 # Environment variables (API key)
├── pyproject.toml       # Project configuration
├── requirements.txt     # Dependencies
└── README.md           # Main documentation
```

## Key Commands

```bash
# Convert ZIP files to TXT format (with deduplication)
python -m src.cli.main convert

# Check for duplicates
python -m src.cli.main list duplicates

# Remove duplicates
python -m src.cli.main clean --dry-run
python -m src.cli.main clean

# List corpus files
python -m src.cli.main list files

# List available targets
python -m src.cli.main list targets

# Run single language benchmark
python -m src.cli.main run --target de --samples 10

# Run multi-language benchmark
python -m src.cli.main run-multi --samples 10
```

## Configuration

Create `.env` file:
```env
OPENROUTER_API_KEY=your_api_key_here
DEFAULT_MODEL=arcee-ai/trinity-large-preview:free
```

## Data Statistics

- Total unique sentences: ~2.6M
- Languages available: 29+ (ar, bg, cs, da, de, el, es, et, fi, fr, he, hu, id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sl, sv, th, tr, uk, vi)
- Files stored in: `data/tatoeba/`

## Technical Details

### Random Sampling
- Sentences randomly selected from entire corpus
- Different samples each run
- Prevents bias toward beginning of files

### Duplicate Removal
- Removes duplicate sentences during conversion
- Based on exact source text match
- Preserves first occurrence

### Rate Limiting
- 1-second delay between API calls
- Automatic retry on 429 errors
- Exponential backoff

### Metrics
- **BLEU**: N-gram precision (sacrebleu)
- **chrF++**: Character n-gram F-score (sacrebleu)
- **METEOR**: Semantic similarity (NLTK WordNet)

## Ready for Production

All systems operational:
- ✅ CLI commands functional
- ✅ Data conversion with deduplication
- ✅ Random sampling implemented
- ✅ HTML reports generating (with all 3 metrics)
- ✅ Configuration via .env file

## Documentation

- [README.md](README.md) - Main documentation
- [API.md](API.md) - Custom translation API guide
- [SCORING.md](SCORING.md) - Metrics explanation
- [DEVELOPER.md](DEVELOPER.md) - Contributor guide

## Next Steps (Optional)

1. Add more languages to `data/tatoeba/`
2. Run full benchmark across all languages
3. Compare different models
4. Export results to CSV/JSON
5. Add more translation API providers (DeepL, Google Translate, etc.)

---
Generated: 2026-02-13
