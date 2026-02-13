# Translation Benchmark Scoring

This document explains how OPUS-Benchmark evaluates translation quality using BLEU, chrF++, and METEOR metrics.

## Overview

OPUS-Benchmark uses three complementary metrics to evaluate translations:

| Metric | Type | Best For |
|--------|------|----------|
| BLEU | N-gram precision | Overall quality |
| chrF++ | Character n-gram | Morphologically rich languages |
| METEOR | Semantic similarity | Synonym handling |

All metrics range from 0-100, where higher scores indicate better quality.

---

## BLEU (Bilingual Evaluation Understudy)

### What It Measures

BLEU measures how many n-grams (word sequences) in the translation match the reference translation. It's fundamentally a precision-based metric.

### How It Works

1. **Count N-grams**: Break both hypothesis and reference into n-grams (typically 1 to 4 words)

2. **Calculate Precision**: For each n-gram level:
   ```
   1-gram: "the" appears 3 times in both = 3 matches
   2-gram: "the cat" appears 1 time = 1 match
   ```

3. **Apply Brevity Penalty**: Prevent short translations from getting artificially high scores:
   ```
   BP = exp(1 - ref_len / hyp_len)  if hyp_len < ref_len
   BP = 1.0                           otherwise
   ```

4. **Calculate Final Score**:
   ```
   BLEU = BP × exp(mean(log(precision_n)))
   ```

### Example

```
Reference: "The cat is sitting on the mat"
Hypothesis: "The cat sat on the mat"

1-gram precision: 4/4 = 1.0
2-gram precision: 2/3 = 0.67
3-gram precision: 1/2 = 0.5
4-gram precision: 0/1 = 0.0

BLEU ≈ 18.2 (with brevity penalty)
```

### Strengths & Limitations

| Strengths | Limitations |
|-----------|-------------|
| Widely accepted standard | Doesn't capture meaning |
| Fast to compute | Requires high-quality references |
| Language-independent | Penalizes valid synonyms |
| Good for sentence ordering | No semantic understanding |

---

## chrF++ (Character n-gram F-score)

### What It Measures

chrF++ measures character-level n-gram overlap between translation and reference. It's more flexible than word-based metrics because it handles morphological variations better.

### How It Works

1. **Character N-grams**: Break text into character sequences:
   ```
   "cat" → "c", "ca", "cat", "at"
   ```

2. **Calculate F-score**: Combine precision and recall at character level:
   ```
   F = 2 × (precision × recall) / (precision + recall)
   ```

3. **The ++ Version**: Adds word boundary optimization to better handle word-level accuracy

### Why chrF++ Is Better for Some Languages

| Language Type | Example | Why chrF++ Helps |
|--------------|---------|------------------|
| German | "Übersetzung" | Compound words |
| Finnish | "käännös" | Agglutinative morphology |
| Hungarian | "fordítás" | Rich inflection |
| Russian | "перевод" | Case markings |

### Strengths & Limitations

| Strengths | Limitations |
|-----------|-------------|
| Handles morphology better | Less intuitive than BLEU |
| Less sensitive to word order | Not as widely used |
| Good for related languages | Can over-penalize synonyms |
| Matches human judgment better | - |

---

## METEOR (Metric for Evaluation of Translation with Explicit ORdering)

### What It Measures

METEOR measures semantic similarity by creating word-level alignments between hypothesis and reference, considering exact matches, stems, and synonyms.

### How It Works

1. **Create Alignments**: Map words in hypothesis to words in reference:
   ```
   Reference: "The cat is sitting on the mat"
   Hypothesis: "A cat sits on the rug"
   
   Alignment:
   The ↔ The (exact)
   cat ↔ cat (exact)
   is sitting ↔ sits (stem match)
   on ↔ on (exact)
   the ↔ the (exact)
   mat ↔ rug (synonym match?)
   ```

2. **Calculate Scores**:
   ```
   Precision = matches / hypothesis words
   Recall = matches / reference words
   F-mean = 2 × P × R / (P + R)
   ```

3. **Apply Penalties**:
   ```
   Fragmentation penalty = 0.5 × (unigrams - 1) / unigrams
   Final = F-mean × (1 - fragmentation penalty)
   ```

### Matching Types (in order of weight)

1. **Exact** (highest weight): "cat" ↔ "cat"
2. **Stem** (medium weight): "sitting" ↔ "sit"
3. **Synonym** (lower weight): "mat" ↔ "rug"

### Strengths & Limitations

| Strengths | Limitations |
|-----------|-------------|
| Captures semantics | Slower to compute |
| Handles synonyms | Requires WordNet/synonym dict |
| Better human correlation | Can miss context |
| More lenient than BLEU | - |

---

## Score Interpretation

### General Guidelines

| BLEU | chrF++ | METEOR | Quality Level |
|------|--------|--------|---------------|
| < 10 | < 30 | < 30 | **Poor** - Major meaning errors |
| 10-30 | 30-50 | 30-50 | **Fair** - Partial translation |
| 30-50 | 50-70 | 50-70 | **Good** - Mostly accurate |
| 50-70 | 70-85 | 70-85 | **Very Good** - High quality |
| > 70 | > 85 | > 85 | **Excellent** - Near-human |

### Language-Specific Baselines

Different language pairs have different typical score ranges:

| Pair | Expected BLEU | Notes |
|------|---------------|-------|
| en-de | 30-50 | Germanic - similar structure |
| en-fr | 30-50 | Romance - related languages |
| en-ru | 15-35 | Different script |
| en-ja | 10-25 | Completely different |
| en-zh | 15-30 | No shared script |

**Note**: These are rough guidelines. Actual scores depend on:
- Translation model quality
- Domain of text
- Reference quality
- Language pair similarity

---

## How OPUS-Benchmark Uses These Metrics

### Per-Sentence Scoring

Each translated sentence gets individual scores:

```python
for sentence in translations:
    bleu = evaluator._sentence_bleu(reference, hypothesis)
    chrf = evaluator._sentence_chrf(reference, hypothesis)
    meteor = evaluator._sentence_meteor(reference, hypothesis)
```

### Corpus-Level Scoring

Aggregated across all sentences in a language pair:

```python
corpus_bleu = evaluator.bleu(all_references, all_hypotheses)
corpus_chrf = evaluator.chrf(all_references, all_hypotheses)
corpus_meteor = evaluator.meteor(all_references, all_hypotheses)
```

### What Each Metric Tells You

| Metric | Best Use Case |
|--------|---------------|
| **BLEU** | Quick quality check, competitive benchmarking |
| **chrF++** | Languages with rich morphology, word order variations |
| **METEOR** | Semantic quality, synonym handling assessment |

### Report Visualizations

The HTML reports include:

1. **Score Distributions**: Histograms showing score spread
2. **Correlations**: BLEU vs chrF++, BLEU vs METEOR
3. **Per-Sentence Table**: Individual scores for each translation

---

## Technical Implementation

### Libraries Used

- **BLEU**: [sacrebleu](https://github.com/mjpost/sacrebleu) - Standard BLEU implementation
- **chrF++**: [sacrebleu](https://github.com/mjpost/sacrebleu) - Character n-gram F-score
- **METEOR**: [NLTK](https://www.nltk.org/) - WordNet-based Meteor

### Dependencies

```python
import sacrebleu          # For BLEU and chrF++
import nltk              # For METEOR (WordNet, tokenizers)
```

### First-Run Setup

METEOR requires NLTK data downloads (automatic on first run):

```python
# These are downloaded automatically if missing
nltk.download('punkt')        # Tokenizer
nltk.download('punkt_tab')    # Tokenizer data
nltk.download('wordnet')       # Synonym dictionary
nltk.download('omw-1.4')      # Multi-language WordNet
```

---

## Choosing the Right Metric

### Use BLEU when:
- Comparing models on standard benchmarks
- You have high-quality references
- Quick, standard evaluation

### Use chrF++ when:
- Working with morphologically rich languages (German, Finnish, Turkish)
- Word order varies significantly
- You want more human-correlated scores

### Use METEOR when:
- Semantic quality matters more than exact wording
- Translations may use synonyms
- You want the most lenient, human-like assessment

### Recommended: Use All Three

Each metric captures different aspects of translation quality. For comprehensive evaluation:
- Report all three metrics
- Look at correlations between them
- Pay attention to outlier sentences
