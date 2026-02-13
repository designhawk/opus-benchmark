"""ZIP to parallel corpus converter for Tatoeba data."""

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def parse_tatoeba_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Parse Tatoeba.en-es.en -> ('en', 'es')"""
    # Match pattern: Tatoeba.XX-YY.lang
    match = re.match(r"Tatoeba\.(\w+)-(\w+)\.(\w+)", filename)
    if match:
        lang1, lang2, suffix = match.groups()
        # Return consistently ordered pair (alphabetically sorted)
        if lang1 < lang2:
            return (lang1, lang2)
        else:
            return (lang2, lang1)
    return None


def find_language_pairs(
    zip_path: Path,
) -> Dict[Tuple[str, str], Tuple[Optional[Path], Optional[Path]]]:
    """Find all language pairs in a ZIP file."""
    pairs = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        files = zf.namelist()
        lang_files = {}

        for f in files:
            if f.startswith("Tatoeba.") and not f.endswith("/"):
                # Extract language from suffix (e.g., Tatoeba.en-es.en -> en)
                match = re.match(r"Tatoeba\.(\w+)-(\w+)\.(\w+)", f)
                if match:
                    lang1, lang2, suffix = match.groups()
                    # Sort the pair consistently
                    key = tuple(sorted([lang1, lang2]))
                    lang_files.setdefault(key, [None, None])
                    # 0 = source file (first lang), 1 = target file (second lang)
                    if suffix == key[0]:
                        lang_files[key][0] = Path(f)
                    elif suffix == key[1]:
                        lang_files[key][1] = Path(f)

        # Only keep complete pairs
        for key, (src, tgt) in lang_files.items():
            if src and tgt:
                pairs[key] = (src, tgt)

    return pairs


def convert_zip(zip_path: Path, output_dir: Path) -> List[Tuple[str, str, str]]:
    """Convert a ZIP file to parallel corpus format.

    Returns list of (output_filename, source_lang, target_lang) for each conversion.
    """
    converted = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        pairs = find_language_pairs(zip_path)

        for (src_lang, tgt_lang), (src_file, tgt_file) in pairs.items():
            # Read source sentences
            with zf.open(str(src_file)) as f:
                src_lines = [line.decode("utf-8").strip() for line in f if line.strip()]

            # Read target sentences
            with zf.open(str(tgt_file)) as f:
                tgt_lines = [line.decode("utf-8").strip() for line in f if line.strip()]

            # Ensure equal length (use minimum)
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]

            # Remove duplicates based on exact source text match (keep first occurrence)
            seen_sources = set()
            unique_src_lines = []
            unique_tgt_lines = []
            for src, tgt in zip(src_lines, tgt_lines):
                if src not in seen_sources:
                    seen_sources.add(src)
                    unique_src_lines.append(src)
                    unique_tgt_lines.append(tgt)

            src_lines = unique_src_lines
            tgt_lines = unique_tgt_lines

            original_count = min_len
            duplicate_count = original_count - len(src_lines)

            # Write as TAB-separated: source[TAB]target
            output_file = output_dir / f"{src_lang}-{tgt_lang}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for src, tgt in zip(src_lines, tgt_lines):
                    f.write(f"{src}\t{tgt}\n")

            converted.append((f"{src_lang}-{tgt_lang}.txt", src_lang, tgt_lang))
            if duplicate_count > 0:
                print(f"  OK {src_lang}-{tgt_lang}.txt: {len(src_lines)} sentence pairs ({duplicate_count} duplicates removed)")
            else:
                print(f"  OK {src_lang}-{tgt_lang}.txt: {len(src_lines)} sentence pairs")

    return converted


def convert_all(data_dir: Optional[Path] = None) -> List[Tuple[str, int]]:
    """Convert all ZIP files in data/tatoeba/ directory.

    Returns: List of (filename, sentence_count) for each converted file.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "tatoeba"

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return []

    zip_files = list(data_dir.glob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {data_dir}")
        return []

    print(f"Found {len(zip_files)} ZIP file(s) in {data_dir}")
    print()

    results = []

    for zip_file in zip_files:
        print(f"Processing: {zip_file.name}")
        converted = convert_zip(zip_file, data_dir)

        for filename, src_lang, tgt_lang in converted:
            # Count lines in output file
            output_file = data_dir / filename
            with open(output_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            results.append((filename, count))

        print()

    return results


if __name__ == "__main__":
    convert_all()
