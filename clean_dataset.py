"""
clean_dataset.py — Sanitise datasetscript/ for Kaggle upload.

Fixes:
  1. Image file names:  "name (N).ext" -> "name_N.ext"  (spaces + parens removed)
  2. Extension casing:  ".PNG" -> ".png"
  3. Class folder names: Unicode diacritics -> ASCII equivalents

Run once from the project root:
    python3 clean_dataset.py
"""

import re
import unicodedata
from pathlib import Path

DATASET_ROOT = Path(__file__).parent / "datasetscript"

# Unicode class folder names -> ASCII equivalents.
# Keys must be exact current folder names; values must not collide with
# other existing class names in the same script folder.
DIR_RENAMES = {
    # Kawi & Pallawa (Sanskrit diacritics)
    "ḍa":  "dda",    # retroflex d
    "ḍha": "ddha",   # retroflex dh
    "ṅa":  "nga",    # velar ng (safe: no 'nga' in Kawi/Pallawa)
    "ña":  "nya",    # palatal ny (safe: no 'nya' in Kawi/Pallawa)
    "ṇa":  "nna",    # retroflex n
    "ṣa":  "ssa",    # retroflex s
    "śa":  "sha",    # palatal s
    "ṭa":  "tta",    # retroflex t
    "ṭha": "ttha",   # retroflex th
    # Sunda
    "é":   "etaling",  # e-taling (closed e)
}

# Used to clean Unicode chars that may appear inside file stems too (e.g. Sunda)
CHAR_SUBS = {
    "ḍ": "dd",
    "ṭ": "tt",
    "ṣ": "ss",
    "ś": "sh",
    "ṅ": "ng",
    "ṇ": "nn",
    "ñ": "ny",
    "é": "etaling",
}

_BRACKET_PATTERN = re.compile(r"^(.*?) \((\d+)\)(\.[a-zA-Z]+)$")


def clean_stem(name: str) -> str:
    name = unicodedata.normalize("NFC", name)
    for char, repl in CHAR_SUBS.items():
        name = name.replace(char, repl)
    return name


def rename_files(root: Path) -> int:
    count = 0
    # Collect first to avoid modifying the iterator mid-loop
    files = [p for p in root.rglob("*") if p.is_file()]
    for p in files:
        new_name = p.name
        m = _BRACKET_PATTERN.match(p.name)
        if m:
            stem, num, ext = m.group(1), m.group(2), m.group(3).lower()
            stem_clean = clean_stem(stem)
            new_name = f"{stem_clean}_{num}{ext}"
        elif not p.stem.isascii() or p.suffix != p.suffix.lower():
            new_name = clean_stem(p.stem) + p.suffix.lower()

        if new_name != p.name:
            target = p.parent / new_name
            if target.exists():
                print(f"  SKIP (conflict): {p.name} -> {new_name}")
                continue
            p.rename(target)
            count += 1
    return count


def rename_dirs(root: Path) -> int:
    count = 0
    for script_dir in sorted(root.iterdir()):
        if not script_dir.is_dir():
            continue
        for cls_dir in sorted(script_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            new_name = DIR_RENAMES.get(unicodedata.normalize("NFC", cls_dir.name))
            if new_name is None:
                continue
            target = cls_dir.parent / new_name
            if target.exists():
                print(f"  SKIP (conflict): {cls_dir} -> {target} already exists")
                continue
            cls_dir.rename(target)
            print(f"  DIR  '{cls_dir.name}'  ->  '{new_name}'  [{script_dir.name}]")
            count += 1
    return count


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    print(f"Dataset root : {DATASET_ROOT.resolve()}")
    print()

    print("Step 1 — renaming image files ...")
    n_files = rename_files(DATASET_ROOT)
    print(f"  {n_files} files renamed.")
    print()

    print("Step 2 — renaming class directories ...")
    n_dirs = rename_dirs(DATASET_ROOT)
    print(f"  {n_dirs} directories renamed.")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
