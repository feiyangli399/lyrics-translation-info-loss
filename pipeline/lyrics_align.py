#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lyrics_align.py — lyric line alignment + cleaning + length-ratio filter
=======================================================================

Protocol reference: v1.1 paper §3.2.2 (quality filtering).

Input  : raw JSONL, one song per line:
         {"song_id": "...", "style": "guofeng|pop|folk",
          "src_lang": "zh|en", "src": "<full lyrics, \\n separated>",
          "tgt": "<full lyrics>", "system": "REF", ...}
         (`system`/`seed` may be added later per translation run; this
         script only produces the ALIGNED REFERENCE pairs.)

Pipeline per song:
  1. split src/tgt into lines, strip whitespace
  2. drop noise lines (pure punctuation / emoji-only / copyright markers)
  3. positional alignment of the surviving lines
  4. length-ratio filter 0.33 <= len(src)/len(tgt) <= 3.0
     (threshold source: Wang & Wang 2024 P5/P95 ~ 0.35/2.95, protocol §3.2.2)
  5. optional seeded subsampling per (style, direction) cell

Output : aligned JSONL, one sentence pair per line:
         {"id","song_id","style","direction","system","seed","line_no",
          "src","tgt","src_lang"}

Usage:
    python lyrics_align.py --in raw/ --out data/aligned.jsonl
    python lyrics_align.py --in raw.jsonl --out data/aligned.jsonl \
        --sample-per-cell 200 --seeds 42 123 2024 6666 88888
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

# reuse syllable counters from ilv_scorer (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilv_scorer import zh_syllables, en_syllables


def line_length(text, lang, unit):
    """Length of a lyric line in the chosen unit.
    'syllable' is the protocol-consistent unit: zh ~ Han-char count,
    en ~ heuristic syllable count. 'char' = raw character count."""
    if unit == "char":
        return len(text)
    return zh_syllables(text) if lang == "zh" else en_syllables(text)

COPYRIGHT_MARKERS = [
    "严禁转载", "禁止转载", "版权所有", "all rights reserved",
    "©", "(c)", "(p)",
]

def is_noise_line(s):
    t = s.strip()
    if not t:
        return True
    # pure punctuation / symbols / emoji (no letters, no CJK, no digits)
    if not re.search(r"[a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf]", t):
        return True
    low = t.lower()
    if any(m in low for m in COPYRIGHT_MARKERS):
        return True
    return False


def clean_lines(text):
    return [ln.strip() for ln in re.split(r"[\r\n]+", text)
            if not is_noise_line(ln)]


def align_song(rec, lo=0.33, hi=3.0, unit="syllable"):
    """Positional line alignment; returns list of aligned pair dicts."""
    src_lines = clean_lines(rec["src"])
    tgt_lines = clean_lines(rec["tgt"])
    n = min(len(src_lines), len(tgt_lines))
    tgt_lang = "en" if rec["src_lang"] == "zh" else "zh"
    pairs, dropped = [], {"noise": abs(len(src_lines) - len(tgt_lines)),
                          "ratio": 0}
    for i in range(n):
        s, t = src_lines[i], tgt_lines[i]
        ls = line_length(s, rec["src_lang"], unit)
        lt = line_length(t, tgt_lang, unit)
        if ls == 0 or lt == 0:
            dropped["ratio"] += 1
            continue
        ratio = ls / lt
        if ratio < lo or ratio > hi:
            dropped["ratio"] += 1
            continue
        pairs.append({
            "id": f"{rec['song_id']}#L{i+1:03d}",
            "song_id": rec["song_id"],
            "style": rec["style"],
            "direction": f"{rec['src_lang']}2{'en' if rec['src_lang']=='zh' else 'zh'}",
            "system": rec.get("system", "REF"),
            "seed": rec.get("seed", 0),
            "line_no": i + 1,
            "src": s,
            "tgt": t,
            "src_lang": rec["src_lang"],
        })
    return pairs, dropped


def load_raw(path):
    """Accept a directory of *.jsonl or a single .jsonl file."""
    files = []
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".jsonl"):
                files.append(os.path.join(path, fn))
    else:
        files = [path]
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="raw JSONL file or directory")
    ap.add_argument("--out", required=True, help="output aligned JSONL")
    ap.add_argument("--lo", type=float, default=0.33)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--unit", choices=["syllable", "char"], default="syllable",
                    help="length unit for the ratio filter; 'syllable' is "
                         "the protocol-consistent choice (zh=Han chars, "
                         "en=heuristic syllables)")
    ap.add_argument("--sample-per-cell", type=int, default=0,
                    help="if >0, subsample each (style,direction) cell")
    ap.add_argument("--seeds", type=int, nargs="*", default=[],
                    help="write one sampled file per seed "
                         "(<out>.seed<seed>.jsonl)")
    args = ap.parse_args()

    all_pairs = []
    stats = defaultdict(lambda: {"songs": 0, "kept": 0,
                                 "noise": 0, "ratio": 0})
    n_songs = 0
    for rec in load_raw(args.inp):
        for k in ("song_id", "style", "src_lang", "src", "tgt"):
            if k not in rec:
                sys.exit(f"[align] record missing field '{k}': {rec}")
        n_songs += 1
        pairs, dropped = align_song(rec, args.lo, args.hi, args.unit)
        cell = (rec["style"], pairs[0]["direction"] if pairs else
                f"{rec['src_lang']}2?")
        stats[cell]["songs"] += 1
        stats[cell]["kept"] += len(pairs)
        stats[cell]["noise"] += dropped["noise"]
        stats[cell]["ratio"] += dropped["ratio"]
        all_pairs.extend(pairs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def dump(pairs, path):
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # full aligned set
    dump(all_pairs, args.out)
    print(f"[align] songs={n_songs} pairs={len(all_pairs)} -> {args.out}")
    for cell, st in sorted(stats.items()):
        print(f"        {cell}: songs={st['songs']} kept={st['kept']} "
              f"dropped(noise={st['noise']}, ratio={st['ratio']})")

    # seeded subsampling per cell
    if args.seeds:
        cells = defaultdict(list)
        for p in all_pairs:
            cells[(p["style"], p["direction"])].append(p)
        for seed in args.seeds:
            rng = random.Random(seed)
            out_p = []
            for cell, pairs in sorted(cells.items()):
                pool = list(pairs)
                rng.shuffle(pool)
                k = min(args.sample_per_cell, len(pool)) \
                    if args.sample_per_cell > 0 else len(pool)
                out_p.extend(pool[:k])
            base, ext = os.path.splitext(args.out)
            path = f"{base}.seed{seed}{ext}"
            dump(out_p, path)
            print(f"[align] seed={seed} sampled={len(out_p)} -> {path}")


if __name__ == "__main__":
    main()
