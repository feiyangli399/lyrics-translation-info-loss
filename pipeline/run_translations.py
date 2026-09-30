#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_translations.py — 批量翻译 6 格抽样语料（NMT=Opus-MT / LLM=DeepSeek）

用法:
  python run_translations.py --system nmt  --cells all          # CPU 批量
  python run_translations.py --system nmt  --cells guofeng_zh2en --limit 50
  python run_translations.py --system llm  --cells pop_zh2en --workers 8

输入 : data/raw/<cell>_sampled.jsonl
输出 : data/translations/<cell>__<SYSTEM>.jsonl
       {song_id, style, direction, system, src_lang, src, tgt, n_lines}
断点 : 已存在的 song_id 跳过；可随时中断重跑。

LLM 每首歌一次调用（全文翻译），temperature=0.3。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "translations")
os.makedirs(OUT, exist_ok=True)

CELLS = ["guofeng_zh2en", "guofeng_en2zh",
         "pop_zh2en", "pop_en2zh",
         "folk_zh2en", "folk_en2zh"]

# ----------------------------------------------------------------------------
# NMT backend
# ----------------------------------------------------------------------------

_nmt_cache = {}

def nmt_translate(lines, direction):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(os.cpu_count() or 8)
    # sentencepiece C++ 层不支持非 ASCII 路径 → 模型放纯英文目录
    d = "opus-mt-zh-en" if direction == "zh2en" else "opus-mt-en-zh"
    path = os.path.join("G:\\datasets\\opus-mt", d)
    if d not in _nmt_cache:
        _nmt_cache[d] = (MarianTokenizer.from_pretrained(path),
                         MarianMTModel.from_pretrained(path))
    tok, model = _nmt_cache[d]
    outs = []
    B = 32
    for i in range(0, len(lines), B):
        chunk = lines[i:i + B]
        batch = tok(chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=128)
        gen = model.generate(**batch, max_length=128, num_beams=4)
        outs.extend(tok.decode(g, skip_special_tokens=True) for g in gen)
    return outs

# ----------------------------------------------------------------------------
# LLM backend (DeepSeek, OpenAI-compatible)
# ----------------------------------------------------------------------------

LLM_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"

def _api_key():
    env = os.path.join(ROOT, ".env")
    for line in open(env, encoding="utf-8"):
        if line.startswith("LLM_API_KEY="):
            return line.strip().split("=", 1)[1]
    return os.environ.get("LLM_API_KEY", "")

PROMPT_ZH2EN = (
    "You are a professional song-lyric translator. Translate the following "
    "Chinese song lyrics into English. Rules: keep the same number of lines; "
    "one output line per input line; preserve imagery and singability where "
    "possible; output ONLY the translated lines, no commentary, no numbering."
)
PROMPT_EN2ZH = (
    "你是专业的歌词翻译。把下面的英文歌词翻译成中文。要求：行数与原文一致；"
    "每行原文对应一行译文；尽量保留意象与可唱性；只输出译文，不要注释、不要编号。"
)

def llm_translate_song(src_text, direction, key, max_retry=4):
    import urllib.request
    prompt = PROMPT_ZH2EN if direction == "zh2en" else PROMPT_EN2ZH
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": src_text},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }).encode()
    for att in range(max_retry):
        try:
            req = urllib.request.Request(
                LLM_URL, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if att == max_retry - 1:
                return None
            time.sleep(3 * (att + 1))
    return None

# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["song_id"])
            except Exception:
                pass
    return done

def run_cell(cell, system, limit, workers):
    src = os.path.join(RAW, f"{cell}_sampled.jsonl")
    dst = os.path.join(OUT, f"{cell}__{system.upper()}.jsonl")
    direction = "zh2en" if cell.endswith("zh2en") else "en2zh"
    style = cell.rsplit("_", 1)[0]
    recs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if limit:
        recs = recs[:limit]
    done = load_done(dst)
    todo = [r for r in recs if r["song_id"] not in done]
    print(f"[{cell}] {len(recs)} songs, {len(done)} done, {len(todo)} todo")

    if not todo:
        return
    fout = open(dst, "a", encoding="utf-8")

    if system == "nmt":
        for i, r in enumerate(todo):
            lines = [ln for ln in r["src"].splitlines() if ln.strip()]
            try:
                trans = nmt_translate(lines, direction)
            except Exception as e:
                print(f"  NMT FAIL {r['song_id']}: {e}")
                continue
            tgt = "\n".join(trans)
            rec = {"song_id": r["song_id"], "style": style,
                   "direction": direction, "system": "NMT",
                   "src_lang": r["src_lang"],
                   "src": r["src"], "tgt": tgt, "n_lines": len(lines)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0 or i == len(todo) - 1:
                print(f"  [{cell}] NMT {i+1}/{len(todo)}", flush=True)
    else:
        key = _api_key()
        results = {}
        def work(r):
            return r["song_id"], llm_translate_song(r["src"], direction, key)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            n = 0
            for fut in as_completed(futs):
                r = futs[fut]
                sid, tgt = fut.result()
                if tgt is None:
                    print(f"  LLM FAIL {sid}")
                    continue
                rec = {"song_id": r["song_id"], "style": style,
                       "direction": direction, "system": "LLM",
                       "src_lang": r["src_lang"],
                       "src": r["src"], "tgt": tgt,
                       "n_lines": len([l for l in r["src"].splitlines() if l.strip()])}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
                if n % 25 == 0:
                    print(f"  [{cell}] LLM {n}/{len(todo)}", flush=True)
    fout.close()
    print(f"[{cell}] finished -> {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["nmt", "llm"], required=True)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cells = CELLS if a.cells == "all" else a.cells.split(",")
    for c in cells:
        run_cell(c, a.system, a.limit, a.workers)
