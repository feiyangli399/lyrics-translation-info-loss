#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ilv_scorer.py — Information-Loss Vector (ILV) scorer
====================================================

Implements the 5-dim ILV defined in the pre-registered protocol (v1.1, §3.3):

    ILV = (l_sem, l_csi, l_rhyme, l_meter, l_emo)   each in [0, 1]

    l_sem   = 1 - BERTScore_F1(src, tgt)
    l_csi   = piecewise: 0 if CSIs_src==0; 1 - CSIs_tgt/CSIs_src if
              0 <= tgt <= src (tgt==0 -> total loss = 1.0);
              0 if tgt > src (translator introduced new CSIs)
    l_rhyme = 1 - min(rhyme_density_tgt / rhyme_density_src, 1)  (per song group)
    l_meter = 1 - 1/(1 + |ds| / s_bar),  ds = syl_src - syl_tgt
    l_emo   = 1 - max(cos(emo_vec_src, emo_vec_tgt), 0)

Input  : aligned JSONL produced by lyrics_align.py
         fields: id, song_id, src, tgt, system(HUM|NMT|LLM),
                 direction(zh2en|en2zh), style(guofeng|pop|folk), seed, line_no
Output : - ilv_per_pair.csv     one row per sentence pair (raw scores)
         - ilv_per_cell.csv     mean +/- SD per (system,direction,style,seed)
         - ilv_summary.csv      mean +/- SD across seeds per cell
         - ilv_summary_ci.csv   bootstrap 95% CI for each cell mean

Usage:
    python ilv_scorer.py --in data/aligned.jsonl --out results/ \
        --csi csi_dict/csi_dict.csv --sem-backend bertscore \
        --boot 10000 --boot-seed 42

Backends (each degrades gracefully, with warnings):
    l_sem   : bertscore (multilingual) > sentence-transformers > None
    l_meter : zh -> pypinyin finals count; en -> pronouncing > heuristic
    l_rhyme : zh -> pypinyin finals; en -> pronouncing > suffix heuristic
    l_emo   : built-in bilingual Plutchik-8 lexicon (REPLACE with NRC EmoLex
              + translated zh lexicon before submission)
    l_csi   : CSV dictionary, greedy longest-match (zh) / regex word-match (en)

NOTE: This file intentionally contains NO mock data. Undefined scores are
written as empty cells (NaN) and counted in the report -- the protocol's
joint ILV uses the per-dimension mean over defined pairs only.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

ILV_DIMS = ["l_sem", "l_csi", "l_rhyme", "l_meter", "l_emo"]

# ----------------------------------------------------------------------------
# Language helpers
# ----------------------------------------------------------------------------

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
WORD_RE_EN = re.compile(r"[a-z']+")

def detect_lang(text):
    """Return 'zh' if text contains CJK chars, else 'en'."""
    return "zh" if CJK_RE.search(text) else "en"

def tokenize_lines(text):
    """Split lyrics text into non-empty lines."""
    return [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]

# ----------------------------------------------------------------------------
# Syllable / rhyme backends
# ----------------------------------------------------------------------------

_PYPINYIN = None
_PRONOUNCING = None

def _load_pypinyin():
    global _PYPINYIN
    if _PYPINYIN is None:
        try:
            from pypinyin import lazy_pinyin, Style
            _PYPINYIN = (lazy_pinyin, Style)
        except ImportError:
            _PYPINYIN = False
    return _PYPINYIN

def _load_pronouncing():
    global _PRONOUNCING
    if _PRONOUNCING is None:
        try:
            import pronouncing
            _PRONOUNCING = pronouncing
        except ImportError:
            _PRONOUNCING = False
    return _PRONOUNCING


def zh_syllables(text):
    """Mandarin syllable count ~= number of Han characters."""
    return len(CJK_RE.findall(text))


def zh_finals(text):
    """Final (yunmu) of each Han char; fallback: the char itself."""
    pp = _load_pypinyin()
    chars = CJK_RE.findall(text)
    if pp:
        lazy_pinyin, Style = pp
        return lazy_pinyin("".join(chars), style=Style.FINALS, strict=False)
    return chars


EN_VOWELS = "aeiouy"
def en_syllables_word(w):
    """Heuristic English syllable count for one word."""
    w = w.lower().strip("'")
    if not w:
        return 0
    groups, prev_v = 0, False
    for ch in w:
        v = ch in EN_VOWELS
        if v and not prev_v:
            groups += 1
        prev_v = v
    if w.endswith("e") and groups > 1 and not w.endswith(("le", "ee", "ye")):
        groups -= 1
    return max(groups, 1)


def en_syllables(text):
    return sum(en_syllables_word(w) for w in WORD_RE_EN.findall(text.lower()))


def en_rhyme_key(word):
    """Rhyme key of an English word: CMU dict phones from last stressed vowel;
    fallback to normalized ending."""
    pr = _load_pronouncing()
    w = word.lower().strip("'")
    if not w:
        return ""
    if pr:
        phones = pr.phones_for_word(w)
        if phones:
            ph = phones[0].split()
            for i in range(len(ph) - 1, -1, -1):
                if ph[i][-1] in "012":          # stressed vowel
                    return " ".join(ph[i:])
    # heuristic fallback: last vowel group + trailing consonants
    m = re.findall(r"([aeiouy]+[^aeiouy]*)$", w)
    return m[-1] if m else w[-3:]


def rhyme_density(lines, lang):
    """Share of line-ending tokens whose rhyme key appears >=2 times
    among the line endings of this lyric."""
    keys = []
    for ln in lines:
        toks = CJK_RE.findall(ln) if lang == "zh" else WORD_RE_EN.findall(ln.lower())
        if not toks:
            continue
        if lang == "zh":
            f = zh_finals(toks[-1])
            keys.append(f[0] if f else toks[-1])
        else:
            keys.append(en_rhyme_key(toks[-1]))
    if len(keys) < 2:
        return 0.0
    freq = defaultdict(int)
    for k in keys:
        if k:
            freq[k] += 1
    rhymed = sum(1 for k in keys if k and freq[k] >= 2)
    return rhymed / len(keys)


def syllable_count(text, lang):
    return zh_syllables(text) if lang == "zh" else en_syllables(text)

# ----------------------------------------------------------------------------
# CSI dictionary
# ----------------------------------------------------------------------------

def load_csi_dict(path):
    """CSV columns: token,type,lang,gloss_en,note  (lang: zh|en)."""
    zh, en = [], []
    if not path or not os.path.exists(path):
        warnings.warn(f"[ilv] CSI dict not found: {path} -> l_csi will be 0")
        return zh, en
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tok = (row.get("token") or "").strip()
            if not tok:
                continue
            (zh if row.get("lang") == "zh" else en).append(tok)
    # longest-first for greedy zh match
    zh.sort(key=len, reverse=True)
    return zh, en


def count_csi(text, lang, zh_tokens, en_tokens):
    """Count CSI occurrences (occurrences, not types)."""
    if lang == "zh":
        n, i, s = 0, 0, text
        while i < len(s):
            for tok in zh_tokens:
                if s.startswith(tok, i):
                    n += 1
                    i += len(tok)
                    break
            else:
                i += 1
        return n
    hits = 0
    for tok in en_tokens:
        hits += len(re.findall(r"(?<![a-z'])" + re.escape(tok.lower()) +
                               r"(?![a-z'])", text.lower()))
    return hits

# ----------------------------------------------------------------------------
# Emotion backend (Plutchik-8 lexicon, bilingual seed -- REPLACE for submission)
# ----------------------------------------------------------------------------

EMO_DIMS = ["joy", "trust", "fear", "surprise", "sadness", "disgust",
            "anger", "anticipation"]

EMO_LEX = {
    "zh": {
        "joy": "喜悦 快乐 开心 欢笑 幸福 甜 温柔 阳光 灿烂 美好 爱 喜欢 "
               "高兴 愉快 欢喜 甜蜜 温暖 微笑 笑容 欢乐 陶醉 浪漫 吻 拥抱 "
               "心跳 心动 宝贝 亲爱 可爱 美丽 漂亮 迷人 惊喜 晴朗 彩虹 花 "
               "玫瑰 星星 月亮 月光 春风 夏天 节日 庆祝 跳舞 歌唱 笑声",
        "trust": "相信 信任 陪伴 永远 承诺 守护 真诚 约定 誓言 忠诚 "
                 "依靠 依赖 坚定 坚持 牵手 一起 同行 相随 不弃 不离 "
                 "永远在一起 一生 一辈子 直到永远",
        "fear": "害怕 恐惧 黑暗 颤抖 惊恐 孤单 孤独 寂寞 恐慌 不安 "
                "寒冷 冰冷 阴影 魔鬼 深渊 坠落 迷失 茫然 无助 危险",
        "surprise": "惊讶 忽然 突然 奇迹 意外 惊喜 没想到 竟 竟然 原来 "
                    "一瞬间 刹那 转身 回眸",
        "sadness": "悲伤 伤心 泪 哭泣 难过 失落 离别 心碎 遗憾 怀念 思念 "
                   "眼泪 流泪 哭泣 伤痛 痛苦 忧伤 哀愁 愁 离别 分开 再见 "
                   "告别 离去 走远 背影 回忆 往事 从前 曾经 想念 挂念 孤单 "
                   "寂寞 空虚 憔悴 疲惫 无奈 叹息 秋风 落叶 黄昏 残阳",
        "disgust": "厌恶 恶心 厌倦 虚伪 谎言 欺骗 背叛 敷衍 冷漠 绝情 肮脏",
        "anger": "愤怒 生气 恨 燃烧 怒吼 咆哮 疯狂 爆发 报复 怒火 讨厌 恨你 "
                 "滚 闭嘴 放开 挣脱 反抗",
        "anticipation": "期待 等待 希望 向往 明天 未来 梦想 追寻 盼望 渴望 "
                        "期盼 憧憬 远方 出发 前进 奔跑 飞翔 自由 黎明 曙光 "
                        "日出 新的一天 旅程 路 远方",
    },
    "en": {
        "joy": "happy joy love smile sunshine sweet beautiful glad delight "
               "wonderful amazing perfect lovely darling honey baby kiss "
               "dance dancing sing singing laugh laughter celebrate party "
               "fun alive free freedom heart paradise heaven dream come true "
               "gold golden shine shining bright light star stars moon "
               "summer spring flower rose beauty pretty gorgeous",
        "trust": "trust believe forever promise together faith loyal "
                 "always never leave stay holding hands side by side "
                 "true truly honest devotion devoted swear vow "
                 "forever yours mine ours eternal eternity",
        "fear": "fear afraid dark darkness tremble scared lonely alone "
                "nightmare cold frozen shadow shadows devil demon hell "
                "danger dangerous lost falling fall drowning scream "
                "terror horror shiver",
        "surprise": "surprise surprised sudden suddenly miracle wonder "
                    "unexpected shock wow unbelievable magic magical "
                    "moment instantly",
        "sadness": "sad cry crying tears sorrow miss missing goodbye "
                   "heartbroken broken lonely loneliness alone empty "
                   "pain painful hurt hurting ache sorrowful grief mourn "
                   "sadness blue blues rain raining tears fall gone left "
                   "leaving apart away distant memories yesterday used to "
                   "regret sorry goodbye farewell",
        "disgust": "hate disgust sick tired fake liar lies lying cheat "
                   "cheating betrayed betrayal dirty filthy disgusting",
        "anger": "angry anger rage burn burning fight fighting scream "
                 "screaming mad furious revenge hate hell damn fire "
                 "explode explode rage fury",
        "anticipation": "wait waiting hope hoping dream dreaming dreams "
                        "tomorrow future wish wishing longing yearning "
                        "someday soon coming go going run running fly "
                        "flying wings journey road way forward ahead "
                        "rise rising dawn sunrise morning new day",
    },
}

def emo_vector(text, lang):
    words = EMO_LEX.get(lang, {})
    vec = [0.0] * len(EMO_DIMS)
    if lang == "zh":
        for d, toks in words.items():
            vec[EMO_DIMS.index(d)] = sum(text.count(t) for t in toks.split())
    else:
        low = text.lower()
        for d, toks in words.items():
            vec[EMO_DIMS.index(d)] = sum(
                len(re.findall(r"(?<![a-z'])" + t + r"(?![a-z'])", low))
                for t in toks.split())
    return vec


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)

# ----------------------------------------------------------------------------
# Semantic backend
# ----------------------------------------------------------------------------

class SemScorer:
    """l_sem = 1 - similarity(src, tgt); similarity in [0,1]."""

    def __init__(self, backend="bertscore", device=None):
        self.backend = None
        self._fn = None
        if backend in ("bertscore", "auto"):
            try:
                from bert_score import score as bscore
                def fn(pairs):
                    _, _, f = bscore([s for s, _ in pairs],
                                     [t for _, t in pairs],
                                     lang="zh",   # multilingual model
                                     model_type="bert-base-multilingual-cased",
                                     verbose=False, device=device)
                    return [float(x) for x in f]
                self._fn, self.backend = fn, "bertscore"
            except ImportError:
                warnings.warn("[ilv] bert_score not installed")
        if self._fn is None and backend in ("st", "auto", "bertscore"):
            try:
                from sentence_transformers import SentenceTransformer, util
                model = SentenceTransformer(
                    "sentence-transformers/LaBSE", device=device)
                def fn(pairs):
                    e1 = model.encode([s for s, _ in pairs],
                                      convert_to_tensor=True)
                    e2 = model.encode([t for _, t in pairs],
                                      convert_to_tensor=True)
                    return [float(util.cos_sim(a, b)) for a, b in zip(e1, e2)]
                self._fn, self.backend = fn, "LaBSE"
            except ImportError:
                warnings.warn("[ilv] sentence_transformers not installed")
        if self._fn is None:
            warnings.warn("[ilv] NO semantic backend -> l_sem = NaN")

    def score(self, pairs):
        if self._fn is None:
            return [None] * len(pairs)
        out = self._fn(pairs)
        return [max(0.0, min(1.0, s)) if s is not None else None for s in out]

# ----------------------------------------------------------------------------
# ILV core
# ----------------------------------------------------------------------------

def l_meter(src, tgt, lsrc, ltgt):
    s, t = syllable_count(src, lsrc), syllable_count(tgt, ltgt)
    bar = (s + t) / 2.0
    if bar == 0:
        return None
    return 1.0 - 1.0 / (1.0 + abs(s - t) / bar)


def l_csi(src, tgt, lsrc, ltgt, zh_toks, en_toks):
    cs = count_csi(src, lsrc, zh_toks, en_toks)
    ct = count_csi(tgt, ltgt, zh_toks, en_toks)
    if cs == 0:
        val = 0.0
    elif ct <= cs:              # includes ct == 0: ALL CSIs lost -> loss 1.0
        val = 1.0 - ct / cs     # NOTE: paper formula says "0 < tgt <= src"
                                # which leaves ct==0 undefined; fixed to
                                # "0 <= tgt <= src" -- flag for v1.2 revision
    else:                       # ct > cs: new CSI introduced
        val = 0.0
    return val, cs, ct


def l_emo(src, tgt, lsrc, ltgt):
    c = cosine(emo_vector(src, lsrc), emo_vector(tgt, ltgt))
    if c is None:
        return None
    return 1.0 - max(c, 0.0)

# ----------------------------------------------------------------------------
# Aggregation + bootstrap
# ----------------------------------------------------------------------------

def mean_sd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def boot_ci(vals, n_boot=10000, seed=42):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    n = len(vals)
    if HAS_NUMPY:
        arr = np.array(vals)
        means = np.array([arr[[rng.randrange(n) for _ in range(n)]].mean()
                          for _ in range(n_boot)])
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ILV scorer v1.1 protocol")
    ap.add_argument("--in", dest="inp", required=True, help="aligned JSONL")
    ap.add_argument("--out", dest="out", required=True, help="output dir")
    ap.add_argument("--csi", default="csi_dict/csi_dict.csv")
    ap.add_argument("--sem-backend", default="auto",
                    choices=["auto", "bertscore", "st", "none"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zh_toks, en_toks = load_csi_dict(args.csi)
    sem = SemScorer(args.sem_backend, args.device)

    # ---- load
    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        sys.exit("[ilv] empty input")
    print(f"[ilv] loaded {len(rows)} pairs; sem backend = {sem.backend}")

    # ---- rhyme density per song group (protocol: line-level ratio per lyric)
    groups = defaultdict(lambda: {"src": [], "tgt": [], "lsrc": None, "ltgt": None})
    for r in rows:
        lsrc = r.get("src_lang") or ("zh" if r["direction"].startswith("zh") else "en")
        ltgt = "en" if lsrc == "zh" else "zh"
        r["_lsrc"], r["_ltgt"] = lsrc, ltgt
        g = groups[(r.get("song_id", r["id"]), r["system"], r["seed"])]
        g["src"].append(r["src"]); g["tgt"].append(r["tgt"])
        g["lsrc"], g["ltgt"] = lsrc, ltgt
    dens = {}
    for k, g in groups.items():
        rs = rhyme_density(g["src"], g["lsrc"])
        rt = rhyme_density(g["tgt"], g["ltgt"])
        dens[k] = 0.0 if rs == 0 else 1.0 - min(rt / rs, 1.0)

    # ---- semantic (batch)
    sims = sem.score([(r["src"], r["tgt"]) for r in rows])

    # ---- per-pair scores
    for r, sim in zip(rows, sims):
        r["l_sem"] = None if sim is None else 1.0 - sim
        v, cs, ct = l_csi(r["src"], r["tgt"], r["_lsrc"], r["_ltgt"],
                          zh_toks, en_toks)
        r["l_csi"], r["csi_src"], r["csi_tgt"] = v, cs, ct
        r["l_rhyme"] = dens[(r.get("song_id", r["id"]), r["system"], r["seed"])]
        r["l_meter"] = l_meter(r["src"], r["tgt"], r["_lsrc"], r["_ltgt"])
        r["l_emo"] = l_emo(r["src"], r["tgt"], r["_lsrc"], r["_ltgt"])

    # ---- write per-pair
    cols = ["id", "song_id", "system", "direction", "style", "seed", "line_no",
            "csi_src", "csi_tgt"] + ILV_DIMS
    with open(os.path.join(args.out, "ilv_per_pair.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- per-cell + summary
    cells = defaultdict(list)
    for r in rows:
        cells[(r["system"], r["direction"], r["style"], r["seed"])].append(r)
    cell_rows, sum_rows, ci_rows = [], [], []
    sup = defaultdict(list)                      # (sys,dir,style) -> cell means
    for (sysm, dire, sty, seed), rs in sorted(cells.items()):
        rec = {"system": sysm, "direction": dire, "style": sty, "seed": seed}
        for d in ILV_DIMS:
            m, sd, n = mean_sd([r[d] for r in rs])
            rec[f"{d}_mean"], rec[f"{d}_sd"], rec[f"{d}_n"] = m, sd, n
            if m is not None:
                sup[(sysm, dire, sty)].append((d, m))
        cell_rows.append(rec)
    fieldnames = ["system", "direction", "style", "seed"] + \
        [f"{d}_{s}" for d in ILV_DIMS for s in ("mean", "sd", "n")]
    with open(os.path.join(args.out, "ilv_per_cell.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(cell_rows)

    for (sysm, dire, sty), items in sorted(sup.items()):
        bydim = defaultdict(list)
        for d, m in items:
            bydim[d].append(m)
        n_seeds = sum(1 for (s, dr, st, _sd) in cells
                      if (s, dr, st) == (sysm, dire, sty))
        srec = {"system": sysm, "direction": dire, "style": sty,
                "n_seeds": n_seeds}
        for d in ILV_DIMS:
            m, sd, n = mean_sd(bydim[d])
            srec[f"{d}_mean"], srec[f"{d}_sd"] = m, sd
            lo, hi = boot_ci(bydim[d], args.boot, args.boot_seed)
            ci_rows.append({"system": sysm, "direction": dire, "style": sty,
                            "dim": d, "mean": m, "ci_lo": lo, "ci_hi": hi})
        sum_rows.append(srec)
    fn2 = ["system", "direction", "style", "n_seeds"] + \
          [f"{d}_{s}" for d in ILV_DIMS for s in ("mean", "sd")]
    with open(os.path.join(args.out, "ilv_summary.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn2)
        w.writeheader(); w.writerows(sum_rows)
    with open(os.path.join(args.out, "ilv_summary_ci.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "direction", "style",
                                          "dim", "mean", "ci_lo", "ci_hi"])
        w.writeheader(); w.writerows(ci_rows)

    print(f"[ilv] wrote 4 CSV files to {args.out}")
    missing = {d: sum(1 for r in rows if r[d] is None) for d in ILV_DIMS}
    print(f"[ilv] undefined scores per dim: {missing}")


if __name__ == "__main__":
    main()
