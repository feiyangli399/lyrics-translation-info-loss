# Information Loss in Cross-cultural Translation of Song Lyrics

Data, code, and analysis pipeline for the paper:

> Wei, S., & Li, F. (2026). Information Loss in Cross-cultural Translation of Song Lyrics: A Computational Analysis of Chinese–English Lyric Translation. *PeerJ Computer Science*.

This repository is the **canonical public release** accompanying the manuscript. It contains the full 30-song Chinese–English parallel lyric corpus, outputs of five translation systems, the analysis pipeline that computes semantic and prosodic information-loss metrics, and the result tables underlying the reported hypotheses.

## Repository layout

```
.
├── data/
│   ├── translations/              # Outputs of 5 translation systems (jsonl)
│   ├── aligned_all.jsonl          # Sentence-aligned corpus (canonical, see Releases)
│   └── aligned_input.jsonl        # Aligned corpus with model-ready fields (see Releases)
├── pipeline/
│   ├── ilv_scorer.py              # Information-loss (ℓ_sem, ℓ_rhyme) scorer
│   ├── lyrics_align.py            # Sentence alignment script
│   ├── run_translations.py        # Driver for the 5 translation systems
│   ├── make_paper_figs.py         # Generates Figures 1–3 in the manuscript
│   └── requirements.txt           # Python dependencies
├── results/
│   ├── h1_direction_tests.csv     # H1: ℓ_sem vs ℓ_rhyme asymmetry
│   ├── h23_system_tests.csv       # H2/H3: per-system metric breakdown
│   ├── h5_style_tests.csv         # H5: style-conditioned loss (folk/guofeng/pop)
│   └── ilv_across_seeds.csv       # Cross-seed robustness (5 random seeds)
├── osf_preregistration.md         # Pre-registered analysis plan
├── deviation_log.md               # Deviations from the pre-registered plan
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Large data files (GitHub Releases)

The two large aligned-corpus files (`data/aligned_all.jsonl`, ~38 MB; `data/aligned_input.jsonl`, ~10 MB) are stored as **GitHub Release assets** because GitHub's web-upload interface caps individual files at 25 MB. They are available at:

```
https://github.com/feiyangli399/lyrics-translation-info-loss/releases/tag/v1.0
```

After cloning this repository, download `aligned_all_v1.jsonl` and `aligned_input_v1.jsonl` from the v1.0 release and place them under `data/`. The pipeline scripts expect the files at exactly these paths.

A one-line download (using `gh`, the GitHub CLI) is included below under "Reproduce the analysis".

## Data description

### `data/translations/` — System outputs

Twelve files, one per (style × direction × system) cell:

```
{Style}_{Direction}__{System}.jsonl
```

where `System ∈ {LLM, NMT}`. LLM outputs come from GPT-4-class models with prompting tuned for lyrics; NMT outputs come from Helsinki-NLP Opus-MT models.

Each line contains the translated lyric aligned to the source:

```json
{
  "song_id": "folk_zh2en_001",
  "system": "LLM",
  "translation": "What a lovely jasmine blossom..."
}
```

### Source corpus attribution

The raw Chinese lyric source corpus is drawn from the open-source [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) repository (MIT-licensed). The 30-song subsample used in this study was selected via stratified sampling across three style strata (folk / guofeng / pop) and two translation directions. The full source corpus is not redistributed in this repository due to size; please cite the upstream repository for the source data.

### `data/aligned_all.jsonl` — Canonical aligned corpus

Sentence-aligned pairs ready for metric computation. Each line has:

- `song_id`, `style`, `direction`, `system`
- `src_lines`: list of source sentences
- `tgt_lines`: list of target sentences
- `csi_flags`: per-line culture-specific-item flags (used in §3.4)

## Pipeline

The five scripts in `pipeline/` reproduce every figure and table in the manuscript.

### Install

```bash
cd pipeline
pip install -r requirements.txt
```

### Reproduce the analysis

```bash
# Step 0: download the large aligned files from GitHub Release v1.0
gh release download v1.0 --repo feiyangli399/lyrics-translation-info-loss \
    --pattern "*.jsonl" --dir data/

# Step 1: align raw pairs (assumes you have the source corpus from chinese-poetry/chinese-poetry)
python lyrics_align.py --input ../data/raw --output ../data/aligned_input.jsonl

# Step 2: run the five translation systems (requires API keys for LLM, models for NMT)
python run_translations.py --input ../data/aligned_input.jsonl --output ../data/translations/

# Step 3: compute information-loss metrics
python ilv_scorer.py --input ../data/aligned_all.jsonl --output ../results/

# Step 4: generate manuscript figures
python make_paper_figs.py --input ../results/ --output figures/
```

Expected runtime: ≈20 minutes on a single CPU.

## Results

All hypothesis-test CSVs in `results/` are the canonical outputs that feed into Tables 2–5 of the manuscript. Column definitions:

- `h1_direction_tests.csv` — columns: `metric, direction, mean, sd, n, p, cohens_d`
- `h23_system_tests.csv` — columns: `system, metric, mean, sd, n, p_vs_NMT`
- `h5_style_tests.csv` — columns: `style, metric, mean, sd, n`
- `ilv_across_seeds.csv` — columns: `seed, metric, mean, sd`

## License

Released under the MIT License (see `LICENSE`). The corpus is released for non-commercial research and educational use; commercial redistribution requires written permission from the authors.

## Citation

If you use this repository, please cite:

```
Wei, S., & Li, F. (2026). Information Loss in Cross-cultural Translation of Song Lyrics:
A Computational Analysis of Chinese–English Lyric Translation. PeerJ Computer Science.
```

## Contact

- Wei Shan (corresponding for data) — fadolshui@163.com
- Li Feiyang (corresponding for paper) — feiyangli399@gmail.com
