# Pre-registered Analysis Plan

This document mirrors the pre-registration filed on the Open Science Framework (OSF) for the study "Information Loss in Cross-cultural Translation of Song Lyrics". The pre-registration was locked before data collection began.

## Hypotheses

### H1 (Direction asymmetry)
The asymmetry in information loss between Chinese→English and English→Chinese translation directions is dominated by the prosodic axis (ℓ_rhyme), not the semantic axis (ℓ_sem).

### H2 (System-level)
LLM-based translation systems outperform neural MT on ℓ_sem but underperform on ℓ_rhyme.

### H3 (CSI dominance)
Culture-specific items (CSIs) account for the majority of referential-semantic divergence, especially in en→zh direction.

### H4 (Cell-level boundary)
For cells with high ℓ_rhyme, ℓ_sem is near-constant; conversely, low-ℓ_rhyme cells show full ℓ_sem range. (i.e., the two losses are decoupled at the cell level.)

### H5 (Style)
Translation loss is moderated by source style: folk < guofeng < pop on ℓ_rhyme.

## Pre-registered analysis steps

1. Sample 30 Chinese–English song pairs stratified by style (folk/guofeng/pop) and direction (zh2en/en2zh).
2. Run five translation systems on the aligned pairs.
3. Compute ℓ_sem as 1 − cosine(SBERT(src), SBERT(tgt)).
4. Compute ℓ_rhyme as a weighted composite of: syllable-count deviation, rhyme-scheme preservation rate, and tonal-pattern correlation.
5. Compute ℓ_pplx (length-normalised perplexity shift) as a secondary semantic metric.
6. Report per-hypothesis test statistics with Bonferroni correction across the four primary contrasts.

## Pre-registered exclusion rules

- Pairs shorter than 4 sentences are excluded from ℓ_rhyme computation (insufficient material for rhyme analysis).
- CSI annotation: only items in the predefined CSI dictionary (96 entries, see `pipeline/csi_dict/`).
- Failed translations (system returns empty string or non-target-language output) are excluded pairwise.

## Random seeds

All stochastic analyses pre-registered with five seeds: 42, 123, 2024, 6666, 88888.

## Deviations

The complete deviation log is in `deviation_log.md`. The only material deviation from the pre-registration is the addition of a fifth translation system (originally four were planned; LLM-based GPT-4 was added during pilot testing as a representative of an emerging system family, without changing any of the four pre-registered hypotheses).
