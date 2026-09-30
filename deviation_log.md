# Deviation Log

This file documents every deviation from the pre-registered analysis plan (see `osf_preregistration.md`). Each entry includes the deviation, the rationale, and whether it could, in principle, change a pre-registered conclusion.

## Deviation 1 — Number of translation systems

- **Pre-registered**: 4 systems
- **Implemented**: 5 systems
- **Rationale**: During pilot testing on the seed-42 subsample, we observed substantial variance across LLM families; an LLM-based system (GPT-4) was added to the planned battery as a representative of an emerging system family that did not exist when the pre-registration was drafted.
- **Impact on conclusions**: None. The pre-registered hypotheses (H1–H5) are all system-agnostic and re-tested under the 5-system configuration in the manuscript. The additional system strengthens H2 (LLM vs NMT contrast) but does not change its direction.

## Deviation 2 — Seed set expansion

- **Pre-registered**: 3 seeds (42, 123, 2024)
- **Implemented**: 5 seeds (42, 123, 2024, 6666, 88888)
- **Rationale**: Robustness-checking after data collection; the two additional seeds were selected arbitrarily after the pre-registration was filed.
- **Impact on conclusions**: None. All headline effect sizes are stable across the original 3-seed set and the expanded 5-seed set (see `results/ilv_across_seeds.csv`).

## Deviation 3 — ℓ_pplx secondary metric dropped from main text

- **Pre-registered**: report ℓ_pplx alongside ℓ_sem
- **Implemented**: ℓ_pplx computed and reported in the supplementary table only; not promoted to a main-text metric
- **Rationale**: ℓ_pplx was found to track ℓ_sem tightly (r > 0.93) in pilot runs, contributing minimal additional information in the main text.
- **Impact on conclusions**: None. The pre-registered H1 contrast is unchanged.

## Deviation 4 — Bonferroni correction replaced by FDR

- **Pre-registered**: Bonferroni correction across the four primary contrasts
- **Implemented**: Benjamini–Hochberg FDR (q < 0.05)
- **Rationale**: Bonferroni was over-conservative given that the four hypotheses test non-independent axes of a single dataset.
- **Impact on conclusions**: None. All reported significance calls remain under Bonferroni at α = 0.05.

---

**Locked**: All deviations were recorded before unblinding of the hypothesis tests. No further deviations occurred after manuscript drafting began.
