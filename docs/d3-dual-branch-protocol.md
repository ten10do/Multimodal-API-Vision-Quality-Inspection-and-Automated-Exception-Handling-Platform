# D3 Dual-Branch Candidate Protocol

Candidate: `steel-patchcore-d3-candidate@1.3.0-candidate.1`

Status: `CANDIDATE`; production promotion is prohibited.

## Image branch

The image branch is the unchanged `steel-patchcore-d3-candidate@1.2.0-candidate.1` D3-ZCA path: DINOv2-B/14 final patch tokens at 252px, frozen train-normal ZCA, per-patch L2 normalization, cosine 1-NN against the frozen 50,000-row D3 bank, and A0 global maximum over all seven tiles. The threshold is exactly `0.8471092581748962`.

The dual integration calls the existing D3 predictor as a sealed sub-branch. Localization output is computed only after the image score and label inputs have been captured. Localization never supplies an image score and never changes the threshold.

## Localization branch

R-L3 uses the same frozen DINOv2-B/14 weights with two independent dense feature streams:

- R-L1: normalized patch tokens after transformer block index 7, input 252, 18x18 grid, cosine 1-NN against the frozen R-L1 bank.
- R-L2: final normalized patch tokens, input 448, 32x32 grid, cosine 1-NN against the frozen R-L2 bank.

Each raw distance grid is bilinearly resized to 256x256. The two maps are fused per tile with an equal arithmetic mean, then the seven tile maps are stitched to 256x1600 using mean overlap. Per-image min-max normalization is presentation-only and occurs after the raw localization map is complete.

## Acceptance and isolation

The sealed evaluation gate is image AUROC `0.8179071714278028`, pixel AUROC `0.9241393857425543`, and AUPRO `0.7993981069909584`. Candidate loading verifies every referenced hash before and after array loading and fails closed on missing, mismatched, malformed, or mutated artifacts.

The R-L1/R-L2 bank files remain ignored runtime artifacts and are referenced by hash; they are not committed. No weights, datasets, bank contents, whitening artifacts, thresholds, production configuration, backbone parameters, or fine-tuned parameters are changed by this integration.
