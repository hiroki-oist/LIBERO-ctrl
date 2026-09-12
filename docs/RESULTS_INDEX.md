# What is in `results/paper/`

One JSON line per rollout, at `results/paper/<run>/rollouts.jsonl`.

These records were collected on two machines and are merged here **with the second machine
taking precedence**: the first machine's working copy still held rows from before the
2026-09-08 oracle-override fix on the language and combination axes, and those rows are stale.
The paper's analysis applied the same precedence.

| run | rollouts | used for |
|---|---:|---|
| `minerva_clean` | 2,000 | tab:clean, denominator S0 (MINERVA) |
| `minerva_eval` | 8,400 | tab:axes / decomp / conj / hidden (MINERVA) |
| `minerva_rep1` | 60 | 60-rollout repeat diagnostic |
| `minerva_rep2` | 60 | 60-rollout repeat diagnostic |
| `minerva_sc_np34` | 60 | 60-rollout single-axis re-collection check |
| `minerva_singlecheck` | 60 | 60-rollout single-axis re-collection check |
| `minerva_singlecheck2` | 60 | 60-rollout single-axis re-collection check |
| `minerva_singlecheck_tk` | 60 | 60-rollout single-axis re-collection check |
| `mseed_s1000` | 10,400 | training-seed variance (MINERVA s1000) |
| `mseed_s2000` | 10,400 | training-seed variance (MINERVA s2000) |
| `mseed_s3000` | 10,400 | training-seed variance (MINERVA s3000) |
| `oft_clean` | 2,000 | tab:clean (OpenVLA-OFT) |
| `oft_eval` | 8,400 | main result (OpenVLA-OFT) |
| `oft_probe` | 1 | exploratory / diagnostic |
| `oft_probe30` | 30 | exploratory / diagnostic |
| `oft_rep1` | 60 | 60-rollout repeat diagnostic |
| `oft_singlecheck` | 60 | 60-rollout single-axis re-collection check |
| `pcv_probe` | 1 | exploratory / diagnostic |
| `pi05_axB` | 7,200 | independent re-collection, six single axes (pi0.5) |
| `pi05_clean` | 2,000 | tab:clean (pi0.5) |
| `pi05_eval` | 8,400 | main result (pi0.5) |
| `pi05_langprobe` | 12 | exploratory / diagnostic |
| `pi05_probe` | 100 | exploratory / diagnostic |
| `pi05_rep1` | 60 | 60-rollout repeat diagnostic |
| `pi05_rep2` | 60 | 60-rollout repeat diagnostic |
| `pi05_repB` | 1,200 | independent re-collection, combination (pi0.5) |
| `pi0_n1` | 30 | exploratory / diagnostic |
| `pi0_n5` | 30 | exploratory / diagnostic |
| `pi0_n50` | 30 | exploratory / diagnostic |
| `pi0_official_probe` | 48 | exploratory / diagnostic |
| `pi0_probe` | 100 | exploratory / diagnostic |
| `predvla_s10_eval` | 2,536 | exploratory / diagnostic |
| `predvla_s10_sh0_clean` | 1,000 | exploratory / diagnostic |
| `predvla_s10_sh0_eval` | 2,917 | exploratory / diagnostic |
| `predvla_s10_sh1_clean` | 1,000 | exploratory / diagnostic |
| `predvla_s10_sh1_eval` | 2,879 | exploratory / diagnostic |
| `predvla_s10_sh2_eval` | 204 | exploratory / diagnostic |
| `predvla_s13_clean` | 2,000 | tab:clean (PredVLA, seed 13) |
| `predvla_s13_eval` | 8,400 | main result (PredVLA, seed 13) |
| `predvla_s13_rep1` | 60 | 60-rollout repeat diagnostic |
| `predvla_s13_sc` | 60 | 60-rollout single-axis re-collection check |
| `predvla_s5_clean` | 2,000 | extra training seed (PredVLA s5) |
| `predvla_s5_eval` | 8,400 | extra training seed (PredVLA s5) |
| `predvla_s8_eval` | 2,594 | exploratory / diagnostic |
| `predvla_s8_sh0_clean` | 1,000 | exploratory / diagnostic |
| `predvla_s8_sh0_eval` | 1,595 | exploratory / diagnostic |
| `predvla_s8_sh1_clean` | 1,000 | exploratory / diagnostic |
| `predvla_s8_sh1_eval` | 1,481 | exploratory / diagnostic |
| `predvla_s8_sh2_eval` | 1,360 | exploratory / diagnostic |
| `predvla_s8_sh3_eval` | 1,383 | exploratory / diagnostic |
| `smolvla_axB` | 7,200 | independent re-collection, six single axes (SmolVLA) |
| `smolvla_clean` | 2,000 | tab:clean (SmolVLA) |
| `smolvla_eval` | 8,400 | main result (SmolVLA) |
| `smolvla_rep1` | 60 | 60-rollout repeat diagnostic |
| `smolvla_rep2` | 60 | 60-rollout repeat diagnostic |
| `smolvla_repB` | 1,200 | independent re-collection, combination (SmolVLA) |
| `sp50_F_paper` | 32 | exploratory / diagnostic |
| `tri_A_nstep1_te` | 15 | exploratory / diagnostic |
| `tri_B_nstep1_noflip` | 5 | exploratory / diagnostic |
| `triage_smolvla` | 5 | exploratory / diagnostic |
| `triage_smolvla_noflip` | 20 | exploratory / diagnostic |
| `univla_clean` | 2,000 | tab:clean (UniVLA) |
| `univla_eval` | 8,400 | main result (UniVLA) |
| `univla_rep1` | 60 | 60-rollout repeat diagnostic |
| `univla_sc` | 60 | 60-rollout single-axis re-collection check |
| `univla_test` | 10 | exploratory / diagnostic |
| `vlajepa_axB` | 7,200 | independent re-collection, six single axes (VLA-JEPA) |
| `vlajepa_clean` | 2,000 | tab:clean (VLA-JEPA) |
| `vlajepa_eval` | 8,400 | main result (VLA-JEPA) |
| `vlajepa_rep1` | 60 | 60-rollout repeat diagnostic |
| `vlajepa_rep2` | 60 | 60-rollout repeat diagnostic |
| `vlajepa_repB` | 1,200 | independent re-collection, combination (VLA-JEPA) |

**162,098 rollouts across 72 runs.**
