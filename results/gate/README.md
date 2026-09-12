# Reproduction-gate records

The 500 nominal LIBERO-Spatial rollouts used for the per-rollout reproduction check reported in
the README:

| directory | policy | GPU relative to the archive |
|---|---|---|
| `oft_clean/` | OpenVLA-OFT | the same one |
| `uv_clean/`  | UniVLA      | a different one |

Compare against `results/paper/{oft,univla}_clean/rollouts.jsonl`, restricted to
`suite == "libero_spatial"`, matching on `rollout_id`.
