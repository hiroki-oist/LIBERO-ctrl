# The seven policies

Checkpoint identifiers are the exact ones evaluated. Where a policy ships a different checkpoint
per suite, `<suite>` is one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

| Policy | Params | Checkpoint | Server | Notes |
|---|---:|---|---|---|
| MINERVA | 0.54 M | `ckpt/t05_l1_0.54M` | `minerva_server.py` | no language encoder; five axes only |
| PredVLA | 0.68 M | `PC-VLA_libero_<suite>_s13/step_30000.pt` | `pcvla_server.py` | seed 13 is the reported one; four seeds released |
| SmolVLA | 450 M | `lerobot/smolvla_libero` | `lerobot_server.py` | **`--n_action_steps 1` is required** |
| VLA-JEPA | 2.77 B | `lerobot/VLA-JEPA-LIBERO` | `lerobot_server.py` | |
| π₀.₅ | 4.14 B | `lerobot/pi05-libero` | `lerobot_server.py` | `pi05_libero_base` is **not** fine-tuned and scores 0% |
| OpenVLA-OFT | 7.54 B | `moojink/openvla-7b-oft-finetuned-<suite>` | `oft_server.py` | one checkpoint per suite |
| UniVLA | 7.54 B | `univla-libero-<suite>` | `univla_server.py` | one checkpoint per suite |

The training-seed variance section additionally uses three released MINERVA seeds,
`ckpt_seeds/abl_l1_s{1000,2000,3000}`.

## References

MINERVA — Sendai, Kohei, Tatsuya Matsushima, and Yusuke Iwasawa. "MINERVA: How Small Can a
Manipulation Policy Be and Still Solve LIBERO?" arXiv preprint arXiv:2609.03715 (2026).

PredVLA — Sawada, Hiroki, and Shunichi Kasahara. "PredVLA: A Sub-Million-Parameter
Predictive-Coding Policy for Robot Manipulation." arXiv preprint arXiv:2608.26673 (2026).

UniVLA — arXiv:2505.06111.

<!-- TODO(authors): fill in the references for SmolVLA, VLA-JEPA, pi_0.5 and OpenVLA-OFT
     before release. They are deliberately left blank rather than guessed. -->
