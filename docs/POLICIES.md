# The seven policies

Listed from the most widely known to the least. Checkpoint identifiers are the exact ones
evaluated. Where a policy ships a different checkpoint per suite, `<suite>` is one of
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

| Policy | Params | Checkpoint | Server | Notes |
|---|---:|---|---|---|
| π₀.₅ | 4.14 B | `lerobot/pi05-libero` | `lerobot_server.py` | `pi05_libero_base` is **not** fine-tuned and scores 0% |
| OpenVLA-OFT | 7.54 B | `moojink/openvla-7b-oft-finetuned-<suite>` | `oft_server.py` | one checkpoint per suite |
| UniVLA | 7.54 B | `univla-libero-<suite>` | `univla_server.py` | one checkpoint per suite; autoregressive latent actions rather than flow matching |
| SmolVLA | 450 M | `lerobot/smolvla_libero` | `lerobot_server.py` | **`--n_action_steps 1` is required** |
| VLA-JEPA | 2.77 B | `lerobot/VLA-JEPA-LIBERO` | `lerobot_server.py` | |
| PredVLA | 0.68 M | `PC-VLA_libero_<suite>_s13/step_30000.pt` | `pcvla_server.py` | seed 13 is the reported one; four seeds released |
| MINERVA | 0.54 M | `ckpt/t05_l1_0.54M` | `minerva_server.py` | no language encoder; five axes only |

The training-seed variance section additionally uses three released MINERVA seeds,
`ckpt_seeds/abl_l1_s{1000,2000,3000}`.

## References

**π₀.₅** — Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia,
Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker,
Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke,
Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch,
Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz,
James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky.
"π₀.₅: a Vision-Language-Action Model with Open-World Generalization."
arXiv preprint arXiv:2504.16054 (2025). https://arxiv.org/abs/2504.16054

**OpenVLA-OFT** — Kim, Moo Jin, Chelsea Finn, and Percy Liang. "Fine-Tuning Vision-Language-Action
Models: Optimizing Speed and Success." Robotics: Science and Systems (RSS), 2025.
arXiv:2502.19645. https://arxiv.org/abs/2502.19645

**UniVLA** — Bu, Qingwen, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao,
Ping Luo, and Hongyang Li. "UniVLA: Learning to Act Anywhere with Task-centric Latent Actions."
Robotics: Science and Systems (RSS), 2025. arXiv:2505.06111. https://arxiv.org/abs/2505.06111

**SmolVLA** — Shukor, Mustafa, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans,
Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti,
Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. "SmolVLA: A Vision-Language-Action
Model for Affordable and Efficient Robotics." arXiv preprint arXiv:2506.01844 (2025).
https://arxiv.org/abs/2506.01844

**VLA-JEPA** — Sun, Jingwen, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu,
Guangzhong Sun, Xin Jin, and Zhibo Chen. "VLA-JEPA: Enhancing Vision-Language-Action Model with
Latent World Model." arXiv preprint arXiv:2602.10098 (2026). https://arxiv.org/abs/2602.10098

**PredVLA** — Sawada, Hiroki, and Shunichi Kasahara. "PredVLA: A Sub-Million-Parameter
Predictive-Coding Policy for Robot Manipulation." arXiv preprint arXiv:2608.26673 (2026).
https://arxiv.org/abs/2608.26673

**MINERVA** — Sendai, Kohei, Tatsuya Matsushima, and Yusuke Iwasawa. "MINERVA: How Small Can a
Manipulation Policy Be and Still Solve LIBERO?" arXiv preprint arXiv:2609.03715 (2026).
https://arxiv.org/abs/2609.03715
