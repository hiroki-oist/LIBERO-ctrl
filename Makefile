# LIBERO-CTRL
# ★PY は LIBERO が動く python を指す。既定はこのリポジトリを開発した環境。
PY ?= python

.PHONY: help smoke paper verify clean-out

help:
	@echo "make smoke   — 3 つの導入レベルが動くことを確認（rollout を数本だけ回す）"
	@echo "make paper   — 論文の表と図を results/paper/ から再生成（analysis/out/ に出る）"
	@echo "make verify  — 論文が述べている数値を生データから再計算して突き合わせる"

smoke:
	cd examples && PYTHONPATH=..:. MUJOCO_GL=egl $(PY) 01_dropin.py
	cd examples && PYTHONPATH=..:. MUJOCO_GL=egl $(PY) 02_hooks.py
	PYTHONPATH=.:examples MUJOCO_GL=egl $(PY) -m libero_ctrl.cli run \
	  --policy 03_policy_adapter:MyPolicy --axis camera --level L2 \
	  --suite libero_spatial --task 0 --limit 2 --out /tmp/libero_ctrl_smoke

paper:
	$(PY) analysis/gen_tab_decomp.py     > analysis/out/tab_decomp.tex
	$(PY) analysis/fig_policy_profiles.py
	$(PY) analysis/fig_composition.py
	$(PY) analysis/aggregate.py
	@echo "-> analysis/out/"

verify:
	$(PY) analysis/manuscript_numbers.py
	$(PY) analysis/single_axis_check.py
	$(PY) analysis/repeat_compare.py
	$(PY) analysis/full_independent.py
	$(PY) analysis/repB_recompute.py

clean-out:
	find analysis/out -type f -name '*' -delete
