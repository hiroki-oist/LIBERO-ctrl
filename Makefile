# LIBERO-CTRL
# PY must point at a python where LIBERO works.
PY ?= python3

.PHONY: help smoke paper verify figs clean-out

help:
	@echo "make smoke   - check that all three entry levels run (a handful of rollouts)"
	@echo "make paper   - regenerate the paper tables and figures from results/paper/ into analysis/out/"
	@echo "make verify  - recompute the numbers the paper states, from the raw records"
	@echo "make figs    - re-render docs/figs/perturbation_grid.png (needs a GPU)"

smoke:
	cd examples && PYTHONPATH=..:. MUJOCO_GL=egl $(PY) 02_dropin.py
	cd examples && PYTHONPATH=..:. MUJOCO_GL=egl $(PY) 03_hooks.py
	PYTHONPATH=.:examples MUJOCO_GL=egl $(PY) -m libero_ctrl.cli run \
	  --policy 01_policy_adapter:MyPolicy --axis camera --level L2 \
	  --suite libero_spatial --task 0 --limit 2 --out /tmp/libero_ctrl_smoke

paper:
	@mkdir -p analysis/out
	$(PY) analysis/gen_tab_decomp.py     > analysis/out/tab_decomp.tex
	$(PY) analysis/fig_policy_profiles.py
	$(PY) analysis/fig_axis_grid.py
	$(PY) analysis/fig_composition.py
	$(PY) analysis/aggregate.py
	@echo "-> analysis/out/"

verify:
	@mkdir -p analysis/out
	$(PY) analysis/manuscript_numbers.py
	$(PY) analysis/emergent_compensated.py
	$(PY) analysis/single_axis_check.py
	$(PY) analysis/repeat_compare.py
	$(PY) analysis/full_independent.py
	$(PY) analysis/repB_recompute.py

figs:
	MUJOCO_GL=egl $(PY) analysis/fig_perturbation_grid.py
	MUJOCO_GL=egl $(PY) analysis/fig_perturbation_grid.py --paper

clean-out:
	find analysis/out -type f -name '*' -delete
