# qwenputer-handover. `make check` is the gate: it runs anywhere (no model, no GPU).
# The run targets below need a CUDA machine; pick the GPU with CUDA_VISIBLE_DEVICES.

UV = env -u PYTHONOPTIMIZE uv run
FIRMWARE = firmware/calc-bridge.elf

.PHONY: check lint format typecheck test changelog changelog-check sync merge textcheck eval-rv1 eval-pytool replay

check: lint typecheck test changelog-check

sync:
	uv sync

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

format:
	$(UV) ruff format .
	$(UV) ruff check --fix .

typecheck:
	$(UV) mypy

# PYTHONOPTIMIZE strips assert, which the tests rely on; every uv run here unsets it.
test:
	$(UV) pytest -q

changelog:
	bun scripts/build-changelog.ts

changelog-check:
	bun scripts/build-changelog.ts --check

# ---- upstream (chipformer) steps on an exported checkpoint -----------------------------------
# SRC: a base-format checkpoint dir (`handover export` output, or the stock Qwen3.5-4B snapshot)
# MERGED: where the merged rv1 checkpoint goes; OUT: result dir; DATA / N: an eval set and size;
# BATCH: conversations the evals decode at once

BATCH ?= 1

merge:  # ~31 GB RAM; fits the anti-sink keys on CUDA when present
	$(UV) python -m chipformer.rv.merge $(SRC) --out $(MERGED)

textcheck:  # loads SRC, then MERGED (~19 GB bf16), on one CUDA device
	mkdir -p $(OUT)
	$(UV) python -m chipformer.rv.textcheck $(SRC) $(MERGED) --out $(OUT)/textcheck.json

eval-rv1:  # the merged model, its circuit lockstep-checked against the reference machine
	$(UV) python -m chipformer.rv.evalarith $(MERGED) --cond rv1 --data $(DATA) --n $(N) \
		--batch $(BATCH) --firmware $(FIRMWARE) --out $(OUT)

eval-pytool:  # the base-format model with the reference machine as its tool (greedy)
	$(UV) python -m chipformer.rv.evalarith $(SRC) --cond pytool --data $(DATA) --n $(N) \
		--batch $(BATCH) --firmware $(FIRMWARE) --out $(OUT)

replay:  # offline IR replay of rv1 records (no GPU)
	$(UV) python -m chipformer.rv.replay $(MERGED) $(OUT)/$(DATA).rv1.jsonl --firmware $(FIRMWARE)
