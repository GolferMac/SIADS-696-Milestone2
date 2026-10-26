PYTHON ?= python
RAW := data/raw_steam_games.parquet
CLEAN := data/clean_steam_games.parquet

.PHONY: all raw clean rebuild clean-files help
all: clean
raw: $(RAW)
clean: $(CLEAN)

$(RAW): load_data.py
	@$(PYTHON) -c "from pathlib import Path; Path('data').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) load_data.py

$(CLEAN): clean_data.py $(RAW)
	@$(PYTHON) -c "from pathlib import Path; Path('data').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) clean_data.py

rebuild:
	@$(PYTHON) -c "from pathlib import Path; Path('data').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) load_data.py
	$(PYTHON) clean_data.py

clean-files:
	$(PYTHON) -c "from pathlib import Path; Path('data/clean_steam_games.parquet').unlink(missing_ok=True)"

help:
	@echo 'make raw      Download the raw snapshot if absent or stale'
	@echo 'make clean    Generate the cleaned parquet if absent or stale'
	@echo 'make rebuild  Force both pipeline steps'
	@echo 'make clean-files  Delete only the generated cleaned parquet'
