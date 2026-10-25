PYTHON ?= python
RAW := data/raw_steam_games.parquet
CLEAN := data/clean_steam_games.parquet

.PHONY: all raw clean rebuild clean-files help
all: clean
raw: $(RAW)
clean: $(CLEAN)

$(RAW): load_data.py
	@mkdir -p data
	$(PYTHON) load_data.py

$(CLEAN): clean_data.py $(RAW)
	@mkdir -p data
	$(PYTHON) clean_data.py

rebuild:
	@mkdir -p data
	$(PYTHON) load_data.py
	$(PYTHON) clean_data.py

clean-files:
	rm -f $(CLEAN)

help:
	@echo 'make raw      Download the raw snapshot if absent or stale'
	@echo 'make clean    Generate the cleaned parquet if absent or stale'
	@echo 'make rebuild  Force both pipeline steps'
	@echo 'make clean-files  Delete only the generated cleaned parquet'
