from datasets import load_dataset
import pandas as pd

ds = load_dataset("FronkonGames/steam-games-dataset", split="train")

df = ds.to_pandas()

df.to_parquet("data/raw_steam_games.parquet", index=False)