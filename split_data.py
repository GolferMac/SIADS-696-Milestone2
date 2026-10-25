import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/clean_steam_games.parquet")
df["release_date"] = pd.to_datetime(df["release_date"])
df = df.sort_values("release_date").reset_index(drop=True)

identifier_text_cols = [
    "appID",
    "name",
    "detailed_description",
    "short_description",
]

raw_engineered_cols = [
    "supported_languages",
    "full_audio_languages",
    "developers",
    "publishers",
    "categories",
    "genres",
    "tags",
]

leakage_cols = [
    "estimated_owners",      # used to create target
    "peak_ccu",
    "reviews",
    "metacritic_url",
    "user_score",
    "positive",
    "negative",
    "score_rank",
    "recommendations",
    "achievements",
]

cols_to_drop = (
    identifier_text_cols
    + raw_engineered_cols
    + leakage_cols
)

df_model = df.drop(
    columns=[col for col in cols_to_drop if col in df.columns]
).copy()

print(f"Modeling dataset: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")

# ---------------------------------------------------------------
# Create random stratified 70/15/15 train, validation, test splits
# ---------------------------------------------------------------

train, temp = train_test_split(
    df_model,
    test_size=0.30,
    stratify=df_model["owner_group"],
    random_state=42,
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["owner_group"],
    random_state=42,
)

# ---------------------------
# Display split information
# ---------------------------
print("\nRandom Stratified Split")
print("=" * 75)

for name, data in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(
        f"{name:12s}: "
        f"{len(data):>7,} "
        f"({len(data) / len(df_model):6.2%})"
    )

print("=" * 75)

# Remove release_date
train = train.drop(columns="release_date")
val = val.drop(columns="release_date")
test = test.drop(columns="release_date")

# Save splits to parquet files
train.to_parquet("data/train_steam_games.parquet", index=False)
val.to_parquet("data/val_steam_games.parquet", index=False)
test.to_parquet("data/test_steam_games.parquet", index=False)

# Check distributions of target variable
owner_dist = pd.concat({
    "Train": train["owner_group"].value_counts(),
    "Validation": val["owner_group"].value_counts(),
    "Test": test["owner_group"].value_counts(),
}, axis=1).fillna(0).astype(int)

# Add percentages within each split
owner_pct = owner_dist.div(owner_dist.sum(axis=0), axis=1) * 100

print("\nOwner Group Counts")
print(owner_dist)

print("\nOwner Group % Within Each Split")
print(owner_pct.round(2))