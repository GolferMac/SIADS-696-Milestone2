"""
This script prepares the raw Steam games dataset for exploratory analysis
and machine learning. It removes unnecessary or unsuitable columns, cleans
selected fields, and engineers features that provide structured representations
of game characteristics.

Major processing steps include:
    - Dropping columns that are unnecessary, redundant, or unsuitable for modeling.
    - Creating game age features from release dates and snapshot pulled date.
    - Creating historical developer and publisher features based on prior releases.
    - Standardizing supported languages and creating language-based features.
    - Cleaning platform availability features for Windows, Mac, and Linux.
    - Encoding core video game genres as binary features.
    - Creating consolidated category features describing gameplay modes,
      multiplayer functionality, controller/VR support, monetization, and
      Steam/community functionality.

Steps NOT included:
    - Data visualization and exploratory analysis.
    - Feature normalization or scaling.
    - TTS or cross-validation splits for machine learning.

The final cleaned dataset is saved as a Parquet file for efficient storage and retrieval.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re


snapshot_date = pd.to_datetime(
    "Sat, 26 Sep 2026 10:59:16 GMT"     # Date of the snapshot of the dataset, used to calculate age of games
).tz_localize(None)

START_YEAR = 2015   # Inclusive
END_YEAR = 2026     # Exclusive

# Top n languages to one-hot encode
top_n = 10

df = pd.read_parquet("data/raw_steam_games.parquet")

# Columns to drop

cols_to_drop = [
    'dlc_count',
    'header_image', 
    'website', 
    'support_url', 
    'support_email',
    'notes',
    'average_playtime_forever',
    'average_playtime_2weeks',
    'median_playtime_forever',
    'median_playtime_2weeks',
    'screenshots', 
    'movies', 
    'packages',
]

df = df.drop(columns=cols_to_drop)

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

df['age'] = (snapshot_date - df['release_date']).dt.days / 365.25

# --------------------------------
# Publisher and Developer Features
# --------------------------------

def add_prior_games_features(df, entity_col, prefix):
    """
    Create historical experience features for entities associated with each game.

    Features are calculated using only games released BEFORE the current
    game's release date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing game data.

    entity_col : str
        Column containing lists/arrays of entities, e.g.
        "publishers" or "developers".

    prefix : str
        Prefix used for the new feature columns, e.g.
        "publisher" or "developer".

    Returns
    -------
    pd.DataFrame
        DataFrame with three additional features:
        - {prefix}_prior_games_max
        - {prefix}_prior_games_mean
        - {prefix}_all_first_release
        - {prefix}_any_first_release
    """
    
    df = df.sort_values("release_date").reset_index(drop=True)
    game_count = defaultdict(int)

    max_col = f"{prefix}_prior_games_max"
    mean_col = f"{prefix}_prior_games_mean"
    all_first_col = f"{prefix}_all_first_release"
    any_first_col = f"{prefix}_any_first_release"

    df[max_col] = 0
    df[mean_col] = 0.0
    df[all_first_col] = 0
    df[any_first_col] = 0

    for date, group in df.groupby("release_date", sort=True):

        # Calculate features for each group
        for idx in group.index:
            entities = df.at[idx, entity_col]

            if len(entities) > 0:
                counts = [
                    game_count[entity]
                    for entity in entities
                ]

                df.at[idx, max_col] = max(counts)
                df.at[idx, mean_col] = np.mean(counts)
                # all entities are releasing first game
                df.at[idx, all_first_col] = int(all(count == 0 for count in counts))
                # at least one entity is releasing first game
                df.at[idx, any_first_col] = int(any(count == 0 for count in counts))

        # Add games from this date to entity history
        for idx in group.index:
            entities = df.at[idx, entity_col]

            if len(entities) > 0:
                for entity in entities:
                    game_count[entity] += 1

    return df

df = add_prior_games_features(df, "publishers", "pub")
df = add_prior_games_features(df, "developers", "dev")

# Filter for snapshot year (after publisher & developer features are added)
df = df[(df['release_year'] >= START_YEAR) & (df['release_year'] < END_YEAR)]

# ----------------------
# Supported Languages
# ----------------------

lang_map = {
    # Chinese
    "Simplified Chinese": "Chinese",
    "Traditional Chinese": "Chinese",

    # Spanish
    "Spanish - Spain": "Spanish",
    "Spanish - Latin America": "Spanish",

    # Portuguese
    "Portuguese - Brazil": "Portuguese",
    "Portuguese - Portugal": "Portuguese",

    # Punjabi
    "Punjabi (Gurmukhi)": "Punjabi",
    "Punjabi (Shahmukhi)": "Punjabi",
}

def standardize_languages(langs):
    mapped = [lang_map.get(lang, lang) for lang in langs]
    return list(dict.fromkeys(mapped))


df['supported_languages'] = (
    df['supported_languages']
    .apply(standardize_languages)
)

# Pull top n languages
language_counts = Counter(
    lang for languages 
    in df['supported_languages'].dropna() 
    for lang in languages
)

top_languages = [lang for lang, count in language_counts.most_common(top_n)]
top_languages_set = set(top_languages)

# Encode top n languages
def clean_feature_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

for lang in top_languages:
    df[f"lang_{clean_feature_name(lang)}"] = (
        df["supported_languages"]
        .apply(lambda x: int(lang in x))
    )

# Other languages count
df["other_langs"] = (
    df["supported_languages"]
    .apply(
        lambda langs: sum(
            lang not in top_languages_set 
            for lang in langs
        )
    )
)

# total_languages count
df["total_langs"] = (
    df["supported_languages"]
    .apply(len)
)


# ------------------
# Platform Features
# ------------------

# Convert T/F columns to int 
platform_cols = ["windows", "mac", "linux"]
df[platform_cols] = df[platform_cols].astype(int)

# create num_platforms
df["num_platforms"] = df[platform_cols].sum(axis=1)


# ------------------
# Genre Features
# ------------------

GENRES = [
    "Action",
    "Adventure",
    "Casual",
    "Early Access",
    "Free To Play",
    "Indie",
    "Massively Multiplayer",
    "Racing",
    "RPG",
    "Simulation",
    "Sports",
    "Strategy",
    "Violent",
    "Gore",
]

def create_genre_features(df, genres=GENRES):
    """
    Create one-hot encoded features for each genre in the list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing game data.

    genres : list of str
        List of genres to create features for.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded genre features.
    """
    
    for genre in genres:
        col = f"genre_{clean_feature_name(genre)}"
        
        df[col] = df["genres"].apply(
            lambda genres: int(genre in genres)
        )

        df["num_game_genres"] = (
            df["genres"]
            .apply(lambda genres: sum(g in genres for g in genres))
        )

    return df

df = create_genre_features(df)

# ------------------
# Category Features
# ------------------

def create_category_features(df):
    """
    Create consolidated features from Steam categories.
    """

    def has_any(categories, values):
        return int(any(value in categories for value in values))

    # Game modes
    df["cat_single_player"] = df["categories"].apply(
        lambda x: int("Single-player" in x)
    )

    df["cat_multiplayer"] = df["categories"].apply(
        lambda x: has_any(x, [
            "Multi-player",
            "PvP",
            "Online PvP",
            "LAN PvP",
            "Cross-Platform Multiplayer",
            "Shared/Split Screen PvP",
        ])
    )

    df["cat_coop"] = df["categories"].apply(
        lambda x: has_any(x, [
            "Co-op",
            "Online Co-op",
            "LAN Co-op",
            "Shared/Split Screen Co-op",
        ])
    )

    df["cat_online"] = df["categories"].apply(
        lambda x: has_any(x, [
            "Online PvP",
            "Online Co-op",
            "Cross-Platform Multiplayer",
        ])
    )

    df["cat_local_multiplayer"] = df["categories"].apply(
        lambda x: has_any(x, [
            "Shared/Split Screen",
            "Shared/Split Screen PvP",
            "Shared/Split Screen Co-op",
            "LAN PvP",
            "LAN Co-op",
        ])
    )

    df["cat_mmo"] = df["categories"].apply(
        lambda x: int("MMO" in x)
    )

    # Controller support
    df["cat_controller_support"] = df["categories"].apply(
        lambda x: has_any(x, [
            "Full controller support",
            "Partial Controller Support",
        ])
    )

    # VR
    df["cat_vr"] = df["categories"].apply(
        lambda x: has_any(x, [
            "VR Only",
            "VR Supported",
            "VR Support",
        ])
    )

    # Steam/community features
    df["cat_achievements"] = df["categories"].apply(
        lambda x: int("Steam Achievements" in x)
    )

    df["cat_trading_cards"] = df["categories"].apply(
        lambda x: int("Steam Trading Cards" in x)
    )

    df["cat_leaderboards"] = df["categories"].apply(
        lambda x: int("Steam Leaderboards" in x)
    )

    df["cat_workshop"] = df["categories"].apply(
        lambda x: int("Steam Workshop" in x)
    )

    # Monetization
    df["cat_in_app_purchases"] = df["categories"].apply(
        lambda x: int("In-App Purchases" in x)
    )

    return df

df = create_category_features(df)

# Save to new parquet file
df.to_parquet("data/clean_steam_games.parquet", index=False)