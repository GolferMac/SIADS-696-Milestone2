# What Makes a Video Game Successful?  
This SIADS-696 Milestone II project examines which Steam game characteristics are associated with commercial success, measured by estimated ownership. We use exploratory analysis, supervised classification, and unsupervised clustering.  
(UMSI MADS Milestone II repository for FA26 Group 1)

**Team Members**:  
Matthew Bellovich (mbellov)  
Yi-Ju Chen (yvettec)  
Yi-Jui Chen (ivachen)  


## Loading Data

Install the project's Python dependencies and GNU Make, then open a terminal in the repository root.

```powershell
# View the available targets
Get-Content Makefile

# Preview the default target without running it
make -n

# Execute the default target
make
```

The data preparation workflow uses `load_data.py` to retrieve the [FronkonGames Steam Games dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) and `clean_data.py` to produce the cleaned analysis data. For individual steps, use the target names defined in Makefile.

Large generated Parquet files are excluded from Git and should be regenerated locally.

## File Descriptions

| File | Description |
|---|---|
| `Makefile` | Automates data loading and initial cleaning. |
| `load_data.py` | Loads the source Steam dataset and saves raw data. |
| `clean_data.py` | Preprocesses raw data into a cleaned dataset. |
| `EDA.ipynb` | Initial exploratory data analysis. |
| `clean_data_EDA.ipynb` | EDA on cleaned features |
| `supervised_models_partA.ipynb` | Supervised learning tasks. |
| `PartB_Unsupervised_Learning.ipynb` | Unsupervised learning tasks. |

## Data Source

[FronkonGames Steam Games Dataset on Hugging Face](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
