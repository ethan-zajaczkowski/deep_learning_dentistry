import os
import sys

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

# Adding BASE_DIR to sys.path for consistent imports
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Paths to raw data files
BLEEDING_PATH = os.path.join(BASE_DIR, "data/raw/Bleeding.xlsx")
CHART_ENDO_PATH = os.path.join(BASE_DIR, "data/raw/ChartEndo_Final.xlsx")
CHART_GENERAL_PATH = os.path.join(BASE_DIR, "data/raw/ChartGeneralNew_Final.xlsx")
CHART_RESTORE_PATH = os.path.join(BASE_DIR, "data/raw/ChartRestorative_Final.xlsx")
DEMOGRAPHIC_PATH = os.path.join(BASE_DIR, "data/raw/DemographicData.xlsx")
MOBILITY_FURCATION_INDEX_MAG_PATH = os.path.join(BASE_DIR, "data/raw/Mobility_Furcation_Index_MAG.xlsx")
POCKETS_PATH = os.path.join(BASE_DIR, "data/raw/Pockets.xlsx")
RECESSIONS_PATH = os.path.join(BASE_DIR, "data/raw/Recessions.xlsx")
SUPPURATION_PATH = os.path.join(BASE_DIR, "data/raw/Suppuration.xlsx")

# Paths to cleaned datasets
BLEEDING_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/bleeding_cleaned.xlsx")
CHART_ENDO_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/chart_endo_cleaned.xlsx")
CHART_GENERAL_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/chart_general_cleaned.xlsx")
CHART_RESTORE_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/chart_restorative_cleaned.xlsx")
FURCATION_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/furcation_cleaned.xlsx")
DEMOGRAPHIC_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/demographic_data_cleaned.xlsx")
INDEX_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/index_cleaned.xlsx")
MAG_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/mag_cleaned.xlsx")
MOBILITY_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/mobility_cleaned.xlsx")
POCKETS_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/pockets_cleaned.xlsx")
RECESSIONS_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/recessions_cleaned.xlsx")
SUPPURATION_CLEANED_PATH = os.path.join(BASE_DIR, "data/cleaned/suppuration_cleaned.xlsx")

# Paths to processed datasets
BLEEDING_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/bleeding_processed.csv")
CHART_ENDO_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/chart_endo_processed.csv")
CHART_GENERAL_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/chart_general_processed.csv")
CHART_RESTORE_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/chart_restore_processed.csv")
FURCATION_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/furcation_processed.csv")
DEMOGRAPHIC_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/demographic_processed.csv")
INDEX_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/index_processed.csv")
MAG_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/mag_processed.csv")
MOBILITY_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/mobility_processed.csv")
POCKETS_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/pockets_processed.csv")
RECESSIONS_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/recessions_processed.csv")
SUPPURATION_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/suppuration_processed.csv")

# Other Paths to main curated/processed datasets
CURATED_PROCESSED_PATH = os.path.join(BASE_DIR, "data/curated_dataset.csv")
TEMP_PATH = os.path.join(BASE_DIR, "data/curated_dataset_sample.xlsx")