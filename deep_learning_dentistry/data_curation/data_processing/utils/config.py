import os
import sys

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

# Adding BASE_DIR to sys.path for consistent imports
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Paths to raw data files
BLEEDING_PATH = os.path.join(BASE_DIR, "data/raw/bleeding")
CHART_GENERAL_PATH = os.path.join(BASE_DIR, "data/raw/chart_general")
CHART_RESTORE_PATH = os.path.join(BASE_DIR, "data/raw/chart_restorative")
DEMOGRAPHIC_PATH = os.path.join(BASE_DIR, "data/raw/demographic_data")
MOBILITY_FURCATION_INDEX_MAG_PATH = os.path.join(BASE_DIR, "data/raw/mobility_furcation_index_mag")
POCKETS_PATH = os.path.join(BASE_DIR, "data/raw/pockets")
RECESSIONS_PATH = os.path.join(BASE_DIR, "data/raw/recessions")
SUPPURATION_PATH = os.path.join(BASE_DIR, "data/raw/suppuration")

# Paths to curated Excel datasets
BLEEDING_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/bleeding_curated.xlsx")
CHART_GENERAL_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/chart_general_curated.xlsx")
CHART_RESTORE_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/chart_restorative_curated.xlsx")
DEMOGRAPHICS_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/demographics_data.xlsx")
FURCATION_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/furcation_curated.xlsx")
INDEX_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/index_curated.xlsx")
MAG_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/mag_curated.xlsx")
MOBILITY_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/mobility_curated.xlsx")
POCKETS_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/pockets_curated.xlsx")
RECESSIONS_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/recessions_curated.xlsx")
SUPPURATION_CURATED_PATH_EXCEL = os.path.join(BASE_DIR, "data/curated/excel/suppurations_curated.xlsx")

# Paths to curated CSV datasets
BLEEDING_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/bleeding_curated.csv")
CHART_GENERAL_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/chart_general_curated.csv")
CHART_RESTORE_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/chart_restorative_curated.csv")
DEMOGRAPHICS_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/demographics_data_curated.csv")
FURCATION_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/furcation_curated.csv")
INDEX_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/index_curated.csv")
MAG_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/mag_curated.csv")
MOBILITY_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/mobility_curated.csv")
POCKETS_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/pockets_curated.csv")
RECESSIONS_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/recessions_curated.csv")
SUPPURATION_CURATED_PATH_CSV = os.path.join(BASE_DIR, "data/curated/csv/suppurations_curated.csv")

# Other Paths to main curated/processed datasets
FULL_DATASET_PATH = os.path.join(BASE_DIR, "data/full_dataset")
ANALYSIS_DATASET_PATH = os.path.join(BASE_DIR, "data/analysis")
TEMP_PATH = os.path.join(BASE_DIR, "data/curated_dataset_sample.xlsx")