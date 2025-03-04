import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.functions import fix_na_except_date, generate_teeth_order

from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    BLEEDING_PATH,
    CHART_ENDO_PATH,
    CHART_GENERAL_PATH,
    CHART_RESTORE_PATH,
    DEMOGRAPHIC_PATH,
    MOBILITY_FURCATION_INDEX_MAG_PATH,
    POCKETS_PATH,
    RECESSIONS_PATH,
    SUPPURATION_PATH
)

## Dataframe Processing ##

def process_dataframe(df):
    """
    Process a DataFrame to replace missing values or whitespace-only cells with pandas' <NA>.
    Returns the processed DataFrame.
    """
    # Title Changes
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Date changes
    if 'CHART DATE' in df.columns:
        df['CHART DATE'] = pd.to_datetime(df['CHART DATE'], errors='coerce')
        df = df.sort_values(by='CHART DATE').reset_index(drop=True)

    # Ensure CHART TITLE and CHART ID are consistently string type
    for col in ["CHART TITLE", "CHART ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Na Replacements except to CHART DATE
    df = fix_na_except_date(df, date_col="CHART DATE")

    return df

## Bleeding Data Loading ##

def load_bleeding_maxillary():
    """
    Load and preprocess the Maxillary worksheet from Bleeding.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(BLEEDING_PATH, sheet_name="Maxillary")
    df = process_dataframe(df)
    return df

def load_bleeding_mandibular():
    """
    Load and preprocess the Mandibular worksheet from Bleeding.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(BLEEDING_PATH, sheet_name="Mandibular")
    df = process_dataframe(df)
    return df

## Chart Endo Data Loading ##

def load_chart_endo():
    """
    Load and preprocess ChartEndo_Final.xlsx .
    """
    df = pd.read_excel(CHART_ENDO_PATH)
    df = process_dataframe(df)
    return df

## Chart General Data Loading ##

def load_chart_general():
    """
    Load and preprocess ChartGeneralNew_Final.xlsx .
    """
    df = pd.read_excel(CHART_GENERAL_PATH)
    df = process_dataframe(df)
    return df

## Chart Restorative Data Loading ##

def load_chart_restorative():
    """
    Load and preprocess ChartRestorative_Final.xlsx .
    """
    df = pd.read_excel(CHART_RESTORE_PATH)
    df = process_dataframe(df)
    return df

## Demographic Data Loading ##

def load_demographic_data():
    """
    Load and preprocess DemographicData.xlsx .
    """
    df = pd.read_excel(DEMOGRAPHIC_PATH)
    df = process_dataframe(df)
    return df

## Mobility Data Loading ##

def load_mobility():
    """
    Load and preprocess the Mobility worksheet from Mobility_Furcation_Index_MAG.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(MOBILITY_FURCATION_INDEX_MAG_PATH, sheet_name="Mobility")
    df = process_dataframe(df)
    return df

## Furcation Data Loading ##

def load_furcation():
    """
    Load and preprocess the Furcation worksheet from Mobility_Furcation_Index_MAG.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(MOBILITY_FURCATION_INDEX_MAG_PATH, sheet_name="Furcation")
    df = process_dataframe(df)
    return df

## Index Data Loading ##

def load_index():
    """
    Load and preprocess the Index worksheet from Mobility_Furcation_Index_MAG.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(MOBILITY_FURCATION_INDEX_MAG_PATH, sheet_name="Index")
    df = process_dataframe(df)
    return df

## MAG Data Loading ##

def load_mag():
    """
    Load and preprocess the MAG worksheet from Mobility_Furcation_Index_MAG.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(MOBILITY_FURCATION_INDEX_MAG_PATH, sheet_name="MAG")
    df = process_dataframe(df)
    return df

## Pockets Data Loading ##

def load_pockets_maxillary():
    """
    Load and preprocess the Pocket_Maxillary worksheet from Pockets.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(POCKETS_PATH, sheet_name="Pocket_Maxillary")
    df = process_dataframe(df)
    return df

def load_pockets_mandibular():
    """
    Load and preprocess the Pocket_Mandibular worksheet from Pockets.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(POCKETS_PATH, sheet_name="Pocket_Mandibular")
    df = process_dataframe(df)
    return df

# Recessions Data Loading ##

def load_recession_maxillary():
    """
    Load and preprocess the Pocket_Maxillary worksheet from Recessions.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(RECESSIONS_PATH, sheet_name="Recession_Maxillary")
    df = process_dataframe(df)
    return df

def load_recession_mandibular():
    """
    Load and preprocess the Pocket_Mandibular worksheet from Recessions.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(RECESSIONS_PATH, sheet_name="Recession_Mandibular")
    df = process_dataframe(df)
    return df

# Suppuration Data Loading ##

def load_suppuration_maxillary():
    """
    Load and preprocess the Maxillary worksheet from Suppuration.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(SUPPURATION_PATH, sheet_name="Maxillary")
    df = process_dataframe(df)
    return df

def load_suppuration_mandibular():
    """
    Load and preprocess the Mandibular worksheet from Suppuration.xlsx.
    Returns a DataFrame.
    """
    df = pd.read_excel(SUPPURATION_PATH, sheet_name="Mandibular")
    df = process_dataframe(df)
    return df