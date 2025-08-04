import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

import matplotlib.pyplot as plt
#import statsmodels.api as sm

class MIMIC_III:
    def __init__(self, raw_path):
        """_summary_

        Args:
            raw_path: path to the data folder
        """
        self.raw_path = raw_path
        self.df = None
        self.raw_df = None # Store pre-cleaned data
    
    def load_data(self):
        """Load and merge datasets with proper handling"""
        
        # Use path.join for OS-agnostic paths
        # Load data with error handling
        try:
            admissions = pd.read_csv(os.path.join(self.raw_path, 'ADMISSIONS.csv.gz'))
            icu = pd.read_csv(os.path.join(self.raw_path, 'ICUSTAYS.csv.gz'))
            drg = pd.read_csv(os.path.join(self.raw_path, 'DRGCODES.csv.gz'))
            #drg = drg_df[drg_df['DRG_TYPE'] == 'APR']
            diag = pd.read_csv(os.path.join(self.raw_path, 'DIAGNOSES_ICD.csv.gz'))
            #diag = diag_df[diag_df['ICD9_CODE'].str.startswith('8060', na=False)]
            d_diag = pd.read_csv(os.path.join(self.raw_path, 'D_ICD_DIAGNOSES.csv.gz'))
            patients = pd.read_csv(os.path.join(self.raw_path, 'PATIENTS.csv.gz'))
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        
        # Pre-filter data
        diag = diag[diag['ICD9_CODE'].str.startswith('8060', na=False)]
        drg = drg[drg['DRG_TYPE'] != 'HCFA']
        drg = drg[drg['DRG_TYPE'] != 'MS']
        
        #Drop ROW ID column
        df_list = [admissions, icu, drg, diag, d_diag, patients]
        for df in df_list:
            if 'ROW_ID' in df.columns:
                df.drop(columns=['ROW_ID'], inplace=True, errors='ignore')
        
        # Handle DRG duplicates - keep first code per admission
        drg = drg.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'])
        
        # Stepwise merging
        merged = diag.merge(admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        merged = merged.merge(patients, on='SUBJECT_ID', how='left')
        merged = merged.merge(icu, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        merged = merged.merge(drg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        
        # Store raw and working copies
        self.raw_df = merged.copy()
        self.df = merged
        
    def clean_data(self):
        """ Safe cleaning with validation"""
        if self.df is None:
            raise ValueError("Load data first using load_data()")
        
        # Identify columns to drop
        drop_cols = [
            'DOD', 'DOD_HOSP', 'DOD_SSN', 'DISCHTIME', 'DEATHTIME',
            'DISCHARGE_LOCATION', 'RELIGION', 'DIAGNOSIS',
            'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA', 
            'EXPIRE_FLAG', 'ICUSTAY_ID', 'DBSOURCE',
            'FIRST_WARDID', 'LAST_WARDID', 'OUTTIME', 
            'DRG_TYPE'
        ]
        
        # selective dropping
        existing_drop_cols = [col for col in drop_cols if col in self.df.columns]
        self.df.drop(columns=existing_drop_cols, inplace=True)
        
        # Handle missing values
        self.df['MARITAL_STATUS'] = self.df['MARITAL_STATUS'].fillna('unknown')
        self.df['LANGUAGE'] = self.df['LANGUAGE'].fillna('?')
        
        #Calculate ED LOS only for ED admissions
        if 'EDREGTIME' in self.df.columns and 'EDOUTTIME' in self.df.columns:
            self.df['EDREGTIME'] = pd.to_datetime(self.df['EDREGTIME'], errors='coerce')
            self.df['EDOUTTIME'] = pd.to_datetime(self.df['EDOUTTIME'], errors='coerce')
            self.df['ed_los'] = (self.df['EDOUTTIME'] - self.df['EDREGTIME']).dt.total_seconds() / 60
            self.df.drop(columns=['EDREGTIME', 'EDOUTTIME'], inplace=True)
            self.df['ed_los'] = self.df['ed_los'].fillna(0)
        else:
            self.df['ed_los'] = 0
            
        # Final cleanup
        self.df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"], inplace=True, keep='first')
        self.df.dropna(subset=['LOS'], inplace=True)
        self.df.columns = self.df.columns.str.lower()
        self.df.rename(columns={'icd9_code': 'icd_code','ethnicity': 'race'}, inplace=True)
        
        # Convert 'dob' to datetime
        self.df['dob'] = pd.to_datetime(self.df['dob'])
        # Convert 'admit_time' to datetime
        self.df['admittime'] = pd.to_datetime(self.df['admittime'], errors='coerce')
        # Create 'anchor_age' by subtracting the year of 'dob' from the year of 'admittime'
        self.df['anchor_age'] = self.df['admittime'].dt.year - self.df['dob'].dt.year
        self.df.drop(columns=['dob'], inplace=True, errors='ignore')
        return self.df  # Return cleaned DataFrame for chaining
        
    def reset_data(self):
        """Revert to original loaded data"""
        self.df = self.raw_df.copy()

class Dataset:
    def __init__(self, MIMIC_IV_raw_path, MIMIC_III_raw_path):
        self.MIMIC_IV = MIMIC_IV(MIMIC_IV_raw_path)
        self.MIMIC_III = MIMIC_III(MIMIC_III_raw_path)
        self.df = None
        
    def load_data(self):
        """Load and merges multiple raw data files into one DataFrame"""
        self.MIMIC_IV.load_data()
        self.MIMIC_III.load_data()
        
    def clean_data(self):
        """Cleans the loaded data"""
        self.MIMIC_IV.clean_data()
        self.MIMIC_III.clean_data()
    
    def build(self):
        """Builds the final dataset by merging MIMIC-IV and MIMIC-III data"""
        self.df = pd.concat([self.MIMIC_IV.df, self.MIMIC_III.df], ignore_index=True)
        self.df.drop_duplicates(subset=["subject_id", "hadm_id"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
    def drop_columns(self, cols):
        """Drops specified columns if they exist"""
        if self.df is not None:
            self.df.drop(columns=cols, inplace=True, errors='ignore')
        else:
            print("DataFrame is empty. Load data before dropping columns.")
    def drop_na(self, subset=None):
        """Drops rows with NaN values in specified columns"""
        if self.df is not None:
            self.df.dropna(subset=subset, inplace=True)
        else:
            print("DataFrame is empty. Load data before dropping NaN values.")
    def extract_datetime_components(self, datetime_col, components):
        """
        Extracts one or more datetime components from a specified column and
        adds them as new columns to the internal DataFrame.

        Args:
            datetime_col (str): Name of the datetime column.
            components (list or str): Components to extract ('month', 'day', 'hour', 'weekday')
        """
        if datetime_col not in self.df.columns:
            raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col], errors='coerce')

        if isinstance(components, str):
            components = [components]

        for component in components:
            new_col = f"{datetime_col}_{component}"
            if component == 'month':
                self.df[new_col] = self.df[datetime_col].dt.month_name()
            elif component == 'day':
                self.df[new_col] = self.df[datetime_col].dt.day_name()
            elif component == 'hour':
                self.df[new_col] = self.df[datetime_col].dt.hour
            elif component == 'weekend':
                self.df[new_col] = self.df[datetime_col].dt.dayofweek.isin([5,6])
            else:
                raise ValueError(f"Unsupported component: '{component}'")
    def change_datatype(self, columns, dtype):
        """
        Changes the datatype of specified columns in the DataFrame.

        Args:
            columns (list): List of column names to change.
            dtype (str or type): Desired datatype (e.g., 'int', 'float', 'str').
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be provided as a list.")
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype)
            else:
                print(f"Column '{col}' does not exist in DataFrame.")
    def log_transform(self, columns):
        """
        Applies log transformation to specified columns in the DataFrame.

        Args:
            columns (list): List of column names to apply log transformation.
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be provided as a list.")
        
        for col in columns:
            if col in self.df.columns:
                self.df[f'log_{col}'] = np.log1p(self.df[col])
                self.df.drop(columns=[col], inplace=True)
            else:
                print(f"Column '{col}' does not exist in DataFrame.")


class MIMIC_IV:
    def __init__(self, raw_path):
        """
        Args:
            raw_path: Path to the MIMIC-IV dataset folder
        """
        self.raw_path = raw_path
        self.df = None
        self.raw_df = None  # Store pre-cleaned data

    def load_data(self):
        """Load and merge datasets with proper handling"""
        # Use path.join for OS-agnostic paths
        hosp_path = os.path.join(self.raw_path, 'hosp')
        icu_path = os.path.join(self.raw_path, 'icu')
        
        # Load data with error handling
        try:
            admissions = pd.read_csv(os.path.join(hosp_path, 'admissions.csv.gz'))
            patients = pd.read_csv(os.path.join(hosp_path, 'patients.csv.gz'))
            diag = pd.read_csv(os.path.join(hosp_path, 'diagnoses_icd.csv.gz'))
            drg = pd.read_csv(os.path.join(hosp_path, 'drgcodes.csv.gz'))
            icu = pd.read_csv(os.path.join(icu_path, 'icustays.csv.gz'))
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None

        # Pre-filter data
        diag = diag[diag['icd_code'].str.startswith('8060', na=False)]
        drg = drg[drg['drg_type'] != 'HCFA']
        
        # Handle DRG duplicates - keep first code per admission
        drg = drg.drop_duplicates(subset=['subject_id', 'hadm_id'])

        # Stepwise merging
        merged = diag.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
        merged = merged.merge(patients, on='subject_id', how='left')
        merged = merged.merge(icu, on=['subject_id', 'hadm_id'], how='left')  # Left join!
        merged = merged.merge(drg, on=['subject_id', 'hadm_id'], how='left')
        
        # Store raw and working copies
        self.raw_df = merged.copy()
        self.df = merged
    
    def drop_columns(self, cols):
        """Drops specified columns if they exist"""
        if self.df is not None:
            self.df.drop(columns=cols, inplace=True, errors='ignore')
        else:
            print("DataFrame is empty. Load data before dropping columns.")
        

    def clean_data(self):
        """Safe cleaning with validation"""
        if self.df is None:
            raise ValueError("Load data first using load_data()")
        
        # Identify columns to drop
        drop_cols = [
            'dod', 'deathtime', 'dischtime', 'admit_provider_id',
            'drg_type', 'hospital_expire_flag', 'anchor_year_group',
            'outtime', 'stay_id'
        ]
        
        # Selective dropping
        existing_drop_cols = [col for col in drop_cols if col in self.df.columns]
        self.df.drop(columns=existing_drop_cols, inplace=True)

        # Handle missing values
        self.df['marital_status'] = self.df['marital_status'].fillna('UNKNOWN')
        
        # Calculate ED LOS only for ED admissions
        if 'edregtime' in self.df.columns and 'edouttime' in self.df.columns:
            self.df['edregtime'] = pd.to_datetime(self.df['edregtime'], errors='coerce')
            self.df['edouttime'] = pd.to_datetime(self.df['edouttime'], errors='coerce')
            self.df['ed_los'] = (self.df['edouttime'] - self.df['edregtime']).dt.total_seconds() / 60
            self.df.drop(columns=['edregtime', 'edouttime'], inplace=True)
            self.df['ed_los'] = self.df['ed_los'].fillna(0)
        else:
            self.df['ed_los'] = 0  # Default if no ED data
            
        # Final cleanup
        self.df.drop_duplicates(subset=["subject_id", "hadm_id"], inplace=True, keep='first')
        self.df.dropna(subset=['los'], inplace=True)
        return self.df  # Return cleaned DataFrame for chaining


    def reset_data(self):
        """Revert to original loaded data"""
        self.df = self.raw_df.copy()