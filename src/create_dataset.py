import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
#import statsmodels.api as sm

class MIMIC_IV:
    def __init__(self, raw_path):
        """_summary_

        Args:
            raw_path: path to the data folder
            output_path : where to save the cleaned dataset
        """
        self.raw_path = raw_path
        self.df = None

    def load_data(self):
        """Load and merges multiple raw data files into one DataFrame"""
        admissions_df = pd.read_csv(self.raw_path + '/hosp/admissions.csv.gz')
        icu_df = pd.read_csv(self.raw_path + '/icu/icustays.csv.gz')
        drg_df = pd.read_csv(self.raw_path + '/hosp/drgcodes.csv.gz')
        drg_df = drg_df[drg_df['drg_type'] != 'HCFA']
        diag_df = pd.read_csv(self.raw_path + '/hosp/diagnoses_icd.csv.gz')
        diag_df = diag_df[diag_df['icd_code'].str.startswith('8060')]
        diag_df = diag_df.drop_duplicates(subset=["subject_id", "hadm_id"])
        d_diag_df = pd.read_csv(self.raw_path + '/hosp/d_icd_diagnoses.csv.gz')
        patients_df = pd.read_csv(self.raw_path + '/hosp/patients.csv.gz')
        
        # Merge dataframes
        self.df = diag_df.merge(admissions_df, on=['subject_id', 'hadm_id'], how='left') \
                             .merge(patients_df, on='subject_id', how='left') \
                             .merge(icu_df, on=['subject_id', 'hadm_id'], how='left') \
                             .merge(drg_df, on=['subject_id', 'hadm_id'], how='left')
        self.df.drop_duplicates(subset=["subject_id", "hadm_id"], inplace=True)
    
    def clean_data(self):