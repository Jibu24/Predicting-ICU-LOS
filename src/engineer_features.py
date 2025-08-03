import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, df, target_col, ordinal_vars=None, ordinal_mapping=None, test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target_col = target_col
        self.ordinal_vars = ordinal_vars or []
        #self.ordinal_mapping = ordinal_mapping or [['unknown', 'single', 'married', 'divorced', 'widowed']]
        self.test_size = test_size
        self.random_state = random_state

        self.nominal_vars = [col for col in self.df.select_dtypes(include='object').columns if col not in self.ordinal_vars]
        self.num_cols = self.df.select_dtypes(include=np.number).columns.difference([self.target_col]).tolist()
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Numerical pipeline
        num_pipeline = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', StandardScaler())
        ])

        # Nominal categorical pipeline
        nominal_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Ordinal categorical pipeline
        #ordinal_pipeline = Pipeline([
            #('ordinal', OrdinalEncoder(categories=self.ordinal_mapping))
        #])

        # ColumnTransformer to combine
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('nominal', nominal_pipeline, self.nominal_vars)
            #('ordinal', ordinal_pipeline, self.ordinal_vars)
        ])

        return preprocessor

    def transform(self):
        X = self.df.drop(columns=[self.target_col])
        X_preprocessed = self.pipeline.fit_transform(X)
        return X_preprocessed

    def split(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test