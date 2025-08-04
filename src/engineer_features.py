import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def remove_outliers(df: pd.DataFrame, column: str, method: str = "iqr") -> pd.DataFrame:
    """Remove outliers using IQR or Z-score method"""
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    elif method == "zscore":
        from scipy import stats
        return df[(np.abs(stats.zscore(df[column])) < 3)]
    else:
        raise ValueError(f"Unknown method: {method}")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Extract temporal features from datetime column"""
    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    return df.drop(columns=[date_column])

def add_interaction_terms(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create interaction features between two columns"""
    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    return df

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

def get_preprocessor_and_split(
    data: pd.DataFrame,
    target_col: str,
    numeric_features: list,
    categorical_features: list,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessing_steps: dict = None
) -> tuple:
    """
    Create preprocessing pipeline and split data into train/test sets
    
    Args:
        data: Input DataFrame with features and target
        target_col: Name of the target column
        numeric_features: List of numerical feature names
        categorical_features: List of categorical feature names
        test_size: Proportion for test split (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        preprocessing_steps: Custom preprocessing steps dictionary
            Example: {
                'numeric': [('imputer', SimpleImputer()), ('scaler', StandardScaler())],
                'categorical': [('encoder', OneHotEncoder())]
            }
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    # Separate features and target
    X = data.drop(columns=[target_col])
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    y = data[target_col]
    
    # Split data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Default preprocessing steps if not provided
    if preprocessing_steps is None:
        preprocessing_steps = {
            'numeric': [('scaler', StandardScaler())]
        }
    
    # Create column transformers
    numeric_transformer = Pipeline(steps=preprocessing_steps['numeric'])
    #categorical_transformer = Pipeline(steps=preprocessing_steps['categorical'])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
            #('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )
    
    # Fit and transform data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    return (
        X_train_preprocessed, 
        X_test_preprocessed, 
        y_train, 
        y_test,
        preprocessor  # Return fitted preprocessor for reuse
    )
