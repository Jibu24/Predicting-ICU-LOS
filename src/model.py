from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, 
                           param_grid=None, cv=5, n_jobs=-1, verbose=1):
    """
    Train and evaluate an SVM model with hyperparameter tuning and feature selection
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        param_grid: Custom parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        dict: Dictionary containing best parameters, test R² score, and best estimator
    """
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'select__k': [40, 55, 75, 100, 'all'],
            'svr__C': [0.1, 1, 10],
            'svr__epsilon': [0.1, 0.2, 0.5],
            'svr__kernel': ['rbf', 'linear', 'poly'],
            'svr__gamma': ['scale', 'auto'],
            'svr__degree': [2, 3]  # Only used for poly kernel
        }
    
    # Create pipeline
    pipeline = Pipeline([
        ('select', SelectKBest(score_func=f_regression)),
        ('svr', SVR())
    ])
    
    # Setup GridSearchCV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    # Train model
    grid.fit(X_train, y_train)
    
    # Evaluate on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Return results
    return {
        'best_params': grid.best_params_,
        'test_r2': r2,
        'best_estimator': best_model
    }