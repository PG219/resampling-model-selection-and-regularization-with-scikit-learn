"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    # TODO: load_diabetes(as_frame=True); return (X DataFrame, y Series)

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X, y

    
    

def describe_data(X, y):
    # TODO: {'n', 'p', 'features', 'y_mean' rounded to 2}



    return {'n':X.shape[0], 'p':X.shape[1], 'features': list(X.columns), 'y_mean': round(float(y.mean()), 2)}

# Step 2 - train_test
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    # TODO: train_test_split; return (X_train, X_test, y_train, y_test)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return (X_train, X_test, y_train, y_test)

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    # TODO: PolynomialFeatures(degree) -> LinearRegression
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())
    

def validation_set_curve(X, y, feature, degrees, random_state):
    # TODO: 50/50 split of X[[feature]]; {degree: validation MSE rounded to 1 decimal}

    X_sub = X[[feature]]
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y, test_size=0.5, random_state=random_state
    )

    result ={}


    for deg in degrees:
        model = poly_model(deg)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        result[deg] = round(float(mse), 1)
        
    return result


def curve_spread(X, y, feature, degrees, seeds):
    # TODO: per degree, max - min of the validation MSE across seeds, rounded to 1 decimal
    mse_per_degree = {deg: [] for deg in degrees}
    
    for seed in seeds:
        # Get the validation curve for the current seed
        seed_curve = validation_set_curve(X, y, feature, degrees, random_state=seed)
        for deg in degrees:
            mse_per_degree[deg].append(seed_curve[deg])
            
    # Calculate max - min for each degree and round to 1 decimal place
    spread_result = {}
    for deg in degrees:
        spread = max(mse_per_degree[deg]) - min(mse_per_degree[deg])
        spread_result[deg] = round(float(spread), 1)
        
    return spread_result

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    Kf = KFold(n_splits = k, shuffle=True, random_state = random_state)
    neg_mse_scores = cross_val_score(estimator, X, y, cv = Kf, scoring='neg_mean_squared_error')


    mse_scores = -neg_mse_scores

    mean_mse = np.mean(mse_scores)


    se_mse = np.std(mse_scores)

    mean_mse = np.mean(mse_scores)

    se_mse = np.std(mse_scores, ddof=1)/np.sqrt(k)

    return round(float(mean_mse), 2), round(float(se_mse), 2)



def loocv_mse(estimator, X, y):
    # TODO: LeaveOneOut; mean squared error over held-out points, rounded to 2 decimals
    loo = LeaveOneOut()

    neg_mse_scores = cross_val_score(estimator, X, y, cv = loo, scoring = 'neg_mean_squared_error')


    mse_scores = -neg_mse_scores
    mean_mse = np.mean(mse_scores)

    return round(float(mean_mse), 2)

# Step 5 - cv_spread_by_k
import numpy as np

def cv_spread_by_k(estimator, X, y, ks, seeds):
    # TODO: {k: (mean of cv estimate over seeds, std ddof=1 over seeds)} rounded to 1 decimal
    result = {}
    for k in ks:
        seed_estimates = []
        for seed in seeds:
            mean_mse, _ = cv_mse(estimator, X, y, k=k, random_state = seed)
            seed_estimates.append(mean_mse)

        mean_val = round(float(np.mean(seed_estimates)), 1)

        std_val = round(float(np.std(seed_estimates, ddof=1)))


        result[k] = (mean_val, std_val)


    return result




def compare_with_loocv(estimator, X, y, ks, seeds):
    # TODO: {'loocv': loocv_mse(...), 'kfold': cv_spread_by_k(...)}
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator,X,y,ks,seeds)
    }

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

