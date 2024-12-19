import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10))
    return (rmse)

# This function performs Lasso feature selection and visualizes the coefficients of 3 largest and smallest variables
def lasso_features_selection(data, X_train, y):
    model_lasso = LassoCV(alphas=[0.1, 1, 0.001, 0.0005]).fit(X_train, y)
    # Print the optimal alpha
    print(model_lasso.alpha_)
    # Print how many variables are picked and how many are eliminated
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    # Print the mean RMSE
    print(rmse_cv(model_lasso, X_train, y).mean())
    # Plot the coefficients of the 3 smallest and largest variables
    imp_coef = pd.concat([coef.sort_values().head(3),
                          coef.sort_values().tail(3)])
    plt.rcParams['figure.figsize'] = (12, 5)
    imp_coef.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.gcf().set_dpi(300)
    plt.show()
    return model_lasso.coef_

