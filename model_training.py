from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This part is setting up my model pipelines for both GLM and LGBM
# GLM model pipeline
def glm_pipeline():
    glm_model = Pipeline(steps=[
        ('scaler', StandardScaler()), 
        ('glm', ElasticNet(alpha=0.5, l1_ratio=0.01))  
    ])
    return glm_model

# LGBM model pipeline
def lgbm_pipeline():
    lgbm_model = Pipeline(steps=[
        ('scaler', StandardScaler()),  
        ('lgbm', LGBMRegressor(n_estimators=20))  
    ])
    return lgbm_model



# Split sample function
def split_data(X, y, random_state=202412):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test

#  Model evaluation function
def model_metrics(true, pred, title, if_show):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    mae = mean_absolute_error(true, pred)
    if if_show:
        plt.figure(figsize=(12, 4), dpi=300)
        plt.plot(range(len(pred)), pred, label="pred", c='green')
        plt.scatter(range(len(pred)), true, label="true", c='red')
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.savefig(f"save/{title}.png")
        #plt.show()
    return rmse, r2, mape, mae

# k-fold cross validation function
def grid_search_model(params, model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, params, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    # The detailed info of best model, best params and cv results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = pd.DataFrame(grid_search.cv_results_)
    return best_model, best_params, cv_results


# Load clean data
clean_data = pd.read_parquet("data/data.parquet")
    
# Split data into train and test sets
X, y = clean_data.drop(['Adj Close'], axis=1), clean_data['Adj Close']
x_train, x_test, y_train, y_test = split_data(X, y)

# GLM Model before hyperparameter tuning
glm_model = glm_pipeline()
glm_model.fit(x_train, y_train)
glm_pred = glm_model.predict(x_test)
    
# LGBM Model before hyperparameter tuning
lgbm_model = lgbm_pipeline()
lgbm_model.fit(x_train, y_train)
lgbm_pred = lgbm_model.predict(x_test)

# Model evaluation before hyperparameter tuning 
glm_rmse, glm_r2, glm_mape, glm_mae = model_metrics(y_test, glm_pred, title="GLM Model Curve Fit", if_show= False)
lgbm_rmse, lgbm_r2, lgbm_mape, lgbm_mae = model_metrics(y_test, lgbm_pred, title="LGBM Model Curve Fit", if_show= False)


# Tune the model pipelines
# GLM model tuning: alpha and l1_ratio
glm_params = {
    'alpha': [0.001, 0.01, 0.1, 0.005, 1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
base_glm = ElasticNet()
glm_best_model, glm_best_params, glm_cv_results = grid_search_model(glm_params, base_glm, X, y)
print("GridSearchCV GLM best params: ", glm_best_params, 'cv results: ', np.mean(glm_cv_results['std_test_score']))

# LGBM model tuning: learning_rate, n_estimators, n_leaves, min_child_weight
lgbm_param_grid = {
    'learning_rate': [0.0001, 0.01, 0.001, 0.1], 
    'n_estimators': [100, 150, 200, 500, 650, 800],
    'num_leaves': [3, 5, 12],  
    'min_child_weight': [2, 5, 10]  
}
base_lgbm = LGBMRegressor()
lgbm_best_model, lgbm_best_params, lgbm_cv_results = grid_search_model(lgbm_param_grid, base_lgbm, X, y)
print("GridSearchCV LGBM best params: ", lgbm_best_params, 'cv results: ', np.mean(lgbm_cv_results['std_test_score']))

# Evaluate the predictions of tuned GLM and LGBM pipelines on your validation set
# Create the "Predicted vs Actual" plot for tuned GLM model and LGBM model
# From the printed results, it can be seen that the tuned GLM has lower RMSE, higher R2 score, lower Mape, and lower MAE than the original one.
# This proves that the tuned GLM model is better than the original one
glm_best_model.fit(x_train, y_train)
tuned_glm_pred = glm_best_model.predict(x_test)
tuned_glm_rmse, tuned_glm_r2, tuned_glm_mape, tuned_glm_mae = model_metrics(y_test, tuned_glm_pred, title="Predicted vs Actual for Tuned GLM Model", if_show = True) 
print("Tuned GLM RMSE: ", tuned_glm_rmse, 'R2 Score: ', tuned_glm_r2, 'Mape: ', tuned_glm_mape, 'MAE:', tuned_glm_mae)
print("GLM RMSE: ", glm_rmse, 'R2 Score: ', glm_r2, 'Mape: ', glm_mape, 'MAE:' , glm_mae)

# From the printed results, it can be seen that the tuned LGBM has lower RMSE, higher R2 score, lower Mape, and lower MAE
lgbm_best_model.fit(x_train, y_train)
tuned_lgbm_pred = lgbm_best_model.predict(x_test)
tuned_lgbm_rmse, tuned_lgbm_r2, tuned_lgbm_mape, tuned_lgbm_mae = model_metrics(y_test, tuned_lgbm_pred, title="Predicted vs Actual for Tuned LGBM Model", if_show = True)
print("Tuned LGBM RMSE: ", tuned_lgbm_rmse, 'R2 Score: ', tuned_lgbm_r2, 'Mape: ', tuned_lgbm_mape, 'MAE:', tuned_lgbm_mae)
print("LGBM RMSE: ", lgbm_rmse, 'R2 Score: ', lgbm_r2, 'Mape: ', lgbm_mape, 'MAE:' , lgbm_mae)

# Get the feature importances of the tuned LGBM and GLM models and create dataframes
imp_lgbm = lgbm_best_model.feature_importances_
imp_glm = np.abs(glm_best_model.coef_)
imp_df = pd.DataFrame([imp_lgbm, imp_glm]).T
imp_df.columns = ['LGBM', 'GLM']
imp_df.index = X.columns

# Get the top 5 important features and plot them 
top5_lgbm = imp_df['LGBM'].sort_values(ascending=False).head(5)
top5_glm = imp_df['GLM'].sort_values(ascending=False).head(5)

plt.figure(figsize=(12, 4), dpi=300)
# Top 5 important features in LGBM model    
plt.subplot(1, 2, 1)
top5_lgbm.plot(kind='bar', color='skyblue')
plt.title('Top 5 Features Importance in LGBM')
plt.xlabel('Features')
plt.ylabel('Importance Score')
# Top 5 important features in GLM model
plt.subplot(1, 2, 2)
top5_glm.plot(kind='bar', color='lightgreen')
plt.title('Top 5 Features Importance in GLM')
plt.xlabel('Features')
plt.ylabel('Importance Score')

plt.tight_layout()
plt.savefig("save/feature_importance_top5.png")
plt.show()
'''
From the plot, it can be seen that the top 5 important features are different in two models

For LGBM, the top 5: 
1. USB_High (US bond rate high data), 
2. USDI_Volume (US Dollar Index volume data), 
3. Volume (Gold ETF volume data), 
4. USO_Volume (Oil ETF volume data), 
5. EG_Volume (Eldorado Gold Corporations volume data)

For GLM, the top 5: 
1. USB_High (US bond rate high data), 
2. SP_Open (S&P 500 Index open data), 
3. SP_Ajclose (S&P 500 Index adjusted close data), 
4. PLD_Trend (Palladium price trend data)
5. PLT_Trend (Platinum price trend data)
'''

# The partial dependence plot of the top 5 important features of tuned LGBM model
