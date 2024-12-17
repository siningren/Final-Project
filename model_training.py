from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
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
def model_metrics(true, pred, title):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    return rmse, r2, mape


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

# Model evaluation (This is just to check the model performance before hyperparameter tuning and check whether the pipelines can be used)   
glm_rmse, glm_r2, glm_mape = model_metrics(y_test, glm_pred, title="GLM Model Curve Fit")
lgbm_rmse, lgbm_r2, lgbm_mape = model_metrics(y_test, lgbm_pred, title="LGBM Model Curve Fit")
    
print("GLM RMSE: ", glm_rmse, 'R2 Score: ', glm_r2, 'Mape: ', glm_mape)
print("LGBM RMSE: ", lgbm_rmse, 'R2 Score: ', lgbm_r2, 'Mape: ', lgbm_mape)




