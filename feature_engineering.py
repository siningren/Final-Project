import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# simple scikit-learn transformer in a feature_engineering
def scaler_transform(df):
    scaler = StandardScaler()
    scaler.fit(df)
    sc_df = scaler.transform(df)
    sc_df = pd.DataFrame(sc_df, columns=df.columns)
    return sc_df, scaler

