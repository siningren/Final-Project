import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This function converts the 'Date' column to datetime
def convert_datetime(data):
    data1 = data.copy()
    data1['Date'] = pd.to_datetime(data1['Date'])
    # Set the date column as the index
    data1.set_index('Date', inplace=True)
    return data1

# This function makes a boxplot and removes the outliers
def remove_filers_with_boxplot(data):
    plt.figure(figsize=(12,5),dpi=300)
    p = data.boxplot(return_type='dict')
    plt.xticks(rotation=90,fontsize=10)
    plt.title("outlier data analysis")
    for index,value in enumerate(data.columns):
        try:
            fliers_value_list = p['fliers'][index].get_ydata()
        except:pass
        for flier in fliers_value_list:
            data = data[data.loc[:,value] != flier]
    return data

