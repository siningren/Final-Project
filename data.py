#This is a module for data loading

import pandas as pd
def load_data(path):
    data=pd.read_csv(path)
    return data
if __name__ == '__main__':
    data_path="./data/FINAL_USO.csv"
    data=load_data(data_path)
    print(data.head())