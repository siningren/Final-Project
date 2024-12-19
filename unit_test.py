import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engineering import scaler_transform

class TestScalerTransform(unittest.TestCase):
    def test_scaler_transform(self):
        # Create a sample DataFrame
        data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)

        # The scaler_transform function from feature_engineering 
        sc_df, scaler = scaler_transform(df)

        # Check the output DF's shape
        self.assertEqual(sc_df.shape, df.shape)

        # Check the mean of each column in the scaled DataFrame is approximately 0
        np.testing.assert_almost_equal(sc_df.mean().values, np.zeros(df.shape[1]), decimal=5)

        # Check the standard deviation of each column matches StandardScaler's calculation
        actual_std = sc_df.std(ddof=0).values  
        np.testing.assert_almost_equal(actual_std, np.ones(df.shape[1]), decimal=5)

        # Check the scaler is an instance of StandardScaler
        self.assertIsInstance(scaler, StandardScaler)

if __name__ == "__main__":
    unittest.main()
