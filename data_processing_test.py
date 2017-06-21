import data_processing
import unittest
from unittest import TestCase

class OptionLoading(unittest.TestCase):
    def test_load_option(self):
        path = "data/bigdf.pkl"
        df = data_processing.load_options(path)
        TestCase.assertEqual(self, df.shape, (32789,9))

    def test_train_test_split(self):
        path = "data/bigdf.pkl"
        df = data_processing.load_options(path)

        count = 0
        for data in  data_processing.train_test_split(big_df=df, window=1):
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            count+=1

        TestCase.assertEqual(self, count, 2)

if __name__ == "__main__":
    data_processing.to_sample_options()
    unittest.main()