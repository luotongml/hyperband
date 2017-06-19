import data_processing
import unittest

class OptionLoading(unittest.TestCase):
    def test_load_option(self):
        path = "data/bigdf.pkl"
        df = data_processing.load_options(path)
        print(df.shape)

    def test_loading(self):
        path = "data/bigdf.pkl"
        df = data_processing.load_options(path)
        for data in  data_processing.train_test_split(big_df=df, window=2):
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            print(x_train.shape)
            print(y_train.shape)
            print(x_test.shape)
            print(y_test.shape)

if __name__ == "__main__":
    data_processing.to_sample_options()
    unittest.main()