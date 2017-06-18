import data_processing
import unittest

class OptionLoading(unittest.TestCase):
    def testLoading(self):
        path = "data/bigdf.pkl"
        data = data_processing.load_option(path)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

if __name__ == "__main__":
    unittest.main()