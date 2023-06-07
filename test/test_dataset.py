import unittest
import numpy as np
from dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data1 = {
            "X": np.array([
                [1, 2],
                [3, 4],
                [5, 6],
            ]),
            "y": np.array([7, 8, 9]),
            "ids": np.array(["1", "2", "3"]),
        }
        self.data2 = {
            "X": np.array([
                [7, 8],
                [9, 10],
                [11, 12],
            ]),
            "y": np.array([13, 14, 15]),
            "ids": np.array(["4", "5", "6"]),
        }
        self.desired_data = {
            "X": np.array([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
            ]),
            "y": np.array([7, 8, 9, 13, 14, 15]),
            "ids": np.array(["1", "2", "3", "4", "5", "6"]),
        }

    def test_merge(self):
        dataset1 = Dataset(**self.data1)
        dataset2 = Dataset(**self.data2)
        actual_dataset = Dataset.merge(dataset1, dataset2)
        desired_dataset = Dataset(**self.desired_data)
        np.testing.assert_equal(actual_dataset, desired_dataset)
        return


if __name__ == "__main__":
    unittest.main()
