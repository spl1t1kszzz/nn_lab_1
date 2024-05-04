import unittest
import pandas as pd
from main import target_entropy


class TargetEntropyTests(unittest.TestCase):
    def testGo(self):
        true = 0.88
        actual = target_entropy(pd.read_csv('../Walk.csv'), 'Go')
        self.assertAlmostEqual(true, actual, 2)

    def testPrice(self):
        true = 9.9657
        actual = target_entropy(pd.read_csv('../Laptop_price.csv'), 'Price')
        self.assertAlmostEqual(true, actual, 2)

    def testBrand(self):
        true = 2.32
        actual = target_entropy(pd.read_csv('../Laptop_price.csv'), 'Brand')
        self.assertAlmostEqual(true, actual, 2)


if __name__ == '__main__':
    unittest.main()
