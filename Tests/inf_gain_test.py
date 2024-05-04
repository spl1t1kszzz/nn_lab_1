import unittest
import pandas as pd
from main import information_gain


class InfGainTests(unittest.TestCase):

    def testStorageCapacityIG(self):
        true = 1.337
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = information_gain(laptops, 'Storage_Capacity', 'Price')
        self.assertAlmostEqual(true, actual, places=2)

    def testBrandIG(self):
        true = 0.009
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = information_gain(laptops, 'Brand', 'Price')
        self.assertAlmostEqual(true, actual, places=2)


if __name__ == '__main__':
    unittest.main()
