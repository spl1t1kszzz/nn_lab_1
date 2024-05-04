import unittest
import pandas as pd
from main import intrinstic_information


class IntrinsticInfTests(unittest.TestCase):
    def testWeatherII(self):
        true = 0.93
        actual = intrinstic_information(pd.read_csv('../Walk.csv'), 'Weather')
        self.assertAlmostEqual(true, actual, 2)

    def testBrandII(self):
        true = 2.32
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = intrinstic_information(laptops, 'Brand')
        self.assertAlmostEqual(true, actual, 2)

    def testStorageCapacityII(self):
        true = 1.582
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = intrinstic_information(laptops, 'Storage_Capacity')
        self.assertAlmostEqual(true, actual, 2)


if __name__ == '__main__':
    unittest.main()
