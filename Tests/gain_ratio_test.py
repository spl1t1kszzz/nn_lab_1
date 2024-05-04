import unittest
import pandas as pd
from main import gain_ratio


class GainRatioTests(unittest.TestCase):
    def testWeatherGR(self):
        true = 0.347823
        actual = gain_ratio(pd.read_csv('../Walk.csv'), 'Weather', 'Go')
        self.assertAlmostEqual(true, actual, places=2)

    def testDayGR(self):
        true = 0.00306
        actual = gain_ratio(pd.read_csv('../Walk.csv'), 'Day', 'Go')
        self.assertAlmostEqual(true, actual, places=2)

    def testStorageCapacityGR(self):
        true = 0.845
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = gain_ratio(laptops, 'Storage_Capacity', 'Price')
        self.assertAlmostEqual(true, actual, places=2)

    def testBrandGR(self):
        true = 0.0041
        laptops = pd.read_csv('../Laptop_price.csv')
        target = 'Price'
        laptops[target] = pd.cut(laptops[target], bins=3)
        actual = gain_ratio(laptops, 'Brand', 'Price')
        self.assertAlmostEqual(true, actual, places=2)


if __name__ == '__main__':
    unittest.main()
