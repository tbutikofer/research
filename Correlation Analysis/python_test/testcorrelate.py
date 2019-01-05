import unittest
import correlate
import numpy as np

class CorrelateTestCase(unittest.TestCase):

    def test_get_first_day(self):
        result = correlate.get_first_day('14.11.2018', 365)
        self.assertTrue(result == '14.11.2017', "result was {}".format(result))

if __name__ == '__main__':
    unittest.main()


