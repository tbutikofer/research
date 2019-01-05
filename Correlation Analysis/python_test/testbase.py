import unittest
import pandas as pd
import base as base

class BaseTestCase(unittest.TestCase):
    @staticmethod
    def create_test_raw_data():
        df = pd.DataFrame(
            data = {'date': [
                            '2018-10-01','2018-10-01','2018-10-01',
                            '2018-10-02','2018-10-02','2018-10-02',
                            '2018-10-03','2018-10-03','2018-10-03',
                            '2018-10-04','2018-10-04','2018-10-04',
                            '2018-10-05','2018-10-05','2018-10-05'
                            ],
                    'isin': [
                            'isin1','isin2','isin3',
                            'isin1','isin2','isin3',
                            'isin1','isin2','isin3',
                            'isin1','isin2','isin3',
                            'isin1','isin2','isin3'
                            ],
                    'quote': [
                                1.1,1.2,1.3,
                                1.4,1.5,1.6,
                                1.2,1.3,1.4,
                                1.9,1.8,1.7,
                                1.0,1.5,1.7
                                ]
            })
        return df


    def test_get_db_connection(self):
        connection = base.get_db_connection()
        self.assertFalse(connection is None, "DB connection could not be established")

    def test_read_raw_data(self):
        db_connection = base.get_db_connection()
        df = base.read_raw_data(db_connection, 'SWX', 5)
        i = 42

    def test_cleanup_date(self):
        df = BaseTestCase.create_test_raw_data()
        df = base.cleanup_data(df)

    def test_calc_correlation(self):
        df = BaseTestCase.create_test_raw_data()
        df = base.cleanup_data(df)
        df = base.calc_correlation(df)

    def test_get_clean_data(self):
        df = base.get_clean_data('SWX', 30)

if __name__ == '__main__':
    unittest.main()
