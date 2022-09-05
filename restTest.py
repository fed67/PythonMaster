import unittest

import libmyrustlib

class MyTestCase(unittest.TestCase):
    def test_something(self):
        r = libmyrustlib.count_doubles(2,3)

        self.assertEqual(r,5)  # add assertion here


if __name__ == '__main__':
    unittest.main()
