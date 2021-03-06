from __future__ import absolute_import
import unittest
import doctest
import sys

def all_tests_suite():
    def get_suite():
        return additional_tests(
            unittest.TestLoader().loadTestsFromNames([
                'ewaim.tests.test_ewaim'
            ]))
    suite = get_suite()
    import calcalc
    return suite

def main():
    runner = unittest.TextTestRunner(verbosity = 1 + sys.argv.count('-v'))
    suite = all_tests_suite()
    raise SystemExit(not runner.run(suite).wasSuccessful())

if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
