from __future__ import division
import unittest
from unittest import TestCase

import numpy as np

from multihist import Hist1d, Histdd

n_bins = 100
test_range = (-3, 4)


class TestHist1d(TestCase):

    def setUp(self):
        self.m = Hist1d(bins=n_bins, range=test_range)

    def test_is_instance(self):
        self.assertIsInstance(self.m, Hist1d)

    def test_list_like_access(self):
        m = self.m
        self.assertEqual(m[3], 0)
        self.assertEqual(m[3:5].tolist(), [0, 0])
        m[3] = 6
        self.assertEqual(m[3], 6)
        m[3:5] = [4, 3]
        self.assertEqual(m[3:5].tolist(), [4, 3])
        self.assertEqual(len(m), 100)

    def test_iteritems(self):
        m = self.m
        it = m.items()
        self.assertEqual(next(it), (-2.965, 0))

    def test_init_from_data(self):
        ex_data = list(range(11))
        m = Hist1d(ex_data, bins=len(ex_data) - 1)
        self.assertEqual(m.bin_edges.tolist(), ex_data)
        self.assertTrue(m.histogram.tolist(), [1]*n_bins)

    def test_init_from_histogram(self):
        m = Hist1d.from_histogram([0, 1, 0], [0, 1, 2, 3])
        self.assertEqual(m.histogram.tolist(), [0, 1, 0])
        self.assertEqual(m.bin_centers.tolist(), [0.5, 1.5, 2.5])

    def test_add_data(self):
        m = self.m
        m.add([0, 3, 4])
        self.assertEqual(m.histogram.tolist(),
                         np.histogram([0, 3, 4],
                                      bins=n_bins, range=test_range)[0].tolist())
        m.add([0, 3, 4])
        self.assertEqual(m.histogram.tolist(),
                         np.histogram([0, 3, 4]*2,
                                      bins=n_bins, range=test_range)[0].tolist())
        m.add([0, 3, 4, 538])
        self.assertEqual(m.histogram.tolist(),
                         np.histogram([0, 3, 4]*3,
                                      bins=n_bins, range=test_range)[0].tolist())

    def test_overload(self):
        m = self.m
        m.add([test_range[0]])
        m2 = self.m + self.m
        self.assertEqual(m2.histogram[0], [2])
        self.assertEqual(m2.histogram[1], [0])
        self.assertEqual(m2.bin_edges.tolist(), m.bin_edges.tolist())

test_range_2d = ((-1, 1), (-10, 10))
test_bins_2d = 3


class TestHistdd(TestCase):

    def setUp(self):
        self.m = Histdd(range=test_range_2d, bins=test_bins_2d, axis_names=['foo', 'bar'])

    def test_is_instance(self):
        self.assertIsInstance(self.m, Histdd)

    def test_add_data(self):
        m = self.m
        x = [0.1, 0.8, -0.4]
        y = [0, 0, 0]
        m.add(x, y)
        self.assertEqual(m.histogram.tolist(),
                         np.histogram2d(x, y,
                                        range=test_range_2d,
                                        bins=test_bins_2d)[0].tolist())
        m.add(x, y)
        self.assertEqual(m.histogram.tolist(),
                         (np.histogram2d(x*2, y*2,
                                         range=test_range_2d,
                                         bins=test_bins_2d)[0].tolist()))
        m.add([999, 999], [111, 111])
        self.assertEqual(m.histogram.tolist(),
                         np.histogram2d(x*2, y*2,
                                        range=test_range_2d,
                                        bins=test_bins_2d)[0].tolist())

    def test_pandas(self):
        import pandas as pd
        m = self.m
        test_data = pd.DataFrame([{'foo': 0, 'bar': 0}, {'foo': 0, 'bar': 5}])
        m.add(test_data)
        self.assertEqual(m.histogram.tolist(),
                         np.histogram2d([0, 0], [0, 5],
                                        range=test_range_2d,
                                        bins=test_bins_2d)[0].tolist())

    def test_projection(self):
        m = self.m
        x = [0.1, 0.8, -0.4]
        y = [0, 0, 0]
        m.add(x, y)
        p1 = m.projection(0)
        self.assertEqual(p1.histogram.tolist(), [1, 1, 1])
        self.assertAlmostEqual(np.sum(p1.bin_edges - np.array([-1, -1/3, 1/3, 1])), 0)
        p2 = m.projection(1)
        self.assertEqual(p2.histogram.tolist(), [0, 3, 0])
        self.assertAlmostEqual(np.sum(p2.bin_edges - np.array([-1, -1/3, 1/3, 1])), 0)
        p_2 = m.projection('bar')
        self.assertEqual(p2.histogram.tolist(), p_2.histogram.tolist())
        self.assertEqual(p2.bin_edges.tolist(), p_2.bin_edges.tolist())

    def test_cumulate(self):
        self.m.add([-1, 0, 1], [-10, 0, 10])
        np.testing.assert_equal(self.m.histogram,
                                np.array([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]]))
        np.testing.assert_equal(self.m.cumulate(0).histogram,
                                np.array([[1, 0, 0],
                                          [1, 1, 0],
                                          [1, 1, 1]]))
        np.testing.assert_equal(self.m.cumulate(1).histogram,
                                np.array([[1, 1, 1],
                                          [0, 1, 1],
                                          [0, 0, 1]]))
        np.testing.assert_equal(self.m.cumulate(1).histogram,
                                self.m.cumulative_density(1).histogram)
        self.m.add([-1, 0, 1], [-10, 0, 10])
        np.testing.assert_equal(self.m.cumulate(1).histogram,
                                2 * self.m.cumulative_density(1).histogram)

if __name__ == '__main__':
    unittest.main()
