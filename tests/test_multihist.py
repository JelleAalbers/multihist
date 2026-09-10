from __future__ import division
import unittest
import warnings
from unittest import TestCase

import numpy as np

from multihist import Hist1d, Histdd, CAN_PLOT, CoordinateOutOfRangeException

n_bins = 100
test_range = (-3, 4)


def without_warnings(fn, *args, **kwargs):
    """Call fn, asserting it emits no FutureWarning, and return its result"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    unexpected = [str(w.message) for w in caught
                  if issubclass(w.category, FutureWarning)]
    if unexpected:
        raise AssertionError("Unexpected FutureWarning(s): %s" % unexpected)
    return result


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

    def test_reflected_operators(self):
        # Regression test for non-commutative operator bug
        m = Hist1d.from_histogram([1, 2, 3], [0, 1, 2, 3])

        # Reflected operators must not be confused with their forward version
        self.assertEqual((0 - m).histogram.tolist(), (-m).histogram.tolist())
        self.assertEqual((0 - m).histogram.tolist(), [-1, -2, -3])
        self.assertEqual((5 - m).histogram.tolist(), [4, 3, 2])
        self.assertEqual((m - 5).histogram.tolist(), [-4, -3, -2])

        self.assertEqual((12 / m).histogram.tolist(), [12, 6, 4])
        self.assertEqual((12 // m).histogram.tolist(), [12, 6, 4])
        self.assertEqual((7 % m).histogram.tolist(), [0, 1, 1])
        self.assertEqual((2 ** m).histogram.tolist(), [2, 4, 8])
        self.assertEqual((m ** 2).histogram.tolist(), [1, 4, 9])
        self.assertEqual((16 >> m).histogram.tolist(), [8, 4, 2])
        self.assertEqual((1 << m).histogram.tolist(), [2, 4, 8])

        # Bin edges must survive the reflected operators
        self.assertEqual((5 - m).bin_edges.tolist(), m.bin_edges.tolist())

        # Commutative operators keep working
        self.assertEqual((5 + m).histogram.tolist(), (m + 5).histogram.tolist())
        self.assertEqual((5 * m).histogram.tolist(), (m * 5).histogram.tolist())

    def test_std(self):
        m = Hist1d.from_histogram([1, 2, 1], [0, 1, 2, 3])

        # The histogram describes this data (each point at its bin center):
        data = [0.5, 1.5, 1.5, 2.5]
        self.assertEqual(m.n, len(data))
        self.assertAlmostEqual(m.mean, np.mean(data))

        # Hist1d.std applies the Bessel correction, so it estimates the std of
        # the underlying data: that means ddof=1, not ddof=0.
        self.assertAlmostEqual(m.std, np.std(data, ddof=1))

        # Explicitly: the correction is sqrt(n/(n-1)) on the std,
        # i.e. n/(n-1) on the variance
        n = len(data)
        self.assertAlmostEqual(m.std, np.std(data) * np.sqrt(n / (n - 1)))

    def test_get_axis_bin_index(self):
        m = Hist1d.from_histogram([1, 2, 3], [0, 1, 2, 3])

        self.assertEqual(m.get_axis_bin_index(0), 0)
        self.assertEqual(m.get_axis_bin_index(0.5), 0)
        self.assertEqual(m.get_axis_bin_index(1), 1)
        self.assertEqual(m.get_axis_bin_index(2.5), 2)

        # The rightmost bin edge is inclusive, like in np.histogram
        self.assertEqual(m.get_axis_bin_index(3), 2)

        # Below the range: raises, as it always has
        for value in (-0.5, -99):
            self.assertRaises(CoordinateOutOfRangeException,
                              m.get_axis_bin_index, value)

        # Above the range: still returns an index that is one too high to be a
        # bin, but now warns that this will become an exception
        for value in (3.5, 99):
            with self.assertWarns(FutureWarning):
                index = m.get_axis_bin_index(value)
            self.assertEqual(index, len(m.histogram))

        # Whatever comes back must be a usable index into the histogram
        for value in np.linspace(0, 3, 25):
            index = m.get_axis_bin_index(value)
            self.assertTrue(0 <= index < len(m.histogram),
                            "Bin index %s out of range for %s bins" % (index, len(m.histogram)))

    def test_slice_out_of_range(self):
        m = Hist1d.from_histogram([10, 20, 30, 40, 50], [0, 1, 2, 3, 4, 5])

        # Slicing is lenient above the range: a stop beyond the last bin edge
        # slices up to the end, and a start beyond it gives an empty histogram.
        # This is deliberate, so it must not warn.
        self.assertEqual(without_warnings(m.slice, 0, 99).histogram.tolist(),
                         [10, 20, 30, 40, 50])
        self.assertEqual(without_warnings(m.slice, 2.5, 99).histogram.tolist(),
                         [30, 40, 50])
        self.assertEqual(without_warnings(m.slice, 99, 100).histogram.tolist(), [])

        # Below the range it raises, as it always has
        self.assertRaises(CoordinateOutOfRangeException, m.slice, -1, 2)


test_range_2d = ((-1, 1), (-10, 10))
test_bins_2d = 3


class TestHistdd(TestCase):

    def setUp(self):
        self.m = Histdd(range=test_range_2d, bins=test_bins_2d, axis_names=['foo', 'bar'])

    def test_is_instance(self):
        self.assertIsInstance(self.m, Histdd)

    def test_get_bin_indices(self):
        m = Histdd.from_histogram(np.arange(6).reshape(2, 3),
                                  bin_edges=[np.array([0, 1, 2]),
                                             np.array([0, 1, 2, 3])],
                                  axis_names=['x', 'y'])

        self.assertEqual(m.get_bin_indices((0.5, 0.5)), (0, 0))
        self.assertEqual(m.get_bin_indices((1.5, 2.5)), (1, 2))

        # Rightmost bin edges are inclusive along every axis
        self.assertEqual(m.get_bin_indices((2, 3)), (1, 2))

        # The indices must actually index the histogram
        self.assertEqual(m.histogram[m.get_bin_indices((1.5, 2.5))], 5)

        # Below the range along either axis: raises
        for values in [(-1, 1.5), (1.5, -1)]:
            self.assertRaises(CoordinateOutOfRangeException,
                              m.get_bin_indices, values)

        # Above the range along either axis: warns, and still returns the
        # index that blows up with an IndexError later
        for values, bad in [((2.5, 1.5), (2, 1)), ((1.5, 99), (1, 3))]:
            with self.assertWarns(FutureWarning):
                self.assertEqual(m.get_bin_indices(values), bad)
        with self.assertWarns(FutureWarning):
            m.get_axis_bin_index(99, 'y')

    def test_slice_out_of_range(self):
        m = Histdd.from_histogram(np.arange(20).reshape(4, 5),
                                  bin_edges=[np.array([0, 1, 2, 3, 4]),
                                             np.array([0, 1, 2, 3, 4, 5])],
                                  axis_names=['x', 'y'])

        # As for Hist1d: lenient above the range without warning, raises below it
        self.assertEqual(without_warnings(m.slice, 0, 99, axis=0).histogram.tolist(),
                         m.histogram.tolist())
        self.assertEqual(without_warnings(m.slice, 2, 99, axis=0).histogram.tolist(),
                         m.histogram[2:].tolist())
        self.assertEqual(without_warnings(m.slice, 99, 100, axis=0).histogram.shape,
                         (0, 5))
        self.assertRaises(CoordinateOutOfRangeException, m.slice, -1, 2, 0)

    def test_reflected_operators(self):
        m = self.m
        m.add([0, 0, 0], [0, 0, 5])

        for h in (5 - m, m.__rsub__(5)):
            self.assertIsInstance(h, Histdd)
            self.assertEqual(h.histogram.tolist(),
                             (5 - m.histogram).tolist())
            self.assertEqual(h.axis_names, m.axis_names)
        self.assertEqual((m - 5).histogram.tolist(),
                         (m.histogram - 5).tolist())
        self.assertEqual((0 - m).histogram.tolist(),
                         (-m).histogram.tolist())
        self.assertEqual((2 ** m).histogram.tolist(),
                         (2 ** m.histogram).tolist())

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

    @unittest.skipIf(not CAN_PLOT, "cannot test plotting if matplotlib is not installed")
    def test_lognorm(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        # Create and show a 2d histogram. Axis names are optional.
        m2 = Histdd(bins=10, range=[[-5, 3], [-3, 5]], axis_names=['x', 'y'])
        m2.add(np.random.normal(1, 1, 10**3), np.random.normal(1, 1, 10**3))
        m2.add(np.random.normal(-2, 1, 10**3), np.random.normal(2, 1, 10**3))
        m2.plot(vmin=1, vmax=10, log_scale=True,)
        plt.clf()
        m2.plot(norm=LogNorm(vmin=1, vmax=10))  # correct matplotlib3.5 style
        plt.clf()
        m2.plot(log_scale=True,)
        plt.clf()


if __name__ == '__main__':
    unittest.main()
