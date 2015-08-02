from unittest import TestCase

from multihist import Hist1d

class TestHist1d(TestCase):

    def setUp(self):
        self.m = Hist1d(bins=100, range=(-3, 4))

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
        m = Hist1d(ex_data, bins=len(ex_data) -1)
        self.assertEqual(m.bin_edges, ex_data)