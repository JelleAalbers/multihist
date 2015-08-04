from __future__ import division
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


class MulitHistBase(object):

    def similar_blank_hist(self):
        newhist = deepcopy(self)
        newhist.histogram = np.zeros_like(self.histogram)
        return newhist

    @property
    def n(self):
        """Returns number of data points loaded into histogram"""
        return np.sum(self.histogram)

    @property
    def normed_histogram(self):
        return self.histogram.astype(np.float)/self.n

    # Overload binary numeric operators to work on histogram
    # TODO: logical operators

    def __getitem__(self, item):
        return self.histogram[item]

    def __setitem__(self, key, value):
        self.histogram[key] = value

    def __len__(self):
        return len(self.histogram)

    def __add__(self, other):
        return self.__class__.from_histogram(self.histogram.__add__(other), *self.bin_edges_list)

    def __sub__(self, other):
        return self.__class__.from_histogram(self.histogram.__sub__(other), *self.bin_edges_list)

    def __mul__(self, other):
        return self.__class__.from_histogram(self.histogram.__mul__(other), *self.bin_edges_list)

    def __truediv__(self, other):
        return self.__class__.from_histogram(self.histogram.__truediv__(other), *self.bin_edges_list)

    def __floordiv__(self, other):
        return self.__class__.from_histogram(self.histogram.__floordiv__(other), *self.bin_edges_list)

    def __mod__(self, other):
        return self.__class__.from_histogram(self.histogram.__mod__(other), *self.bin_edges_list)

    def __divmod__(self, other):
        return self.__class__.from_histogram(self.histogram.__divmod__(other), *self.bin_edges_list)

    def __pow__(self, other):
        return self.__class__.from_histogram(self.histogram.__pow__(other), *self.bin_edges_list)

    def __lshift__(self, other):
        return self.__class__.from_histogram(self.histogram.__lshift__(other), *self.bin_edges_list)

    def __rshift__(self, other):
        return self.__class__.from_histogram(self.histogram.__rshift__(other), *self.bin_edges_list)

    def __and__(self, other):
        return self.__class__.from_histogram(self.histogram.__and__(other), *self.bin_edges_list)

    def __xor__(self, other):
        return self.__class__.from_histogram(self.histogram.__xor__(other), *self.bin_edges_list)

    def __or__(self, other):
        return self.__class__.from_histogram(self.histogram.__or__(other), *self.bin_edges_list)

    def __neg__(self):
        return self.__class__.from_histogram(-self.histogram, *self.bin_edges_list)

    def __pos__(self):
        return self.__class__.from_histogram(+self.histogram, *self.bin_edges_list)

    def __abs__(self):
        return self.__class__.from_histogram(abs(self.histogram), *self.bin_edges_list)

    def __invert__(self):
        return self.__class__.from_histogram(~self.histogram, *self.bin_edges_list)


class Hist1d(MulitHistBase):

    @classmethod
    def from_histogram(cls, histogram, bin_edges):
        """Make a Hist1D from a numpy bin_edges + histogram pair
        :param histogram: Initial histogram
        :param bin_edges: Bin edges of histogram. Must be one longer than length of histogram
        :return:
        """
        if len(bin_edges) != len(histogram) + 1:
            raise ValueError("Bin edges must be of length %d, you gave %d!" % (len(histogram) + 1, len(bin_edges)))
        self = cls(bins=bin_edges)
        self.histogram = np.array(histogram)
        return self

    def __init__(self, data=None, bins=10, range=None, weights=None):
        """
        :param data: Initial data to histogram.
        :param bins: Number of bins, or list of bin edges (like np.histogram)
        :param weights: Weights for initial data.
        :param range: Range of histogram.
        :return: None
        """
        if data is None:
            data = []
        self.histogram, self.bin_edges = np.histogram(data, bins=bins, range=range, weights=weights)

    def add(self, data, weights=None):
        hist, _ = np.histogram(data, self.bin_edges, weights=weights)
        self.histogram += hist

    @property
    def bin_centers(self):
        return 0.5*(self.bin_edges[1:] + self.bin_edges[:-1])

    @property
    def cumulative_histogram(self):
        return np.cumsum(self.histogram)

    @property
    def normed_cumulative_histogram(self):
        return np.cumsum(self.normed_histogram)

    def items(self):
        """Iterate over (bin_center, hist_value) from left to right"""
        return zip(self.bin_centers, self.histogram)

    @property
    def mean(self):
        """Estimates mean of underlying data, assuming each datapoint was exactly in the center of a bin"""
        return np.average(self.bin_centers, weights=self.histogram)

    @property
    def std(self):
        """Estimates std of underlying data, assuming each datapoint was exactly in the center of a bin"""
        return np.sqrt(np.average((self.bin_centers-self.mean)**2, weights=self.histogram))

    def plot(self, normed=False, scale_errors_by=1.0, scale_histogram_by=1.0, plt=plt, **kwargs):
        """Plots the histogram with Poisson (sqrt(n)) error bars
          - scale_errors_by multiplies the error bars by its argument
          - scale_histogram_by multiplies the histogram AND the error bars by its argument
          - plt thing to call .errorbar on (pylab, figure, axes, whatever the matplotlib guys come up with next)
        """
        kwargs.setdefault('linestyle', 'none')
        yerr = np.sqrt(self.histogram)
        if normed:
            y = self.normed_histogram
            yerr /= self.n
        else:
            y = self.histogram.astype(np.float)
        yerr *= scale_errors_by * scale_histogram_by
        y *= scale_histogram_by
        plt.errorbar(
            self.bin_centers,
            y,
            yerr,
            marker='.',
            **kwargs
        )

    @property
    def bin_edges_list(self):
        return [self.bin_edges]


class Hist2d(MulitHistBase):
    """
    2D histogram object

    """

    @classmethod
    def from_histogram(cls, histogram, bin_edges_x, bin_edges_y):
        """Make a Hist2D from numpy histogram + bin edges x + bin edges y
        :param histogram: Initial histogram
        :param bin_edges_x: x bin edges of histogram.
        :param bin_edges_y: y bin edges of histogram.
        :return:
        """
        self = cls(bins=[bin_edges_x, bin_edges_y])
        self.bin_edges_x = bin_edges_x
        self.bin_edges_y = bin_edges_y
        self.histogram = histogram

    def __init__(self, x=None, y=None, bins=10, range=None, weights=None):
        if x is None:
            x = []
        if y is None:
            y = []
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        self.histogram, self.bin_edges_x, self.bin_edges_y = np.histogram2d(y, x,
                                                                            bins=bins,
                                                                            weights=weights,
                                                                            range=range)

    def add(self, x, y, weights=None):
        hist, _, _ = np.histogram2d(x, y,
                                    bins=[self.bin_edges_x, self.bin_edges_y],
                                    weights=weights)
        self.histogram += hist

    @property
    def bin_centers_x(self):
        return 0.5*(self.bin_edges_x[1:] + self.bin_edges_x[:-1])

    @property
    def bin_centers_y(self):
        return 0.5*(self.bin_edges_y[1:] + self.bin_edges_y[:-1])

    def projection(self, axis='x'):
        """Sums all data along opposing axis, then return Hist1D
        """
        if axis == 'x':
            return self.slice(self.bin_centers_y[0], self.bin_centers_y[-1], axis='y')
        else:
            return self.slice(self.bin_centers_x[0], self.bin_centers_x[-1], axis='x')

    def average(self, axis='x'):
        """Estimate data mean along axis, then return Hist1d
        """
        bin_centers = self.bin_centers_x if axis == 'x' else self.bin_edges_y
        bin_centers_other_axis = self.bin_centers_x if axis == 'y' else self.bin_edges_y
        hist = self.histogram if axis == 'y' else self.histogram.T
        return Hist1d.from_histogram(histogram=np.array([np.average(bin_centers_other_axis, weights=column)
                                                         if np.sum(column) != 0 else float('nan')
                                                         for column in hist]),
                                     bin_edges=bin_centers)

    def slice(self, start, stop=None, axis='x'):
        """Sums all bins data along axis between start and stop (both inclusive), then return Hist1d
        If stop is not given, take a 1-bin slice over the axis.
        """
        if stop is None:
            stop = start
        bin_edges = self.bin_edges_x if axis == 'x' else self.bin_edges_y
        bin_edges_other_axis = self.bin_edges_y if axis == 'x' else self.bin_edges_x
        start_bin = np.digitize([start], bin_edges)[0]
        stop_bin = np.digitize([stop], bin_edges)[0]
        if not (1 <= start_bin <= len(bin_edges)-1 and 1 <= stop_bin <= len(bin_edges)-1):
            raise ValueError("Slice start/stop values are not in range of histogram")
        if axis == 'x':
            hist = self.histogram
        else:
            hist = self.histogram.T
        return Hist1d.from_histogram(histogram=np.sum(hist[start_bin - 1:stop_bin], axis=0),
                                     bin_edges=bin_edges_other_axis)

    def plot(self, **kwargs):
        plt.pcolormesh(self.bin_edges_x, self.bin_edges_y, self.histogram.T, **kwargs)
        plt.xlim(np.min(self.bin_edges_x), np.max(self.bin_edges_x))
        plt.ylim(np.min(self.bin_edges_y), np.max(self.bin_edges_y))
        plt.colorbar()

    @property
    def bin_edges_list(self):
        return [self.bin_edges_x, self.bin_edges_y]


if __name__ == '__main__':
    # Create a 1d histogram and add some data
    m = Hist1d(bins=100, range=(-3, 4))
    m.add(np.random.normal(0, 0.5, 10**4))
    m.add(np.random.normal(2, 0.2, 10**3))

    # Get the data back out:
    print(m.histogram, m.bin_edges)

    # For plotting you might prefer bin_centers:
    plt.plot(m.bin_centers, m.histogram)
    plt.show()

    # Or use a sensible canned plotting function
    m.plot()
    plt.show()

    # You can also create histograms immediately
    m_instant = Hist1d([0, 3, 1, 6, 2, 9], bins=3)
    m_instant.plot()
    plt.show()

    # Create and show a 2d histogram
    m2 = Hist2d(bins=100, range=[[-5, 3], [-3, 5]])
    m2.add(np.random.normal(1, 1, 10**6), np.random.normal(1, 1, 10**6))
    m2.add(np.random.normal(-2, 1, 10**6), np.random.normal(2, 1, 10**6))
    m2.plot()
    plt.show()

    # Show the x and y projections
    m2.projection('x').plot(label='x projection')
    m2.projection('y').plot(label='y projection', linestyle=':')
    plt.legend()
    plt.show()
