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

    def __getitem__(self, item):
        return self.histogram[item]

    def __setitem__(self, key, value):
        self.histogram[key] = value

    def __len__(self):
        return len(self.histogram)


class Hist1d(MulitHistBase):

    @classmethod
    def from_histogram(cls, bin_edges, histogram):
        """Make a Hist1D from a numpy bin_edges + histogram pair
        :param bin_edges: Bin edges of histogram. Must be one longer than length of histogram
        :param histogram: Initial histogram
        :return:
        """
        if len(bin_edges) != len(histogram) + 1:
            raise ValueError("Bin edges must be of length %d, you gave %d!" % (len(histogram) + 1, len(bin_edges)))
        self = cls(bins=bin_edges)
        self.histogram = histogram

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


class Hist2d(MulitHistBase):
    """
    2D histogram object
    """

    def __init__(self, histogram=None, bin_edges_x=None, bin_edges_y=None, **kwargs):
        self.bin_edges_x = bin_edges_x
        self.bin_edges_y = bin_edges_y
        self.histogram = histogram
        self.kwargs = kwargs
        # np.histogram2d has x and y backwards, see its documentation
        # We workaround this by feeding x and y in reverse.
        # Hence we must also flip a user-specified bins and range argument
        for argname in ('range', 'bins'):
            if argname in kwargs:
                self.kwargs[argname] = list(reversed(self.kwargs[argname]))

    def add(self, x, y, weights=None):
        if self.histogram is None:
            # First time we run!
            self.histogram, self.bin_edges_y, self.bin_edges_x = np.histogram2d(y, x,
                                                                                weights=weights,
                                                                                **self.kwargs)
        else:
            # Pass previous hist's bins instead of (a possibly present) user-defined bins
            hist, _, _  = np.histogram2d(y, x,
                                         bins=[self.bin_edges_y, self.bin_edges_x],
                                         weights=weights,
                                         **{k:v for k, v in self.kwargs.items() if k != 'bins'})
            self.histogram += hist

    def projection(self, axis='x'):
        return Hist1d(
            bin_edges=(self.bin_edges_x if axis=='x' else self.bin_edges_y),
            histogram=np.sum(self.histogram, axis=(0 if axis=='x' else 1))
        )

    def average(self, axis='x'):
        other_axis = 'x' if axis == 'y' else 'y'
        bin_centers_other_axis = self.projection(other_axis).bin_centers
        bin_centers = self.projection(axis).bin_centers
        hist = self.histogram if axis == 'y' else self.histogram.T
        return bin_centers, np.array([
            np.average(bin_centers_other_axis, weights=column) if np.sum(column) != 0 else float('nan')
            for column in hist])

    def slice(self, start, stop=None, axis='x'):
        if stop is None:
            stop = start
        bin_edges = (self.bin_edges_x if axis=='x' else self.bin_edges_y)
        start_bin = np.digitize([start], bin_edges)[0]
        stop_bin =  np.digitize([stop], bin_edges)[0]
        if not (1 <= start_bin <= len(bin_edges)-1 and 1 <= stop_bin <= len(bin_edges)-1):
            raise ValueError("Slice start/stop values are not in range of histogram")
        if axis == 'x':
            hist = self.histogram.T
        else:
            hist = self.histogram
        return Hist1d(
            # bin_edges of the other axis
            bin_edges=(self.bin_edges_y if axis=='x' else self.bin_edges_x),
            histogram=np.sum(hist[start_bin-1:stop_bin], axis=0)
        )

    def plot(self, **kwargs):
        plt.pcolormesh(self.bin_edges_x, self.bin_edges_y, self.histogram, **kwargs)
        plt.xlim(np.min(self.bin_edges_x), np.max(self.bin_edges_x))
        plt.ylim(np.min(self.bin_edges_y), np.max(self.bin_edges_y))
        plt.colorbar()


if __name__ == '__main__':
    # Be careful, if you don't give a range, it is auto-determined by the first data you put in!
    m = Hist1d(bins=100, range=(-3, 4))
    m.add(np.random.normal(0, 0.5, 10**6))
    m.add(np.random.normal(2, 0.2, 10**6))
    m.plot()
    plt.show()

    # Be careful, if you don't give a range, it is auto-determined by the first data you put in!
    m2 = Hist2d(bins=100, range=[[-5,3],[-3,5]])
    m2.add(np.random.normal(1, 1, 10**6), np.random.normal(1, 1, 10**6))
    m2.add(np.random.normal(-2, 1, 10**6), np.random.normal(2, 1, 10**6))
    m2.plot()
    plt.show()