from __future__ import division
from copy import deepcopy
try:
    from itertools import izip as zip
except ImportError:
    # Hello, python 3!
    pass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter


class CoordinateOutOfRangeException(Exception):
    pass


class MultiHistBase(object):

    def similar_blank_hist(self):
        newhist = deepcopy(self)
        newhist.histogram = np.zeros_like(self.histogram)
        return newhist

    @property
    def n(self):
        """Returns number of data points loaded into histogram"""
        return np.sum(self.histogram)

    # Overload binary numeric operators to work on histogram
    # TODO: logical operators

    def __getitem__(self, item):
        return self.histogram[item]

    def __setitem__(self, key, value):
        self.histogram[key] = value

    def __len__(self):
        return len(self.histogram)

    def __add__(self, other):
        return self.__class__.from_histogram(self.histogram.__add__(other), self.bin_edges, self.axis_names)

    def __sub__(self, other):
        return self.__class__.from_histogram(self.histogram.__sub__(other), self.bin_edges, self.axis_names)

    def __mul__(self, other):
        return self.__class__.from_histogram(self.histogram.__mul__(other), self.bin_edges, self.axis_names)

    def __truediv__(self, other):
        return self.__class__.from_histogram(self.histogram.__truediv__(other), self.bin_edges, self.axis_names)

    def __floordiv__(self, other):
        return self.__class__.from_histogram(self.histogram.__floordiv__(other), self.bin_edges, self.axis_names)

    def __mod__(self, other):
        return self.__class__.from_histogram(self.histogram.__mod__(other), self.bin_edges, self.axis_names)

    def __divmod__(self, other):
        return self.__class__.from_histogram(self.histogram.__divmod__(other), self.bin_edges, self.axis_names)

    def __pow__(self, other):
        return self.__class__.from_histogram(self.histogram.__pow__(other), self.bin_edges, self.axis_names)

    def __lshift__(self, other):
        return self.__class__.from_histogram(self.histogram.__lshift__(other), self.bin_edges, self.axis_names)

    def __rshift__(self, other):
        return self.__class__.from_histogram(self.histogram.__rshift__(other), self.bin_edges, self.axis_names)

    def __and__(self, other):
        return self.__class__.from_histogram(self.histogram.__and__(other), self.bin_edges, self.axis_names)

    def __xor__(self, other):
        return self.__class__.from_histogram(self.histogram.__xor__(other), self.bin_edges, self.axis_names)

    def __or__(self, other):
        return self.__class__.from_histogram(self.histogram.__or__(other), self.bin_edges, self.axis_names)

    def __neg__(self):
        return self.__class__.from_histogram(-self.histogram, self.bin_edges, self.axis_names)

    def __pos__(self):
        return self.__class__.from_histogram(+self.histogram, self.bin_edges, self.axis_names)

    def __abs__(self):
        return self.__class__.from_histogram(abs(self.histogram), self.bin_edges, self.axis_names)

    def __invert__(self):
        return self.__class__.from_histogram(~self.histogram, self.bin_edges, self.axis_names)

MultiHistBase.similar_blank_histogram = MultiHistBase.similar_blank_hist

class Hist1d(MultiHistBase):
    axis_names = None

    @classmethod
    def from_histogram(cls, histogram, bin_edges, axis_names=None):
        """Make a Hist1D from a numpy bin_edges + histogram pair
        :param histogram: Initial histogram
        :param bin_edges: Bin edges of histogram. Must be one longer than length of histogram
        :param axis_names: Ignored. Sorry :-)
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
    def density(self):
        """Gives emprical PDF, like np.histogram(...., density=True)"""
        h = self.histogram.astype(np.float)
        bindifs = np.array(np.diff(self.bin_edges), float)
        return h/(bindifs * self.n)

    @property
    def normalized_histogram(self):
        """Gives histogram with sum of entries normalized to 1."""
        return self.histogram/self.n

    @property
    def cumulative_histogram(self):
        return np.cumsum(self.histogram)

    @property
    def cumulative_density(self):
        cs = np.cumsum(self.histogram)
        return cs/cs[-1]

    def get_random(self, *args, **kwargs):
        """Returns random variates from the histogram. Only bin centers can be returned."""
        return np.random.choice(self.bin_centers, p=self.normalized_histogram, *args, **kwargs)

    def items(self):
        """Iterate over (bin_center, hist_value) from left to right"""
        return zip(self.bin_centers, self.histogram)

    @property
    def mean(self):
        """Estimates mean of underlying data, assuming each datapoint was exactly in the center of its bin."""
        return np.average(self.bin_centers, weights=self.histogram)

    @property
    def std(self, bessel_correction=True):
        """Estimates std of underlying data, assuming each datapoint was exactly in the center of its bin."""
        if bessel_correction:
            n = self.n
            bc = n/(n-1)
        else:
            bc = 1
        return np.sqrt(np.average((self.bin_centers-self.mean)**2, weights=self.histogram)) * bc

    def plot(self, normed=False, scale_errors_by=1.0, scale_histogram_by=1.0, plt=plt, errors=False, **kwargs):
        """Plots the histogram with Poisson (sqrt(n)) error bars
          - scale_errors_by multiplies the error bars by its argument
          - scale_histogram_by multiplies the histogram AND the error bars by its argument
          - plt thing to call .errorbar on (pylab, figure, axes, whatever the matplotlib guys come up with next)
        """

        if errors:
            kwargs.setdefault('linestyle', 'none')
            yerr = np.sqrt(self.histogram)
            if normed:
                y = self.normed_histogram
                yerr /= self.n
            else:
                y = self.histogram.astype(np.float)
            yerr *= scale_errors_by * scale_histogram_by
            y *= scale_histogram_by
            plt.errorbar(self.bin_centers, y, yerr,
                         marker='.', **kwargs)
        else:
            kwargs.setdefault('linestyle', 'steps-mid')
            plt.plot(self.bin_centers, self.histogram, **kwargs)


class Histdd(MultiHistBase):
    """multidimensional histogram object
    """

    @classmethod
    def from_histogram(cls, histogram, bin_edges, axis_names=None):
        """Make a HistdD from numpy histogram + bin edges
        :param histogram: Initial histogram
        :param bin_edges: x bin edges of histogram, y bin edges, ...
        :return: Histnd instance
        """
        bin_edges = np.array(bin_edges)
        self = cls(bins=bin_edges, axis_names=axis_names)
        self.histogram = histogram
        return self

    def __init__(self, *data, **kwargs):
        for k, v in {'bins': 10, 'range': None, 'weights': None, 'axis_names': None}.items():
            kwargs.setdefault(k, v)
    
        if len(data) == 0:
            if kwargs['range'] is None:
                if kwargs['bins'] is None:
                    raise ValueError("Must specify data, bins, or range")
                try:
                    dimensions = len(kwargs['bins'])
                except TypeError:
                    raise ValueError("If you specify no data and no ranges, must specify a bin specification "
                                     "which tells me what dimension you want. E.g. [10, 10, 10] instead of 10.")
            else:
                dimensions = len(kwargs['range'])
            data = np.zeros((0, dimensions)).T
        self.histogram, self.bin_edges = np.histogramdd(np.array(data).T, 
                                                        bins=kwargs['bins'], 
                                                        weights=kwargs['weights'], 
                                                        range=kwargs['range'])
        self.axis_names = kwargs['axis_names']

    def add(self, *data, **kwargs):
        kwargs.setdefault('weights', None)
        self.histogram += np.histogramdd(np.array(data).T, bins=self.bin_edges, weights=kwargs['weights'])[0]

    @property
    def dimensions(self):
        return len(self.bin_edges)

    ##
    # Axis selection
    ##
    def get_axis_number(self, axis):
        if isinstance(axis, int):
            return axis
        if isinstance(axis, str):
            if self.axis_names is None:
                raise ValueError("Axis name %s not in histogram: histogram has no named axes." % axis)
            if axis in self.axis_names:
                return self.axis_names.index(axis)
            raise ValueError("Axis name %s not in histogram. Axis names which are: %s" % (axis, self.axis_names))
        raise ValueError("Argument to get_axis_number should be string or integer, but you gave %s" % axis)

    def other_axes(self, axis):
        axis = self.get_axis_number(axis)
        return tuple([i for i in range(self.dimensions) if i != axis])

    def axis_names_without(self, axis):
        """Return axis names without axis, or None if axis_names is None"""
        if self.axis_names is None:
            return None
        return itemgetter(*self.other_axes(axis))(self.axis_names)

    ##
    # Bin wrangling: centers <-> edges, values <-> indices
    ##
    def bin_centers(self, axis=None):
        """Return bin centers along an axis, or if axis=None, list of bin_centers along each axis"""
        if axis is None:
            return np.array([self.bin_centers(axis=i) for i in range(self.dimensions)])
        axis = self.get_axis_number(axis)
        return 0.5*(self.bin_edges[axis][1:] + self.bin_edges[axis][:-1])

    def get_axis_bin_index(self, value, axis):
        """Returns index along axis of bin in histogram which contains value
        Inclusive on both endpoints
        """
        axis = self.get_axis_number(axis)
        bin_edges = self.bin_edges[axis]
        # The right bin edge of np.histogram is inclusive:
        if value == bin_edges[-1]:
            # Minus two: one for bin edges rather than centers, one for 0-based indexing
            return len(bin_edges) - 2
        # For all other bins, it is exclusive.
        result = np.searchsorted(bin_edges, [value], side='right')[0] - 1
        if not 0 <= result <= len(bin_edges) - 1:
            raise CoordinateOutOfRangeException("Value %s is not in range (%s-%s) of axis %s" % (
                value, bin_edges[0], bin_edges[-1], axis))
        return result

    def get_bin_indices(self, values):
        """Returns index tuple in histogram of bin which contains value"""
        return tuple([self.get_axis_bin_index(values[ax_i], ax_i)
                      for ax_i in range(self.dimensions)])

    def all_axis_bin_centers(self, axis):
        """Return ndarray of same shape as histogram containing bin center value along axis at each point"""
        # Arcane hack that seems to work, at least in 3d... hope
        axis = self.get_axis_number(axis)
        return np.meshgrid(*self.bin_centers(), indexing='ij')[axis]

    ##
    # Data reduction: sum, slice, project, ...
    ##
    def sum(self, axis):
        """Sums all data along axis, returns d-1 dimensional histogram"""
        axis = self.get_axis_number(axis)
        if self.dimensions == 2:
            new_hist = Hist1d
        else:
            new_hist = Histdd
        return new_hist.from_histogram(np.sum(self.histogram, axis=axis),
                                       bin_edges=itemgetter(*self.other_axes(axis))(self.bin_edges),
                                       axis_names=self.axis_names_without(axis))

    def rebin_axis(self, reduction_factor, axis):
        """Returns histogram where bins along axis have been reduced by reduction_factor"""
        raise NotImplementedError

    def slice(self, start, stop=None, axis=0):
        """Restrict histogram to bins whose data values (not bin numbers) along axis are between start and stop
        (both inclusive). Returns d dimensional histogram."""
        if stop is None:
            # Make a 1=bin slice
            stop = start
        axis = self.get_axis_number(axis)
        start_bin = self.get_axis_bin_index(start, axis)
        stop_bin = self.get_axis_bin_index(stop, axis)
        new_bin_edges = self.bin_edges.copy()
        new_bin_edges[axis] = new_bin_edges[axis][start_bin:stop_bin + 2]   # TODO: Test off by one here!
        return Histdd.from_histogram(np.take(self.histogram, np.arange(start_bin, stop_bin + 1), axis=axis),
                                     bin_edges=new_bin_edges, axis_names=self.axis_names)

    def slicesum(self, start, stop=None, axis=0):
        """Slices the histogram along axis, then sums over that slice, returning a d-1 dimensional histogram"""
        return self.slice(start, stop, axis).sum(axis)

    def projection(self, axis):
        """Sums all data along all other axes, then return Hist1D"""
        axis = self.get_axis_number(axis)
        projected_hist = np.sum(self.histogram, axis=self.other_axes(axis))
        return Hist1d.from_histogram(projected_hist, bin_edges=self.bin_edges[axis])

    ##
    # Density methods: cumulate, normalize, ...
    ##

    def cumulate(self, axis):
        """Returns new histogram with all data cumulated along axis."""
        axis = self.get_axis_number(axis)
        return Histdd.from_histogram(np.cumsum(self.histogram, axis=axis),
                                     bin_edges=self.bin_edges,
                                     axis_names=self.axis_names)
                                     
    def _simsalabim_slice(self, axis):
        return [slice(None) if i != axis else np.newaxis
                            for i in range(self.dimensions)]

    def normalize(self, axis):
        """Returns new histogram where all values along axis (in one bin of the other axes) sum to 1"""
        axis = self.get_axis_number(axis)
        sum_along_axis = np.sum(self.histogram, axis=axis)
        # Don't do anything for subspaces without any entries -- this avoids nans everywhere
        sum_along_axis[sum_along_axis == 0] = 1
        hist = self.histogram / sum_along_axis[self._simsalabim_slice(axis)]
        return Histdd.from_histogram(hist,
                                     bin_edges=self.bin_edges,
                                     axis_names=self.axis_names)

    def cumulative_density(self, axis):
        """Returns new histogram with all values replaced by their cumulative densities along axis."""
        return self.normalize(axis).cumulate(axis)

    def central_likelihood(self, axis):
        """Returns new histogram with all values replaced by their central likelihoods along axis."""
        result = self.cumulative_density(axis)
        result.histogram = 1 - 2 * np.abs(result.histogram - 0.5)
        return result

    ##
    # Mixed methods: both reduce and summarize the data
    ##

    def percentile(self, percentile, axis, inclusive=True):
        """Returns d-1 dimensional histogram containing percentile of values along axis
        if inclusive=True, will report bin center of first bin for which percentile% of data lies in or below the bin
                    =False, ... data lies strictly below the bin
        10% percentile is calculated as: value at least 10% data is LOWER than
        """
        axis = self.get_axis_number(axis)

        # Using np.where here is too tricky, as it may not return a value for each "bin-columns"
        # First get an array which has a minimum at the percentile-containing bin on each axis
        ecdf = self.cumulative_density(axis).histogram
        if not inclusive:
            density = self.normalize(axis).histogram
            ecdf = ecdf - density
        hist_with_extrema = ecdf - 2 * (ecdf >= percentile / 100)

        # Now find the extremum indices using np.argmin
        percentile_indices = np.argmin(hist_with_extrema, axis=axis)

        # Finally, convert from extremum indices to bin centers
        # See first example under 'Advanced indexing' in http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        index = [np.arange(q) for q in self.histogram.shape]
        index[axis] = percentile_indices
        result = self.all_axis_bin_centers(axis=axis)[index]

        if self.dimensions == 2:
            new_hist = Hist1d
        else:
            new_hist = Histdd
        return new_hist.from_histogram(histogram=result,
                                       bin_edges=itemgetter(*self.other_axes(axis))(self.bin_edges),
                                       axis_names=self.axis_names_without(axis))

    def average(self, axis):
        """Returns d-1 dimensional histogram of (estimated) mean value of axis
        NB this is very different from averaging over the axis!!!
        """
        axis = self.get_axis_number(axis)
        avg_hist = np.ma.average(self.all_axis_bin_centers(axis),
                                 weights=self.histogram, axis=axis)
        if self.dimensions == 2:
            new_hist = Hist1d
        else:
            new_hist = Histdd
        return new_hist.from_histogram(histogram=avg_hist,
                                       bin_edges=itemgetter(*self.other_axes(axis))(self.bin_edges),
                                       axis_names=self.axis_names_without(axis))

    def std(self, axis):
        """Returns d-1 dimensional histogram of (estimated) std value along axis
        NB this is very different from just std of the histogram values (which describe bin counts)
        """
        def weighted_std(values, weights, axis):
            # Stolen from http://stackoverflow.com/questions/2413522
            average = np.average(values, weights=weights, axis=axis)
            average = average[self._simsalabim_slice(axis)]
            variance = np.average((values-average)**2, weights=weights, axis=axis)
            return np.sqrt(variance)
        
        axis = self.get_axis_number(axis)
        std_hist = weighted_std(self.all_axis_bin_centers(axis),
                                weights=self.histogram, axis=axis)
        if self.dimensions == 2:
            new_hist = Hist1d
        else:
            new_hist = Histdd
        return new_hist.from_histogram(histogram=std_hist,
                                       bin_edges=itemgetter(*self.other_axes(axis))(self.bin_edges),
                                       axis_names=self.axis_names_without(axis))

    ##
    # Other stuff
    ##
    def get_random(self, size=10):
        """Returns (size, n_dim) array of random variates from the histogram. 
        Only bin centers can be returned!
        """
        bin_centers_ravel = np.array(np.meshgrid(*self.bin_centers(), 
                                                 indexing='ij')).reshape(2, -1).T
        hist_ravel = self.histogram.ravel()
        hist_ravel = hist_ravel.astype(np.float) / np.nansum(hist_ravel)
        return bin_centers_ravel[np.random.choice(len(bin_centers_ravel), 
                                                  p=hist_ravel, 
                                                  size=size)]

    def lookup(self, *coordinate_arrays):
        """Lookup values at specific points.
        coordinate_arrays: numpy arrays of coordinates, one for each dimension
        e.g. lookup(np.array([0, 2]), np.array([1, 3])) looks up (x=0, y=1) and (x=2, y3).

        Clips if out of range!! TODO: option to throw exception instead.
        TODO: Needs tests!!
        TODO: port to Hist1d... or finally join the classes
        """
        assert len(coordinate_arrays) == self.dimensions
        # Convert each coordinate array to an index array
        index_arrays = [np.clip(np.searchsorted(self.bin_edges[i], coordinate_arrays[i]) - 1,
                                0,
                                len(self.bin_edges[i])-2)
                        for i in range(self.dimensions)]
        # Use the index arrays to slice the histogram
        return self.histogram[index_arrays]

        # Check against slow version:
        # def hist_to_interpolator_slow(mh):
        #      bin_centers_ravel = np.array(np.meshgrid(*mh.bin_centers(), indexing='ij')).reshape(2, -1).T
        #      return NearestNDInterpolator(bin_centers_ravel, mh.histogram.ravel())
        # x = np.random.uniform(0, 400, 100)
        # y = np.random.uniform(0, 200, 100)
        # hist_to_interpolator(mh)(x, y) - hist_to_interpolator_slow(mh)(x, y)

    def plot(self, log_scale=False, cblabel='Number of entries', log_scale_vmin=1, plt=plt, **kwargs):
        if self.dimensions == 1:
            Hist1d.from_histogram(self.histogram, self.bin_edges[0]).plot(**kwargs)
        elif self.dimensions == 2:
            if log_scale:
                kwargs.setdefault('norm', matplotlib.colors.LogNorm(vmin=max(log_scale_vmin, self.histogram.min()),
                                                                    vmax=self.histogram.max()))
            plt.pcolormesh(self.bin_edges[0], self.bin_edges[1], self.histogram.T, **kwargs)
            plt.xlim(np.min(self.bin_edges[0]), np.max(self.bin_edges[0]))
            plt.ylim(np.min(self.bin_edges[1]), np.max(self.bin_edges[1]))
            cb = plt.colorbar(label=cblabel)
            cb.ax.minorticks_on()
            if self.axis_names:
                plt.xlabel(self.axis_names[0])
                plt.ylabel(self.axis_names[1])
        else:
            raise ValueError("Can only plot 1- or 2-dimensional histograms!")

Histdd.project = Histdd.projection


if __name__ == '__main__':
    # Create histograms just like from numpy...
    m = Hist1d([0, 3, 1, 6, 2, 9], bins=3)

    # ...or add data incrementally:
    m = Hist1d(bins=100, range=(-3, 4))
    m.add(np.random.normal(0, 0.5, 10**4))
    m.add(np.random.normal(2, 0.2, 10**3))

    # Get the data back out:
    print(m.histogram, m.bin_edges)

    # Access derived quantities like bin_centers, normalized_histogram, density, cumulative_density, mean, std
    plt.plot(m.bin_centers, m.normalized_histogram, label="Normalized histogram", linestyle='steps')
    plt.plot(m.bin_centers, m.density, label="Empirical PDF", linestyle='steps')
    plt.plot(m.bin_centers, m.cumulative_density, label="Empirical CDF", linestyle='steps')
    plt.title("Estimated mean %0.2f, estimated std %0.2f" % (m.mean, m.std))
    plt.legend(loc='best')
    plt.show()

    # Slicing and arithmetic behave just like ordinary ndarrays
    print("The fourth bin has %d entries" % m[3])
    m[1:4] += 4 + 2 * m[-27:-24]
    print("Now it has %d entries" % m[3])

    # Of course I couldn't resist adding a canned plotting function:
    m.plot()
    plt.show()

    # Create and show a 2d histogram. Axis names are optional.
    m2 = Histdd(bins=100, range=[[-5, 3], [-3, 5]], axis_names=['x', 'y'])
    m2.add(np.random.normal(1, 1, 10**6), np.random.normal(1, 1, 10**6))
    m2.add(np.random.normal(-2, 1, 10**6), np.random.normal(2, 1, 10**6))
    m2.plot()
    plt.show()

    # x and y projections return Hist1d objects
    m2.projection('x').plot(label='x projection')
    m2.projection(1).plot(label='y projection')
    plt.legend()
    plt.show()
