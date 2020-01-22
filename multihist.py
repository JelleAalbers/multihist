from __future__ import division
from copy import deepcopy
from functools import reduce
try:
    from itertools import izip as zip
except ImportError:
    # Hello, python 3!
    pass

import numpy as np

try:
    from scipy.ndimage import zoom
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    plt = None
    CAN_PLOT = False

COLUMNAR_DATA_SOURCES = []

try:
    import dask
    import dask.dataframe
    import dask.multiprocessing
    WE_HAVE_DASK = True
    DEFAULT_DASK_COMPUTE_KWARGS = dict(get=dask.multiprocessing.get)
    COLUMNAR_DATA_SOURCES.append(dask.dataframe.DataFrame)
except Exception:           # Sometimes dask import succeeds, but throws error when starting up
    WE_HAVE_DASK = False
    pass

try:
    import pandas as pd
    COLUMNAR_DATA_SOURCES.append(pd.DataFrame)
except ImportError:
    pass

COLUMNAR_DATA_SOURCES = tuple(COLUMNAR_DATA_SOURCES)

from operator import itemgetter

__version__ = '0.6.3'


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

    # Let unary operators work on wrapped histogram:
    def min(self):
        return self.histogram.min()

    def max(self):
        return self.histogram.max()

    def __len__(self):
        return len(self.histogram)

    def __neg__(self):
        return self.__class__.from_histogram(-self.histogram, self.bin_edges, self.axis_names)

    def __pos__(self):
        return self.__class__.from_histogram(+self.histogram, self.bin_edges, self.axis_names)

    def __abs__(self):
        return self.__class__.from_histogram(abs(self.histogram), self.bin_edges, self.axis_names)

    def __invert__(self):
        return self.__class__.from_histogram(~self.histogram, self.bin_edges, self.axis_names)

    # Let binary operators work on wrapped histogram

    @classmethod
    def _make_binop(cls, opname):
        def binop(self, other):
            return self.__class__.from_histogram(
                getattr(self.histogram, opname)(other),
                self.bin_edges,
                self.axis_names)
        return binop

for methodname in 'add sub mul div truediv floordiv mod divmod pow lshift rshift and or'.split():
    dundername = '__%s__' % methodname
    setattr(MultiHistBase,
            dundername,
            MultiHistBase._make_binop(dundername))
    setattr(MultiHistBase,
            '__r%s__' % methodname,
            getattr(MultiHistBase, dundername))

# Verbose alias
MultiHistBase.similar_blank_histogram = MultiHistBase.similar_blank_hist


class Hist1d(MultiHistBase):
    axis_names = None
    dimensions = 1

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
        return 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

    def bin_volumes(self):
        return np.diff(self.bin_edges)

    @property
    def density(self):
        """Gives emprical PDF, like np.histogram(...., density=True)"""
        h = self.histogram.astype(np.float)
        bindifs = np.array(np.diff(self.bin_edges), float)
        return h / (bindifs * self.n)

    @property
    def normalized_histogram(self):
        """Gives histogram with sum of entries normalized to 1."""
        return self.histogram / self.n

    @property
    def cumulative_histogram(self):
        return np.cumsum(self.histogram)

    @property
    def cumulative_density(self):
        cs = np.cumsum(self.histogram)
        return cs / cs[-1]

    def get_random(self, size=10):
        """Returns random variates from the histogram.
        Note this assumes the histogram is an 'events per bin', not a pdf.
        Inside the bins, a uniform distribution is assumed.
        """
        bin_i = np.random.choice(np.arange(len(self.bin_centers)), size=size, p=self.normalized_histogram)
        return self.bin_centers[bin_i] + np.random.uniform(-0.5, 0.5, size=size) * self.bin_volumes()[bin_i]

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
            bc = n / (n - 1)
        else:
            bc = 1
        return np.sqrt(np.average((self.bin_centers - self.mean) ** 2, weights=self.histogram)) * bc

    def plot(self,
             normed=False, scale_histogram_by=1.0, scale_errors_by=1.0,
             errors=False, error_style='bar', error_alpha=0.3,
             plt=plt, set_xlim=False,
             **kwargs):
        """Plot the histogram, with error bars if desired.

        :param normed: Scale the histogram so the sum is 1 before plotting.
        Errors are computed before scaling, then scaled accordingly.
        :param scale_histogram_by: Custom multiplier to apply to histogram.
        Errors are computed before scaling, then scaled accordingly.
        :param scale_errors_by: Custom multiplier to apply to errors.
        :param errors: Whether and how to plot 1 sigma error bars
            * False for no errors
            * True or 'fc' for Feldman-Cousin errors.
              For > 20 events, central Poisson intervals are used
            * 'central' for central Poisson confidence intervals
            * 'sqrtn' for sqrt(n) errors
        :param errorstyle: How to plot errors (if errors is not False)
         * 'bar' for error bars
         * 'band' for shaded bands
        :param error_alpha: Alpha multiplier for errorstyle='band'
        :param plt: Object to call plt... on; matplotlib.pyplot by default.
        :param set_xlim: If True, set xlim to the range of the hist
        """
        if not CAN_PLOT:
            raise ValueError(
                "matplotlib did not import, so can't plot your histogram...")
        x = self.bin_edges
        y = self.histogram
        if normed:
            scale_histogram_by /= y.sum()
        if errors == 'sqrtn':
            _yerr = y**0.5
            ylow, yhigh = y - _yerr, y + _yerr
        elif errors == 'central':
            ylow, yhigh = poisson_1s_interval(y, fc=False)
        elif errors:
            ylow, yhigh = poisson_1s_interval(y, fc=True)
        else:
            ylow, yhigh = y, y

        y = y.astype(np.float) * scale_histogram_by
        ylow = ylow.astype(np.float) * scale_histogram_by * scale_errors_by
        yhigh = yhigh.astype(np.float) * scale_histogram_by * scale_errors_by

        if errors and error_style == 'bar':
            kwargs.setdefault('linestyle', 'none')
            kwargs.setdefault('marker', '.')
            plt.errorbar(self.bin_centers,
                         y,
                         yerr=[y - ylow, yhigh - y],
                         **kwargs)
        else:
            # Note we use steps-pre, not steps-mid.
            # If we would have plotted values only vs the centers
            #  * the steps won't be correct for log scales
            #  * the final bins will not fully show
            def fix(q):
                return np.concatenate([[q[0]], q])
            y = fix(y)
            ylow = fix(ylow)
            yhigh = fix(yhigh)

            if errors and error_style == 'band':
                plt.plot(x, y, drawstyle='steps-pre', **kwargs)
                alpha = error_alpha
                if 'alpha' in kwargs:
                    alpha *= kwargs['alpha']
                    del kwargs['alpha']
                kwargs['linewidth'] = 0
                if 'label' in kwargs:
                    # Don't want to double-label!
                    del kwargs['label']
                plt.fill_between(x, ylow, yhigh,
                                 alpha=alpha,
                                 step='pre', **kwargs)
            else:
                plt.plot(x, y, drawstyle='steps-pre', **kwargs)

        if set_xlim:
            plt.xlim(x[0], x[-1])

    def percentile(self, percentile):
        """Return bin center nearest to percentile"""
        return self.bin_centers[np.argmin(np.abs(self.cumulative_density * 100 - percentile))]

    def lookup(self, coordinates):
        """Lookup values at coordinates.
        coordinates: arraylike of coordinates.

        Clips if out of range!! TODO: option to throw exception instead.
        TODO: Needs tests!!
        """
        # Convert coordinates to indices
        index_array = np.clip(np.searchsorted(self.bin_edges, coordinates) - 1,
                              0,
                              len(self.bin_edges) - 2)

        # Use the index array to slice the histogram
        return self.histogram[index_array]


class Histdd(MultiHistBase):
    """multidimensional histogram object
    """
    axis_names = None

    @classmethod
    def from_histogram(cls, histogram, bin_edges, axis_names=None):
        """Make a HistdD from numpy histogram + bin edges
        :param histogram: Initial histogram
        :param bin_edges: x bin edges of histogram, y bin edges, ...
        :param axis_names: Names of axes
        :return: Histnd instance
        """
        bin_edges = np.array(bin_edges)
        self = cls(bins=bin_edges, axis_names=axis_names)
        self.histogram = histogram
        return self

    def __init__(self, *data, **kwargs):
        for k, v in {'bins': 10, 'range': None, 'weights': None, 'axis_names': None}.items():
            kwargs.setdefault(k, v)

        # dimensions is a shorthand [(axis_name_1, bins_1), (axis_name_2, bins_2), ...]
        if 'dimensions' in kwargs:
            kwargs['axis_names'], kwargs['bins'] = zip(*kwargs['dimensions'])

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

        self.axis_names = kwargs['axis_names']
        self.histogram, self.bin_edges = self._data_to_hist(data, **kwargs)

    def add(self, *data, **kwargs):
        self.histogram += self._data_to_hist(data, **kwargs)[0]

    @staticmethod
    def _is_columnar(x):
        if isinstance(x, COLUMNAR_DATA_SOURCES):
            return True
        if isinstance(x, np.ndarray) and x.dtype.fields:
            return True
        return False

    def _data_to_hist(self, data, **kwargs):
        """Return bin_edges, histogram array"""
        if hasattr(self, 'bin_edges'):
            kwargs.setdefault('bins', self.bin_edges)

        if len(data) == 1 and self._is_columnar(data[0]):
            data = data[0]

            if self.axis_names is None:
                raise ValueError("When histogramming from a columnar data source, "
                                 "axis_names or dimensions is mandatory")
            is_dask = False
            if WE_HAVE_DASK:
                is_dask = isinstance(data, dask.dataframe.DataFrame)

            if is_dask:
                fake_histogram = Histdd(axis_names=self.axis_names, bins=kwargs['bins'])

                partial_hists = []
                for partition in data.to_delayed():
                    ph = dask.delayed(Histdd)(partition, axis_names=self.axis_names, bins=kwargs['bins'])
                    ph = dask.delayed(lambda x: x.histogram)(ph)
                    ph = dask.array.from_delayed(ph,
                                                 shape=fake_histogram.histogram.shape,
                                                 dtype=fake_histogram.histogram.dtype)
                    partial_hists.append(ph)
                partial_hists = dask.array.stack(partial_hists, axis=0)

                compute_options = kwargs.get('compute_options', {})
                for k, v in DEFAULT_DASK_COMPUTE_KWARGS.items():
                    compute_options.setdefault(k, v)
                histogram = partial_hists.sum(axis=0).compute(**compute_options)

                bin_edges = fake_histogram.bin_edges
                return histogram, bin_edges

            else:
                data = np.vstack([data[x].values if isinstance(data, pd.DataFrame) else data[x]
                                  for x in self.axis_names])

        data = np.array(data).T
        return np.histogramdd(data,
                              bins=kwargs.get('bins'),
                              weights=kwargs.get('weights'),
                              range=kwargs.get('range'))

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
        return 0.5 * (self.bin_edges[axis][1:] + self.bin_edges[axis][:-1])

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

    def slice(self, start, stop=None, axis=0):
        """Restrict histogram to bins whose data values (not bin numbers) along axis are between start and stop
        (both inclusive). Returns d dimensional histogram."""
        if stop is None:
            # Make a 1=bin slice
            stop = start
        axis = self.get_axis_number(axis)
        start_bin = max(0, self.get_axis_bin_index(start, axis))
        stop_bin = min(len(self.bin_centers(axis)) - 1,  # TODO: test off by one!
                       self.get_axis_bin_index(stop, axis))
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
        hist = self.histogram / sum_along_axis[tuple(self._simsalabim_slice(axis))]
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

        # Shape of histogram
        s = self.histogram.shape

        # Shape of histogram after axis has been collapsed to 1
        s_collapsed = list(s)
        s_collapsed[axis] = 1

        # Shape of histogram with axis removed entirely
        s_removed = np.concatenate([s[:axis], s[axis + 1:]]).astype(np.int)

        # Using np.where here is too tricky, as it may not return a value for each "bin-columns"
        # First, get an array which has a minimum at the percentile-containing bins
        # The minimum may not be unique: if later bins are empty, they will not be
        ecdf = self.cumulative_density(axis).histogram
        if not inclusive:
            raise NotImplementedError("Non-inclusive percentiles not yet implemented")
        ecdf = np.nan_to_num(ecdf)    # Since we're relying on self-equality later
        x = ecdf - 2 * (ecdf >= percentile / 100)

        # We now want to get the location of the minimum
        # To ensure it is unique, add a very very very small monotonously increasing bit to x
        # Nobody will want 1e-9th percentiles, right? TODO
        sz = np.ones(len(s), dtype=np.int)
        sz[axis] = -1
        x += np.linspace(0, 1e-9, s[axis]).reshape(sz)

        # 1. Find the minimum along the axis
        # 2. Reshape to s_collapsed and perform == to get a mask
        # 3. Apply the mask to the bin centers along axis
        # 4. Unflatten with reshape
        result = self.all_axis_bin_centers(axis)[
            x == np.min(x, axis=axis).reshape(s_collapsed)
        ]
        result = result.reshape(s_removed)

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
    def bin_volumes(self):
        return reduce(np.multiply, np.ix_(*[np.diff(bs) for bs in self.bin_edges]))

    def rebin(self, *factors, **kwargs):
        """Return a new histogram that is 'rebinned' (zoomed) by factors (tuple of floats) along each dimensions
          factors: tuple with zoom factors along each axis. e.g. 2 = double number of bins, 0.5 = halve them.
          order: Order for spline interpolation in scipy.ndimage.zoom. Defaults to  linear interpolation (order=1).

        The only accepted keyword argument is 'order'!!! (python 2 is not nice)

        The normalization is set to the normalization of the current histogram
        The factors don't have to be integers or fractions: scipy.ndimage.zoom deals with the rebinning arcana.
        """
        if not HAVE_SCIPY:
            raise NotImplementedError("Rebinning requires scipy.ndimage")
        if any([x != 'order' for x in kwargs.keys()]):
            raise ValueError("Only 'order' keyword argument is accepted. Yeah, this is confusing.. blame python 2.")
        order = kwargs.get('order', 1)

        # Construct a new histogram
        mh = self.similar_blank_histogram()

        if not len(factors) == self.dimensions:
            raise ValueError("You must pass %d rebin factors to rebin a %d-dimensional histogram" % (
                self.dimensions, self.dimensions
            ))

        # Zoom the bin edges.
        # It's a bit tricky for non-uniform bins:
        # we first construct a linear interpolator to take
        # fraction along axis -> axis coordinate according to current binning.
        # Then we feed it the new desired binning fractions.
        for i, f in enumerate(factors):
            x = self.bin_edges[i]
            mh.bin_edges[i] = np.interp(
                x=np.linspace(0, 1, round((len(x) - 1) * f) + 1),
                xp=np.linspace(0, 1, len(x)),
                fp=x)

        # Rebin the histogram using ndimage.zoom, then renormalize
        mh.histogram = zoom(self.histogram, factors, order=order)
        if mh.histogram.sum() != 0:
            mh.histogram *= self.histogram.sum() / mh.histogram.sum()
        # mh.histogram /= np.product(factors)

        return mh

    def get_random(self, size=10):
        """Returns (size, n_dim) array of random variates from the histogram.
        Inside the bins, a uniform distribution is assumed
        Note this assumes the histogram is an 'events per bin', not a pdf.
        TODO: test more.
        """
        # Sample random bin centers
        bin_centers_ravel = np.array(np.meshgrid(*self.bin_centers(),
                                                 indexing='ij')).reshape(self.dimensions, -1).T
        hist_ravel = self.histogram.ravel()
        hist_ravel = hist_ravel.astype(np.float) 
        hist_ravel = hist_ravel / np.nansum(hist_ravel)
        result = bin_centers_ravel[np.random.choice(len(bin_centers_ravel),
                                                    p=hist_ravel,
                                                    size=size)]

        # Randomize the position inside the bin
        for dim_i in range(self.dimensions):
            bin_edges = self.bin_edges[dim_i]
            bin_widths = np.diff(bin_edges)

            # Note the - 1: for the first bin's bin center, searchsorted gives 1, but we want 0 here:
            index_of_bin = np.searchsorted(bin_edges, result[:, dim_i]) - 1
            result[:, dim_i] += (np.random.rand(size) - 0.5) * bin_widths[index_of_bin]

        return result

    def lookup(self, *coordinate_arrays):
        """Lookup values at specific points.
        coordinate_arrays: numpy arrays of coordinates, one for each dimension
        e.g. lookup(np.array([0, 2]), np.array([1, 3])) looks up (x=0, y=1) and (x=2, y3).

        Clips if out of range!! TODO: option to throw exception instead.
        TODO: Needs tests!!
        TODO: port to Hist1d... or finally join the classes
        TODO: Support for scalar arguments
        """
        assert len(coordinate_arrays) == self.dimensions
        # Convert each coordinate array to an index array
        index_arrays = [np.clip(np.searchsorted(self.bin_edges[i], coordinate_arrays[i]) - 1,
                                0,
                                len(self.bin_edges[i]) - 2)
                        for i in range(self.dimensions)]
        # Use the index arrays to slice the histogram
        return self.histogram[tuple(index_arrays)]

        # Check against slow version:
        # def hist_to_interpolator_slow(mh):
        #      bin_centers_ravel = np.array(np.meshgrid(*mh.bin_centers(), indexing='ij')).reshape(2, -1).T
        #      return NearestNDInterpolator(bin_centers_ravel, mh.histogram.ravel())
        # x = np.random.uniform(0, 400, 100)
        # y = np.random.uniform(0, 200, 100)
        # hist_to_interpolator(mh)(x, y) - hist_to_interpolator_slow(mh)(x, y)

    def lookup_hist(self, mh):
        """Return histogram within binning of Histdd mh, with values looked up in this histogram.

        This is not rebinning: no interpolation /renormalization is performed.
        It's just a lookup.
        """
        result = mh.similar_blank_histogram()
        points = np.stack([mh.all_axis_bin_centers(i)
                           for i in range(mh.dimensions)]).reshape(mh.dimensions, -1)
        values = self.lookup(*points)
        result.histogram = values.reshape(result.histogram.shape)
        return result

    def plot(self, log_scale=False, log_scale_vmin=1,
             colorbar=True,
             cblabel='Number of entries',
             colorbar_kwargs=None,
             plt=plt,
             **kwargs):

        if colorbar_kwargs is None:
            colorbar_kwargs = dict()
        colorbar_kwargs['label'] = cblabel

        if not CAN_PLOT:
            raise ValueError("matplotlib did not import, so can't plot your histogram...")

        if self.dimensions == 1:
            return Hist1d.from_histogram(self.histogram, self.bin_edges[0]).plot(**kwargs)

        elif self.dimensions == 2:
            if log_scale:
                kwargs.setdefault('norm', matplotlib.colors.LogNorm(vmin=max(log_scale_vmin, self.histogram.min()),
                                                                    vmax=self.histogram.max()))
            mesh = plt.pcolormesh(self.bin_edges[0], self.bin_edges[1], self.histogram.T, **kwargs)
            plt.xlim(np.min(self.bin_edges[0]), np.max(self.bin_edges[0]))
            plt.ylim(np.min(self.bin_edges[1]), np.max(self.bin_edges[1]))
            if self.axis_names:
                plt.xlabel(self.axis_names[0])
                plt.ylabel(self.axis_names[1])
            if colorbar:
                cb = plt.colorbar(**colorbar_kwargs)
                cb.ax.minorticks_on()
                return mesh, cb
            return mesh

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

    if CAN_PLOT:
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
    if CAN_PLOT:
        m.plot()
        plt.show()

        # Create and show a 2d histogram. Axis names are optional.
        m2 = Histdd(bins=100, range=[[-5, 3], [-3, 5]], axis_names=['x', 'y'])
        m2.add(np.random.normal(1, 1, 10**6), np.random.normal(1, 1, 10**6))
        m2.add(np.random.normal(-2, 1, 10**6), np.random.normal(2, 1, 10**6))

        # x and y projections return Hist1d objects
        m2.projection('x').plot(label='x projection')
        m2.projection(1).plot(label='y projection')

        plt.legend()
        plt.show()


##
# Error bar helpers
##

# Zero-background 1 sigma Poisson Feldman-Cousins intervals
# From table II in https://arxiv.org/pdf/physics/9711021.pdf
_fc_intervals = np.array([
    [0.0, 1.29],
    [0.37, 2.75],
    [0.74, 4.25],
    [1.1, 5.3],
    [2.34, 6.78],
    [2.75, 7.81],
    [3.82, 9.28],
    [4.25, 10.3],
    [5.3, 11.32],
    [6.44, 12.79],
    [6.78, 13.81],
    [7.81, 14.82],
    [8.83, 16.29],
    [9.28, 17.3],
    [10.3, 18.32],
    [11.32, 19.32],
    [12.33, 20.8],
    [12.79, 21.81],
    [13.81, 22.82],
    [14.82, 23.82],
    [15.83, 25.3]])


def poisson_central_interval(k, cl=0.6826894921370859):
    """Return central Poisson confidence interval
    :param k: observed events
    :param cl: confidence level
    """
    if not HAVE_SCIPY:
        raise NotImplementedError("Poisson errors require scipy")
    # Adapted from https://stackoverflow.com/a/14832525
    k = np.asarray(k).astype(np.int)
    alpha = 1 - cl
    low = stats.chi2.ppf(alpha / 2, 2 * k) / 2
    high = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return np.stack([np.nan_to_num(low), high])


def poisson_1s_interval(k, fc=True):
    """Return (low, high) 1 sigma Poisson confidence intervals

    :param k: Observed events (int or array of ints)
    :param fc: if True (default), use Feldman-Cousins for k <= 20,
    and central intervals otherwise.
    (at k = 20, the difference between these is 1-2%).
    """
    k = np.asarray(k).astype(np.int)
    result = poisson_central_interval(k)
    if fc:
        mask = k <= 20
        result[:, mask] = _fc_intervals[k[mask]].T
    return result
