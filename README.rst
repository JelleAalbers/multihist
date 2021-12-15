multihist
===========

.. image:: https://github.com/JelleAalbers/multihist/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/JelleAalbers/multihist/actions/workflows/tests.yml

`https://github.com/JelleAalbers/multihist`

Thin wrapper around numpy's histogram and histogramdd.

Numpy has great histogram functions, which return (histogram, bin_edges) tuples. This package wraps these in a class
with methods for adding new data to existing histograms, take averages, projecting, etc.

For 1-dimensional histograms you can access cumulative and density information, as well as basic statistics (mean and std).
For d-dimensional histograms you can name the axes, and refer to them by their names when projecting / summing / averaging.

**NB**: For a faster and richer histogram package, check out `hist <https://github.com/scikit-hep/hist>`_ from scikit-hep. Alternatively, look at its parent library `boost-histogram <https://github.com/scikit-hep/boost-histogram>`_, which has `numpy-compatible features <https://boost-histogram.readthedocs.io/en/latest/usage/numpy.html>`_. Multihist was created back in 2015, long before those libraries existed.

Synopsis::

    # Create histograms just like from numpy...
    m = Hist1d([0, 3, 1, 6, 2, 9], bins=3)

    # ...or add data incrementally:
    m = Hist1d(bins=100, range=(-3, 4))
    m.add(np.random.normal(0, 0.5, 10**4))
    m.add(np.random.normal(2, 0.2, 10**3))

    # Get the data back out:
    print(m.histogram, m.bin_edges)

    # Access derived quantities like bin_centers, normalized_histogram, density, cumulative_density, mean, std
    plt.plot(m.bin_centers, m.normalized_histogram, label="Normalized histogram", drawstyle='steps')
    plt.plot(m.bin_centers, m.density, label="Empirical PDF", drawstyle='steps')
    plt.plot(m.bin_centers, m.cumulative_density, label="Empirical CDF", drawstyle='steps')
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
