multihist
===========

.. image:: https://travis-ci.org/JelleAalbers/multihist.svg?branch=master
    :target: https://travis-ci.org/JelleAalbers/multihist

`https://github.com/JelleAalbers/multihist`

Thin wrapper around numpy's histogram and histogramdd.

Numpy has great histogram functions, which return (histogram, bin_edges) tuples. This package wraps these in a class
with methods for adding new data to existing histograms, take averages, projecting, etc.

For 1-dimensional histograms you can access cumulative and density information, as well as basic statistics (mean and std).
For d-dimensional histograms you can name the axes, and refer to them by their names when projecting / summing / averaging.

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


Alternatives
------------
Of course, the easiest alternative is just to use np.histogram without any wrappers.

If you're looking for a more fancy histogram class, and don't mind installing a big framework,
you might like the particle physics workhorse ROOT (`https://root.cern.ch/root/html/TH1.html`) and one of its python bindings (pyROOT, rootpy).

If you do come from a ROOT background, you might appreciate pyhistogram (`https://github.com/cbourjau/pyhistogram`) instead,
which has a much more ROOT-like interface.

Another python histogram package oriented towards physics is `http://docs.danse.us/histogram/0.2.1/intro.html`. This has support for physical units in histograms and error propagation, but the interface is further removed from numpy. 
