multihist
===========
Convenience wrappers around numpy's histogram and histogram2d.

Numpy has great histogram functions, which return (histogram, bin_edges) tuples.
These are a bit cumbersome to keep dragging around; adding new data to a histogram,
and especially doing operations like slicing and projection on 2d histograms while keeping the histogram <-> bins association
can be difficult.

Synopsis::

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

If you're looking for a more fancy histogram class, you might like / eventually get used to ROOT (`https://root.cern.ch/root/html/TH1.html`) and one of its python bindings (pyROOT, rootpy).