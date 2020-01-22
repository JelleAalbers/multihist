.. :changelog:

History
-------

------------------
0.6.3 (2020-01-22)
------------------
* Feldman-Cousins errors for Hist1d.plot (#10)


------------------
0.6.2 (2020-01-15)
------------------
* Fix rebinning for empty histograms (#9)

------------------
0.6.1 (2019-12-05)
------------------
* Fixes for #7 (#8)

------------------
0.6.0 (2019-06-30)
------------------
* Correct step plotting at edges, other plotting fixes
* Histogram numpy structured arrays
* Fix deprecation warnings (#6)
* `lookup_hist`
* `.max()` and `.min()` methods
* percentile support for higher-dimensional histograms
* Improve Hist1d.get_random (also randomize in bin)

------------------
0.5.4 (2017-09-20)
------------------
* Fix issue with input from dask

------------------
0.5.3 (2017-09-18)
------------------
* Fix python 2 support

------------------
0.5.2 (2017-08-08)
------------------
* Fix colorbar arguments to Histdd.plot (#4)
* percentile for Hist1d
* rebin method for Histdd (experimental)

------------------
0.5.1 (2017-03-22)
------------------
* get_random for Histdd no longer just returns bin centers (Hist1d does stil...)
* lookup for Hist1d. When will I finally merge the classes...

------------------
0.5.0 (2016-10-07)
------------------
* pandas.DataFrame and dask.dataframe support
* dimensions option to Histdd to init axis_names and bin_centers at once

------------------
0.4.3 (2016-10-03)
------------------
* Remove matplotlib requirement (still required for plotting features)

------------------
0.4.2 (2016-08-10)
------------------
* Fix small bug for >=3 d histograms

------------------
0.4.1 (2016-17-14)
------------------
* get_random and lookup for Histdd. Not really tested yet.

------------------
0.4.0 (2016-02-05)
------------------
* .std function for Histdd
* Fix off-by-one errors

------------------
0.3.0 (2015-09-28)
------------------
* Several new histdd functions: cumulate, normalize, percentile...
* Python 2 compatibility

------------------
0.2.1 (2015-08-18)
------------------
* Histdd functions sum, slice, average now also work

----------------
0.2 (2015-08-06)
----------------
* Multidimensional histograms
* Axes naming

--------------------
0.1.1-4 (2015-08-04)
--------------------
Correct various rookie mistakes in packaging...
Hey, it's my first pypi package!

----------------
0.1 (2015-08-04)
----------------
Initial release

* Hist1d, Hist2d
* Basic test suite
* Basic readme
