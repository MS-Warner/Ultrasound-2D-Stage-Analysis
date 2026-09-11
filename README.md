Note: readme isn't finished!

# Ultrasound-2D-Stage-Analysis
Code (analysis and other) used in my SH EMAT array focusing MPhys project with the Warwick University Ultrasonics Group.

This code is designed to work primarily with existing data generated in .lvm format and stored in .npz format. 

Plots produced are not intended to be of publishable quality, and were used for rapid analysis shortly after obtaining data. I think the plots are best for qualitative understanding, but the analysis is proper and quantitative results are very easy to get out.

# Prerequisites:
Python 3.10+

The following Python libraries:

standard: pathlib, datetime, dataclasses, re

additional: numpy, scipy, matplotlib

# Capabilities
#TODO: mention interactivity

-loading an arbitrary number of different scans, allowing rapid comparison between data sets.

-constructing and running Huygens' principle based simulations of non-dispersive waves in , with each source having configurable wave properties

-obtaining and plotting:

A-scans, sonograms, Spatial voltage distributions, spatial power (and power density) distributions, Hilbert transform based envelopes and instantaneous phase, animations, and more.

# Structure

data.py
-contains two dataclasses, Config and Scan.

Config contains shared details relating to analysis.
Scan contains the actual data, as well as some metadata.

Analysis functions generally then accept a Scan object and a Config object, alongside additional optional arguments.


lvm2npz.py

-allows a folder containing .lvm files to be read, returning data stored in python arrays.


loading.py

-allows a scan to be loaded from a location given by Config, based on an id given by the user.


analysis.py

-contains many different analysis functions, primarily focused on plotting.


simulation.py

-allows construction of simulations, with output stored as a Scan object.
