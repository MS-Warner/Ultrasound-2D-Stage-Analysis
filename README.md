# Ultrasound-2D-Stage-Analysis
Code (analysis and other) used in my SH EMAT array focusing MPhys project with the Warwick University Ultrasonics Group. 

## Notes
This code is designed to be runnable straight away, as long as you have the prerequisites installed. After downloading, try opening/running `compare_sim.py` or `all_scans.py`

I developed the code for use in Python's IDLE, using matplotlib's interactive mode. This allows for multiple plots to be generated separately in the shell without blocking further input.

Plots produced are not intended to be of publishable quality, and were used for rapid analysis shortly after obtaining data. I think the plots are best for qualitative understanding, but the analysis is proper and quantitative results are very easy to get out.  

## Prerequisites:
Python 3.10+

The following Python libraries:

standard: pathlib, datetime, dataclasses, re

additional: numpy, scipy, matplotlib

### Capabilities
- converting plaintext lvm data to faster, binary npz data  
- loading an arbitrary number of different scans  
- simulating non-dispersive waves using Huygens' principle  
  - wave properties configurable per source  
  - generators modeled as lines of coherent point sources  
  - half-sine envelope applied to signal  
  - simulation result stored as a Scan object, allowing all the same processing as for measured data  
  - good accuracy when compared to measured data, although not perfect  
- automatic detection of generation noise  
- calculating and plotting:  
  - A-scans  
  - Hilbert transform based envelopes and instantaneous phase  
  - sonograms  
  - spatial voltage distributions at a given time  
  - spatiotemporal voltage distributions  
  - spatial power (and power density) distributions  
  - animations  
- saving all plots for later use  

### Structure
- data.py  
  - contains two dataclasses, Config and Scan.  
  - Config contains shared details relating to analysis.  
  - Scan contains the actual data, as well as some metadata.  
  - Analysis functions generally then accept a Scan object and a Config object, alongside additional optional arguments.  
- lvm2npz.py  
  - allows a folder containing .lvm files to be read, returning data stored in python arrays.  
- loading.py  
  -allows a scan to be loaded from a location given by Config, based on an id given by the user.  
- analysis.py  
  - contains many different analysis functions, primarily focused on plotting.  
- simulation.py  
  - allows construction of simulations, with output stored as a Scan object.  

