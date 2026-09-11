import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
#custom packages
from lib.data import Config
from lib.loading import load_scan, ScanLoadError
from lib.analysis import (plot_setup, ascan, hilbert,
    singletime, max_amp, sono, sono_amp, sono_int,
    plot_xt, plot_yt, animate_xy, show_data,
    FWHM, halfsine, fit_halfsine)
from lib.simulation import (Wave, Source, Sim, SimGrid,
    line, show_sources, simulate)
plot_setup() #set font and sizes, and turn on interactive mode with plt.ion()

#find current scan without having to write the whole thing
#works either with numbers at start of name, or any string in name
scan_ids = [54,"2D1+2F1 wide",210,"D1 singlepoint"]

C = Config(
    #analysis config
    analysis_duration = 80,
    f_range = (0, 1000),
    #sonogram config
    sono_win_time = 32, sono_overlap_time = 28,
    sono_nfft_factor = 8,
    sono_amp_nfft_factor = 1, sono_int_nfft_factor = 1,
    #save config
    lvm_dir   = Path.cwd() / "Data",
    npz_dir   = Path.cwd() / "Conv",
    save_root = Path.cwd() / "Plots",
    #coordinate corrections (optional)
    shift_x = -80, shift_y = -50,
    #plot config (optional)
    show_figs = True,
    print_progress = True, print_lists = False,
    show_scan_name = True,
    time_samples_anim = 201, time_samples_1D = 1001,
    )

scans = []
#iterate through chosen scans
for scan_id in scan_ids:
    try:
        S = load_scan(scan_id, C)
    except ScanLoadError as e:
        print(f"Scan {scan_id} failed: {e}")
        continue
    scans += [S]
"""
print("\nNote: many different options have been selected for these scans for demonstration purposes. Because of this, the plots may appear inconsistent or unuseful.")
#for a point-scan
input("\nPress enter to generate point-scan plots (scan 227): ")
As =   ascan(scans[3], C, save = True)
Hi = hilbert(scans[3], C, smoothing_n=10)
So =    sono(scans[3], C)
#show_data will call ascan

#for an x-scan
input("\nPress enter to generate x-scan plots (scan 210): ")
As =      ascan(scans[2], C, x=-1, crop_start_spike=False, crop_end=True)
Hi =    hilbert(scans[2], C, smoothing_n=100)
So =       sono(scans[2], C, crop_start_spike=True)
ST = singletime(scans[2], C, t_index = scans[2].clip_end_index)
MA =    max_amp(scans[2], C, crop_end=True, t_range=(scans[2].post_spike_start_time+C.analysis_duration,np.inf))
SA =   sono_amp(scans[2], C)
SI =   sono_int(scans[2], C, crop_end=False)
XT =    plot_xt(scans[2], C)
#show_data will call plot_xt
    
#for a y-scan
input("\nPress enter to generate y-scan plots (scan 54): ")
As =      ascan(scans[0], C, y=7, t_range=(25,165))
Hi =    hilbert(scans[0], C, smoothing_n=1000)
So =       sono(scans[0], C)
ST = singletime(scans[0], C, t_index=-1)
MA =    max_amp(scans[0], C)
SA =   sono_amp(scans[0], C)
SI =   sono_int(scans[0], C, f_range=(300,1000))
YT =    plot_yt(scans[0], C)
#show_data will call plot_yt   
"""
#for a 2D-scan
#input("\nPress enter to generate 2D-scan plots (scan 198): ")
As =      ascan(scans[1], C, y=-1, show_analysis_range=False)
Hi =    hilbert(scans[1], C, x=12, y=27, f_range=(50,250))
So =       sono(scans[1], C)
ST = singletime(scans[1], C)
MA =    max_amp(scans[1], C)
SA =   sono_amp(scans[1], C)
SI =   sono_int(scans[1], C, crop_end=False)
An = animate_xy(scans[1], C, save=True, crop_end = True)
#show_data will call animate_xy 

"""

#simulation

wave = Wave(3.1,
            20,
            4.38789143, #from Hilbert fit
            2.342946,   #from Hilbert fit
            )
sources = line((-20,-90),
               ( 20,-90),
               40,
               wave,
               )
grid = SimGrid((-100,100),
               (   0,200),
               (  0,100),
               101,
               101,
               1001
               )
sim = Sim(grid,
          sources,
          use_simpler_wave = False)

Ssim = simulate(sim, name = "testsim")

show_sources(sim)
out = show_data(Ssim, C, save = True)
"""
