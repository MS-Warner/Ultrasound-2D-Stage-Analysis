import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
#custom packages
from lib.data import Config
from lib.loading import load_scan, ScanLoadError
from lib.analysis import (plot_setup, ascan, plot_hilbert,
    singletime, max_amp, sono, sono_amp, sono_int,
    plot_xt, plot_yt, animate_xy, show_data,
    FWHM, halfsine, fit_halfsine)
from lib.simulation import (Wave, Source, Sim, SimGrid,
    line, show_sources, simulate)
plot_setup() #set font and sizes, and turn on interactive mode with plt.ion()

#find current scan without having to write the whole thing
#works either with numbers at start of name, or any string in name
scan_ids = [54,"2D1+2F1 wide",210,"D1 singlepoint"]

C=Config(
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

scans=[]
#iterate through chosen scans
for scan_id in scan_ids:
    try:
        S = load_scan(scan_id, C)
    except ScanLoadError as e:
        print(f"Scan {scan_id} failed: {e}")
        failed_scan_ids.append(scan_id)
        continue
    scans += [S]


As=ascan(scans[0], C, save=True)
Hi=plot_hilbert(scans[1],C)
So=sono(scans[2], C)
ST=singletime(scans[3], C)
MA=max_amp(scans[0],C)
SA=sono_amp(scans[1], C)
SI=sono_int(scans[2], C)
SD=show_data(scans[3], C)
    
    


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
          use_simpler_wave=False)

Ssim = simulate(sim, name="testsim")

show_sources(sim)
out=show_data(Ssim, C, remove_spike=False, save=True)
"""
