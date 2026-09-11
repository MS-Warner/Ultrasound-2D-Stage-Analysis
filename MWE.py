import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
#custom packages
from lib.data import Config
from lib.loading import load_scan, ScanLoadError
#this could be from lib.analysis import *, but relevant functions are listed here explicitly.
from lib.analysis import (plot_setup, ascan, plot_hilbert,
    singletime, max_amp, sono, sono_amp, sono_int,
    plot_xt, plot_yt, animate_xy, show_data,
    FWHM, halfsine, fit_halfsine)
from lib.simulation import (Wave, Source, Sim, SimGrid,
    line, show_sources, simulate)
plot_setup()

#------------------------------------------------------------------------------#

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
    print_progress = False, print_lists = False,
    time_samples_anim = 201, time_samples_1D = 1001,
    )


#Your code goes here :)
#you probably want to start with S=load_scan(YOUR scan_id, C).
#you can then use most analysis functions with default arguments by simply passing S, C:
#show_data(S,C)
