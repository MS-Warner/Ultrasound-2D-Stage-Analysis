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
from lib.simulation import Sim, SimGrid, Source, line, Wave, show_sources, simulate

plot_setup()

#------------------------------------------------------------------------------#

C=Config(
    #analysis config
    analysis_duration = 100,
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
    show_figs = False,
    print_progress = False, print_lists = False,
    show_scan_name = True,
    time_samples_anim = 201, time_samples_1D = 1001,
    )

#load measured data
S = load_scan(198,C)
H=hilbert(S,C,crop_start_spike=True,crop_end=True)
C.show_figs = True #enable plotting after hilbert
tlin = H["tlin"]
DCo = H["DC_offset"]
hilb = H["hilbert_data"]
env = np.abs(hilb)
arg = np.angle(hilb)

fit_params, fit_params_cov = fit_halfsine(tlin, env)
w = fit_params[1]
fitted_env = halfsine(tlin, *fit_params)


fit_peak = np.max(fitted_env)
fit_peak_idx = np.where(fitted_env == fit_peak)[0][0]

"""
#could calculate f; gives f≈0.147MHz
fit_start_idx = fit_peak_idx - round((w/2) / S.dt)
fit_end_idx   = fit_peak_idx + round((w/2) / S.dt)
delta_phase = np.unwrap(arg)[fit_end_idx] - np.unwrap(arg)[fit_start_idx]
delta_time = tlin[fit_end_idx] - tlin[fit_start_idx]

f = delta_phase / (2*np.pi) / delta_time
"""

f= 0.155#MHz, from pulser
n_cycles = w*f
phase = arg[fit_peak_idx]


#construct sim
wave = Wave(speed       =  3.1,#mm/μs
            wavelength  = 20  ,#mm
            n_cycles    =  n_cycles,
            phase_shift =  phase #rad
            )
grid = SimGrid(x_range = (-100,100),
               y_range = (   0,200),
               t_range = (   0,100),#TODO check
               nx = 41,
               ny = 41,
               nt = 1001
               )

#positions of left-side generators only; right-side is handled with mirror_x=True
D1l_pos = [(-53.408,-85.991),(-13.425,-93.041)] #D1
F1Ll_pos = [(-102.138,-69.733), (-83.952,-80.233)] #left half of F1
F1Rl_pos = [(-81.968,-80.955), (-61.287,-84.602)] #right half of F1
#create sources
D1l_sources = line(start = D1l_pos[0], end = D1l_pos[1],
                   n_sources = 50, wave = wave,
                   delays = (50-48.48),
                   amplitudes = 2/6.990398833917378,
                   mirror_x = True)
F1Ll_sources = line(start = F1Ll_pos[0], end = F1Ll_pos[1],
                    n_sources = 50, wave = wave,
                    delays = 0,
                    amplitudes = 2/12.775651257937035,
                    mirror_x = True)
F1Rl_sources = line(start = F1Rl_pos[0], end = F1Rl_pos[1],
                    n_sources = 50, wave = wave,
                    delays = 0,
                    amplitudes = 2/12.775651257937035,
                    mirror_x = True)
sources = D1l_sources + F1Ll_sources + F1Rl_sources


sim = Sim(grid, sources)
show_sources(sim)
print("Simulating...",end="")
S_sim = simulate(sim, name="198 sim")
print("done. ")

ascan(S, C, y=-1, save=True,
      crop_start_spike=True, crop_end=True)
ascan(S_sim, C, y=-1, save=True,
      crop_start_spike=True, crop_end=True)

sono_int(S,C, save=True)
sono_int(S_sim,C, save=True)

