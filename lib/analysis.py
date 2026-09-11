import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
from scipy.signal import spectrogram, windows
from scipy.signal import hilbert as scipy_hilbert
from scipy.optimize import curve_fit
from datetime import datetime
from pathlib import Path
#handle imports correctly when run as part of the library or standalone
if __package__:
    from .data import Scan, Config
    from .loading import get_save_dir
else:
    from data import Scan, Config
    from loading import get_save_dir

"""--------------------Plotting helper functions-------------------"""

def plot_setup(font_family = "serif",
               font_size   = 18,
               fonts       = ["CMU Serif", "CMU Serif Extra", "DejaVu Serif"],
               mathtext_fontset = "cm"):
    """Perform aesthetic setup for plotting. Should be called before generating any plots for consistent results."""
    if font_family not in ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']:
        print("plot_setup failed: font_family {font_family} not recognised. ")
    else:
        
        plt.rcParams["font.family"] = font_family
        plt.rcParams[f"font.{font_family}"] = fonts
    plt.rcParams["font.size"] = font_size
    plt.rcParams["mathtext.fontset"] = mathtext_fontset
    plt.ion()

def _timestamp() -> str:
    """Generate timestamp used to ensure save files are uniquely named."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _save_fig(fig: Figure,
             loc: Path
             ) -> None:
    """Create scan folder if not present already."""
    loc.parent.mkdir(parents=True, exist_ok=True)
    #save figure
    fig.savefig(loc, bbox_inches="tight", dpi=300)


def _get_default_t_range(scan: Scan,
                        config: Config
                        ) -> tuple[float, float]:
    """Return default time range for analysis, used when t_range is unset.
    Default is to limit analysis to config.analysis_duration.
    """
    if scan.has_generation_spike:
        start = scan.post_spike_start_time
    else:
        start = scan.tlin[0]
    return (start, start + config.analysis_duration)


def _select_time_data(scan: Scan,
                      config: Config,
                      crop_start_spike: bool = True,
                      crop_end: bool = True,
                      t_range: tuple[float, float] = (-np.inf, np.inf),
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Helper used by most plotting functions. Note t_range acts in combination with crop_start_spike and crop_end"""
    if crop_start_spike:
        the_vdata = scan.vdata_post_spike
        the_tlin = scan.tlin_post_spike
    else:
        the_vdata = scan.vdata
        the_tlin = scan.tlin
    if crop_end:
        end_time = the_tlin[0] + config.analysis_duration
        #here, an inclusive upper bound is used. An exclusive bound might be more appropriate.
        t_mask = (the_tlin <= end_time)
        the_tlin = the_tlin[t_mask]
        the_vdata = the_vdata[:,t_mask,:]
    #crop in time too (default uncropped)
    t_mask = (the_tlin >= t_range[0]) & (the_tlin <= t_range[1])
    the_vdata = the_vdata[:,t_mask,:]
    the_tlin = the_tlin[t_mask]
    if the_tlin.size == 0:
        raise IndexError("No time data remains after cropping! Your t_range most likely does not overlap with the selected data.")
    
    return the_vdata, the_tlin

#TODO could also allow indexing with arrays like x=[1,4,6] to extract non-contiguous regions. This would require use of np.ix(..)
def _select_spatial_data(data: np.ndarray,
                         x: int | None = None, #default corresponds to no cropping with slice(None)
                         y: int | None = None, #default corresponds to no cropping with slice(None)
                         ) -> np.ndarray:
    """Helper function to extract subset of data.
    This should generally be called after _select_time_data(..), and has inputs structured accordingly.
    If x/y is omitted, no indexing is performed over that dimension (via use of slice(None))."""
    #validate x,y inputs
    ny, _, nx = np.shape(data)
    if x is None: x = nx // 2
    if y is None: y = ny // 2
    if not -nx <= x < nx:
        raise IndexError(
             f"x={x} outside valid range [{-nx}, {nx-1}]"
             )
    if not -ny <= y < ny:
        raise IndexError(
             f"y={y} outside valid range [{-ny}, {ny-1}]"
             )
    #allows for negative indices to also be handled
    def idx(i: int | None, size: int) -> slice:
        if i is None:
            return slice(None) #no indexing
        if i < 0:
            i += size
        return slice(i, i + 1) #single index in slice form

    return data[idx(y, data.shape[0]),
                :,
                idx(x, data.shape[2])]

def _spectrogram_helper(config: Config,
                        singlepoint: np.ndarray,
                        tlin: np.ndarray,
                        f_range: tuple[float, float] | None = None,
                        nfft_factor: int = 1, #override for config value, used in sono_amp and sono_int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate spectrogram data for a single point.
    If desired, data should be cropped in time before calling this function."""
    #recalculate dt to avoid needing to pass scan.
    dt = tlin[1]-tlin[0]
    
    f, t, Sxx = spectrogram(singlepoint,
                            1/dt,
                            nperseg  = int(config.sono_win_time / dt),
                            noverlap = int(config.sono_overlap_time / dt),
                            nfft     = int(nfft_factor * config.sono_win_time / dt),
                            window   = windows.hann(int(config.sono_win_time / dt))
                            )
    f   *= 1e3 #convert MHz -> kHz
    Sxx *= 1e-3 #convert mV$^2$/MHz -> mV$^2$/kHz

    t += tlin[0] #output from spectrogram(..) starts at zero, so add correction here.
    
    if f_range is None: #default
        f_mask = (f >= config.f_range[0]) & (f <= config.f_range[1])
    else:
        f_mask = (f >= f_range[0] ) & (f <= f_range[1] )
    f = f[f_mask]
    Sxx = Sxx[f_mask]
    return f, t, Sxx

def _finalise_plot(fig: Figure,
                   scan: Scan,
                   config: Config,
                   save: bool,
                   filename: str | None,
                   print_data: np.ndarray | None = None,
                   ) -> None:
    """Perform steps shared by all plots (except animation and FWHM)"""
    if config.print_lists and print_data is not None:
        _print_lists(print_data)
    #add final text overlay if desired, useful when generating many figures for multiple scans.
    if config.show_scan_name:
        #place text above title
        text = fig.suptitle(scan.name,
                            fontsize=plt.rcParams["font.size"],
                            color="k", alpha=0.5,
                            y=1.0)
    fig.tight_layout(pad=0.15)
    if save:
        _save_fig(fig, get_save_dir(scan, config) / filename)
    if config.show_figs: #ensure each figure is rendered sequentially
        if not plt.isinteractive():
            #only necessary without plt.ion()
            plt.show(block=False)
        #these ensure the figure is actually rendered before the
        # next one starts loading, instead of just an empty window.
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        plt.close(fig)

def _print_lists(print_data: dict
                 ) -> None:
    """Given a dictionary, print keys and values."""
    for key in print_data.keys():
        print(f"{key}:")
        data = print_data[key]
        #Convert data to normal list if it's a numpy array etc.
        #Explicitly don't convert strings.
        if hasattr(data, "tolist") and not isinstance(data, str):
            data = data.tolist()
        print(data)
        print()

"""--------------------------Single point--------------------------"""
def ascan(scan: Scan,
          config: Config,
          x: int | None = None,
          y: int | None = None,
          save: bool = False,
          crop_start_spike: bool = False,
          crop_end: bool = False,
          t_range: tuple[float, float] = (-np.inf, np.inf),
          show_analysis_range: bool = True,
          ) -> tuple[np.ndarray, np.ndarray]:
    """Show voltage trace at location (x,y) by index.
    Allows for optional temporal cropping.
    Returns dictionary containing tlin, vdata."""
    #1. select data
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    singlepoint = _select_spatial_data(the_vdata, 
                                       x, y)[0,:,0]
    return_data = {"tlin": the_tlin,
                   "vdata": singlepoint}
    #only generate fig if necessary
    if config.show_figs or save:
        #2. plot
        fig, ax = plt.subplots()
        ax.plot(the_tlin, singlepoint, label="Data", color="b")
        if show_analysis_range:
            t_range = _get_default_t_range(scan,config)
            ax.vlines(t_range, ymin=np.min(singlepoint), ymax=np.max(singlepoint), color="k", linestyle="--")
        #3. format plot
        ax.set_xlim(the_tlin[0], the_tlin[-1]+scan.dt)
        if x is None: x = scan.nx // 2
        if y is None: y = scan.ny // 2
        ax.set_title(f"A-scan for point\n({x},{y}) at ({round(scan.xlin[x])},{round(scan.ylin[y])}) mm")
        ax.set_xlabel(r"Time (μs)")
        ax.set_ylabel("Voltage (mV)")
        #4. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"Ascan ({x},{y}) [{tstamp}].png"
        else:
            filename = None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists:
        _print_lists(return_data)
    return return_data

def hilbert(scan: Scan,
            config: Config,
            x: int | None = None,
            y: int | None = None,
            save: bool = False,
            crop_start_spike: bool = False,
            crop_end: bool = False,
            t_range: tuple[float, float] = (-np.inf, np.inf),
            f_range: tuple[float, float] | None = None,
            show_analysis_range: bool = True,
            smoothing_n: int = 100,
            ) -> tuple[np.ndarray, np.ndarray]:
    """Show voltage trace and hilbert envelope at location (x,y) by index.
    Additionally, plot hilbert phase and derived instantaneous frequency (smoothed by a moving average)
    Returns dictionary containing tlin, vdata, hilbert_data."""
    #1. select data
    the_vdata_t, the_tlin = _select_time_data(scan, config,
                                              crop_start_spike, crop_end,
                                              t_range)
    singlepoint = _select_spatial_data(the_vdata_t,
                                     x, y)[0,:,0]
    #2. calculate hilbert properties
    #calculate and subtract mean as an estimate of DC offset
    DC_offset = np.mean(singlepoint)
    hilb = scipy_hilbert(singlepoint - DC_offset)
    
    phase = np.unwrap(np.angle(hilb))
    freq = np.gradient(phase, scan.dt) / (2*np.pi) * 1e3 #constants convert Mrad/s->MHz->kHz
    averaged_freq = moving_average(freq, smoothing_n)
    
    lower_env = -np.abs(hilb) + DC_offset
    upper_env =  np.abs(hilb) + DC_offset
    
    return_data = {"tlin": the_tlin,
                   "vdata": singlepoint,
                   "DC_offset": DC_offset,
                   "hilbert_data": hilb}
    #only generate fig if necessary
    if config.show_figs or save:
        #3. plot
        fig, [ax, ax_phase, ax_freq] = plt.subplots(nrows=3, height_ratios=[3,1,1], sharex=True)
        figsize = fig.get_size_inches()
        fig.set_size_inches(figsize[0], 5/3*figsize[1])
        
        ax.plot(the_tlin, singlepoint, label="Data", color="b")
        ax_ylim=ax.get_ylim()
        ax.set_ylim(ax_ylim)
        ax.fill_between(the_tlin, lower_env, upper_env, alpha=0.5, facecolor="k", label="Env.")
        if show_analysis_range:
            t_range = _get_default_t_range(scan, config)
            ax.vlines(t_range, ymin=ax_ylim[0], ymax=ax_ylim[1], color="k", linestyle="--")
        
        ax_phase.plot(the_tlin, phase, color="tab:green")
        ax_freq.plot(the_tlin[((smoothing_n-1)//2):-((smoothing_n)//2)], averaged_freq, color="r")
        #4. format plot
        ax.legend()
        ax.set_xlim(the_tlin[0], the_tlin[-1]+scan.dt)
        if x is None: x = scan.nx // 2
        if y is None: y = scan.ny // 2
        ax.set_title(f"Hilbert transform for point\n({x},{y}) at ({round(scan.xlin[x])},{round(scan.ylin[y])}) mm")
        ax.set_xlabel(r"Time (μs)")
        ax.set_ylabel("Voltage (mV)")
        ax_phase.set_ylabel("phase (rad.)")
        ax_freq.set_ylabel("Inst. f (kHz)")
        if f_range is None: #default
            f_range=config.f_range
        ax_freq.set_ylim(f_range)
        #6. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"Ascan ({x},{y}) [{tstamp}].png"
        else:
            filename = None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists:
        _print_lists(return_data)
    return return_data

def sono(scan: Scan,
         config: Config,
         x: int | None = None,
         y: int | None = None,
         save: bool = False,
         crop_start_spike: bool = False,
         crop_end: bool = False,
         t_range: tuple[float, float] = (-np.inf, np.inf),
         f_range: tuple[float, float] | None = None, #uses config.f_range by default.
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sonogram for location (x,y) by index. Accepts custom t_range and f_range for cropping.
    The sonogram is generated using a short time Fourier transform,
    producing the spectral power density (SPD) in mV^2/kHz units.
    Returns dictionary containing f, t, SPD."""
    #1. select data
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    singlepoint = _select_spatial_data(the_vdata, 
                                       x, y)[0,:,0]
    #2. calculate spectrogram
    f, t, Sxx = _spectrogram_helper(config,
                                    singlepoint, the_tlin,
                                    f_range,
                                    nfft_factor = config.sono_nfft_factor)
    return_data={"f": f,
                 "t": t,
                 "SPD": Sxx}
    
    #only generate fig if necessary
    if config.show_figs or save:
        #3. plot
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(t, f, Sxx, cmap="plasma")
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        #4. format plot
        cbar.ax.set_xlabel("  mV$^2\\!$/kHz", labelpad=10)
        cbar.ax.xaxis.set_label_position("top")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Frequency (kHz)")
        if x is None: x = scan.nx // 2
        if y is None: y = scan.ny // 2
        ax.set_title(f"Sonogram for point\n({x},{y}) at ({round(scan.xlin[x])},{round(scan.ylin[y])}) mm")
        #5. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"Sonogram ({x},{y}) [{tstamp}].png"
        else:
            filename = None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists:
        _print_lists(return_data)
    return return_data

"""-------------------------- Multi point--------------------------"""
def singletime(scan: Scan,
               config: Config,
               t_index: int | None = None,
               save: bool = False,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Show spatial voltage distribution at a single time index.
    Returns dictionary containing t, xlin, vdata[:,t_index,:]."""
    #1. validate data
    if t_index is None: t_index = scan.nt // 2
    if not -scan.nt <= t_index < scan.nt:
        raise IndexError(
             f"t_index={t_index} outside valid range [{-scan.nt}, {scan.nt-1}]"
             )
    #2. select data
    t = round(scan.tlin[t_index], 3) #in μs, rounded
    single_data = scan.vdata[:,t_index,:]
    
    return_data = {"t": t,
                   "xlin": scan.xlin, "ylin": scan.ylin,
                   "vdata": single_data}
    
    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate singletime plot for a point scan.\n")
            return return_data
        #3. plot; format plot
        fig, ax = plt.subplots()
        if scan.is_x_scan:
            ax.plot(scan.xlin, single_data[0])
            ax.set_xlim(scan.xlin[0],scan.xlin[-1])
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("Voltage (mV)")
        elif scan.is_y_scan:
            ax.plot(scan.ylin, single_data[:,0])
            ax.set_xlim(scan.ylin[0],scan.ylin[-1])
            ax.set_xlabel("y position (mm)")
            ax.set_ylabel("Voltage (mV)")
        else: #scan.is_xy_scan
            mesh = ax.pcolormesh(scan.xlin, scan.ylin, single_data)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.ax.set_xlabel("  mV", labelpad=10)
            cbar.ax.xaxis.set_label_position("top")
            ax.invert_yaxis()
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("y position (mm)")
        ax.set_title(f"Spatial voltage distr., t={t} μs")
        #5. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"Singletime ({t}μs) [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists:
        _print_lists(return_data)
    return return_data

def max_amp(scan: Scan,
            config: Config,
            save: bool = False,
            crop_start_spike: bool = True,
            crop_end: bool = True,
            t_range: tuple[float, float] = (-np.inf, np.inf)
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Show the spatial distribution of the maximum (over time) voltage. Accepts a custom t_range.
    Returns dictionary containing t_range, xlin, ylin, max_data."""
    #1. select data
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    tlin_range = (float(the_tlin[0]), float(the_tlin[-1]))
    max_vdata = np.max(np.abs(the_vdata), axis=1)
    return_data = {"t_range": tlin_range,
                   "xlin": scan.xlin, "ylin": scan.ylin,
                   "max_vdata": max_vdata}
    
    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate max_amp plot for a point scan.\n")
            return return_data
        #2. plot; format plot
        fig, ax = plt.subplots()
        if scan.is_x_scan:
            ax.plot(scan.xlin, max_vdata[0])
            ax.set_xlim(scan.xlin[0],scan.xlin[-1])
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("Max. voltage (mV)")
            title = f"Maximum voltage across\ny={round(scan.ylin[0],3)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_y_scan:
            ax.plot(scan.ylin, max_vdata[:,0])
            ax.set_xlim(scan.ylin[0],scan.ylin[-1])
            ax.set_xlabel("y position (mm)")
            ax.set_ylabel("Max. voltage (mV)")
            title = f"Maximum voltage across\nx={round(scan.xlin[0],3)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_xy_scan:
            mesh = ax.pcolormesh(scan.xlin, scan.ylin, max_vdata)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.ax.set_xlabel("  mV", labelpad=10)
            cbar.ax.xaxis.set_label_position("top")
            ax.invert_yaxis()
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("y position (mm)")
            title = f"Maximum voltage,\nt$\\in${[round(t,3) for t in tlin_range]} μs"
        ax.set_title(title)
        #3. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"MaxAmp [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists:
        _print_lists(return_data)
    return return_data

#Note: could create max_intensity, which is just max_amp**2. This should be comparable in distribution to sono_amp and sono_int, although more noise sensitive.

def sono_amp(scan: Scan,
             config: Config,
             save: bool = False,
             crop_start_spike: bool = True,
             crop_end: bool = True,
             t_range: tuple[float, float] = (-np.inf, np.inf),
             f_range: tuple[float, float] | None = None,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot the spatial distribution of the maximum SPD value
    Returns dictionary containing t_range, xlin, ylin, peak SPD."""
    #1. select data
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    tlin_range = (float(the_tlin[0]), float(the_tlin[-1]))
    #2. calculate peak sonogram values for each x,y
    amp_peaks = np.empty((scan.ny, scan.nx))
    if (scan.is_xy_scan or scan.is_y_scan) and config.print_progress:
        print("sono_amp row: ", end="")
    elif scan.is_x_scan and config.print_progress:
        print("sono_amp col: ", end="")
    for yi in range(scan.ny):
        if (scan.is_xy_scan or scan.is_y_scan) and config.print_progress:
            print(f"{yi} ", end="")
        for xi in range(scan.nx):
            if scan.is_x_scan and config.print_progress:
                print(f"{xi} ", end="")
            singlepoint = the_vdata[yi,:,xi]
            #note: f, t unused
            f, t, Sxx = _spectrogram_helper(config,
                                            singlepoint, the_tlin,
                                            f_range,
                                            nfft_factor = config.sono_amp_nfft_factor)
            amp_peaks[yi,xi] = np.max(Sxx)
    return_data = {"t_range": tlin_range,
                   "xlin": scan.xlin, "ylin": scan.ylin,
                   "peak SPD": amp_peaks}
    if config.print_progress:
        print() #for nice formatting

    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate sono_amp plot for a point scan.\n")
            return return_data
        #3. plot; format plot
        fig, ax = plt.subplots()
        if scan.is_x_scan:
            amp_peaks = amp_peaks[0]
            ax.plot(scan.xlin, amp_peaks)
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel(f"Peak SPD (mV$^2$/kHz)")
            title = f"Peak spectral power density across\ny={round(scan.ylin[0],3)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_y_scan:
            amp_peaks = amp_peaks[:,0]
            ax.plot(scan.ylin, amp_peaks)
            ax.set_xlabel("y position (mm)")
            ax.set_ylabel(f"Peak SPD (mV$^2$/kHz)")
            title = f"Peak spectral power density across\nx={round(scan.xlin[0],3)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_xy_scan:
            ax.invert_yaxis()
            mesh = ax.pcolormesh(scan.xlin, scan.ylin, amp_peaks, cmap="plasma")
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.ax.set_xlabel("  mV$^2$/kHz", labelpad=10)
            cbar.ax.xaxis.set_label_position("top")
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("y position (mm)")
            title = f"Peak spectral power density,\nt$\\in${[round(t,3) for t in tlin_range]} μs"
        ax.set_title(title)
        #4. finalise plot
        if save:
            tstamp = _timestamp()
            filename = f"SonoAmp [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists: #if plot not generated, _finalise_plot isn't called
        _print_lists(return_data)
    return return_data

def sono_int(scan: Scan,
             config: Config,
             save: bool = False,
             crop_start_spike: bool = True,
             crop_end: bool = True,
             t_range: tuple[float, float] = (-np.inf, np.inf),
             f_range: tuple[float, float] | None = None,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot the spatial distribution of the maximum sonogram value integrated over frequency (units mV^2).
    The sonogram power has units mV^2.
    Returns dictionary containing t_range, f_range, xlin, ylin, peak SP"""
    #1. select data
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    tlin_range = (float(the_tlin[0]), float(the_tlin[-1]))
    #2. calculate the peak integrated sonogram value for each x,y
    int_peaks = np.empty((scan.ny, scan.nx))
    if (scan.is_xy_scan or scan.is_y_scan) and config.print_progress:
        print("sono_int row: ", end="")
    elif scan.is_x_scan and config.print_progress:
        print("sono_int col: ", end="")
    for yi in range(scan.ny):
        if (scan.is_xy_scan or scan.is_y_scan) and config.print_progress:
            print(f"{yi} ", end="")
        for xi in range(scan.nx):
            if scan.is_x_scan and config.print_progress:
                print(f"{xi} ", end="")
            singlepoint = the_vdata[yi,:,xi]
            f, t, Sxx = _spectrogram_helper(config,
                                            singlepoint, the_tlin,
                                            f_range,
                                            nfft_factor = config.sono_int_nfft_factor
                                            )
            df = f[1] - f[0] #kHz
            int_peaks[yi,xi] = np.max(np.sum(Sxx,axis=0)) * df #mV^2
    return_data = {"t_range": tlin_range,
                   "f_range": (f[0], f[-1]),
                   "xlin": scan.xlin, "ylin": scan.ylin,
                   "peak SP": int_peaks}
    if config.print_progress:
        print() #for nice formatting
        
    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate sono_int plot for a point scan.\n")
            return return_data
        #3. plot; format plot
        fig, ax = plt.subplots()
        if scan.is_x_scan:
            ax.plot(scan.xlin, int_peaks[0])
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("Peak power (mV$^2$)")
            title = f"Peak sonogram power across\ny={round(scan.ylin[0],1)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_y_scan:
            ax.plot(scan.ylin, int_peaks[:,0])
            ax.set_xlabel("y position (mm)")
            ax.set_ylabel("Peak power (mV$^2$)")
            title = f"Peak sonogram power across\nx={round(scan.xlin[0],1)} mm, t$\\in${[round(t,3) for t in tlin_range]} μs"
        elif scan.is_xy_scan:
            mesh = ax.pcolormesh(scan.xlin, scan.ylin, int_peaks, cmap="plasma")
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.ax.set_xlabel("  mV$^2$", labelpad=10)
            cbar.ax.xaxis.set_label_position("top")
            ax.invert_yaxis()
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("y position (mm)")
            title = f"Peak sonogram power,\nt$\\in${[round(t,3) for t in tlin_range]} μs"
        ax.set_title(title)
        #finalise
        if save:
            tstamp = _timestamp()
            filename = f"SonoInt [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    elif config.print_lists: #if plot not generated, _finalise_plot isn't called
        _print_lists(return_data)
    return return_data

def plot_xt(scan: Scan,
            config: Config,
            y: int | None = None,
            save: bool = False,
            crop_start_spike: bool = True,
            crop_end: bool = False,
            t_range: tuple[float, float] = (-np.inf, np.inf),
            show_analysis_range: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """plot the space-time voltage distribution for data over a single y position.
    Default y is centred.
    Returns dictionary containing t_samples, xlin, vdata_sampled."""
    if y is None: y = scan.ny // 2
    if not -scan.ny <= y < scan.ny:
        raise IndexError(
             f"y={y} outside valid range [{-scan.ny}, {scan.ny-1}]"
             )
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    t_indices = np.linspace(0, the_vdata.shape[1]-1, config.time_samples_1D, dtype=int)
    ts = the_tlin[t_indices]
    sampled_data = the_vdata[:,t_indices,:]
    
    return_data = {"t_samples": ts, "xlin": scan.xlin, 
                   "vdata_sampled": sampled_data}
    
    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate plot_xt plot for a point scan.\n")
            return return_data
        elif scan.is_y_scan:
            print("\nWarning: cannot generate plot_xt plot for a y-scan.\n")
            return return_data
        fig, ax = plt.subplots()
        ax.invert_yaxis()
        
        mesh = ax.pcolormesh(scan.xlin, ts, sampled_data[y], shading="nearest")

        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.ax.set_xlabel("  mV", labelpad=10)
        cbar.ax.xaxis.set_label_position("top")
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("time (μs)")
        ax.set_title("Spatiotemporal voltage distr.")
        if show_analysis_range:
            ax.hlines(_get_default_t_range(scan, config, crop_start_spike), xmin=scan.xlin[0], xmax=scan.xlin[-1], color="k", linestyle="--")
        if save:
            tstamp = _timestamp()
            filename = f"x-t [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    return return_data

def plot_yt(scan: Scan,
            config: Config,
            x: int | None = None,
            save: bool = False,
            crop_start_spike: bool = True,
            crop_end: bool = False,
            t_range: tuple[float, float] = (-np.inf, np.inf),
            show_analysis_range: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x is None: x = scan.nx // 2
    if not -scan.nx <= x < scan.nx:
        raise IndexError(
             f"x={x} outside valid range [{-scan.nx}, {scan.nx-1}]"
             )
    """plot the space-time voltage distribution for data over a single x position.
    Default x is centred.
    Returns dictionary containing t_samples, ylin, vdata_sampled."""
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    t_indices = np.linspace(0, the_vdata.shape[1]-1, config.time_samples_1D, dtype=int)
    ts = the_tlin[t_indices]
    sampled_data = the_vdata[:,t_indices,:]

    return_data = {"t_samples": ts, "xlin": scan.xlin, 
                   "vdata_sampled": sampled_data}
    
    #only generate fig if necessary
    if config.show_figs or save:
        if scan.is_point:
            print("\nWarning: cannot generate plot_yt plot for a point scan.\n")
            return return_data
        elif scan.is_x_scan:
            print("\nWarning: cannot generate plot_yt plot for an x-scan.\n")
        fig, ax = plt.subplots()
        ax.invert_yaxis()#TODO check maybe not
        
        plot_data = sampled_data[:,:,x]
        mesh = ax.pcolormesh(ts, scan.ylin, plot_data)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.ax.set_xlabel("  mV", labelpad=10)
        cbar.ax.xaxis.set_label_position("top")
        ax.set_ylabel("y position (mm)")
        ax.set_xlabel("time (μs)")
        ax.set_title("Spatiotemporal voltage distr.")
        if show_analysis_range:
            ax.vlines(_get_default_t_range(scan, config, crop_start_spike), ymin=scan.ylin[0], ymax=scan.ylin[-1], color="k", linestyle="--")
        if save:
            tstamp = _timestamp()
            filename = f"y-t [{tstamp}].png"
        else:
            filename=None
        _finalise_plot(fig, scan, config, save, filename, return_data)
    return return_data

def animate_xy(scan: Scan,
               config: Config,
               save: bool = False,
               crop_start_spike: bool = True,
               crop_end: bool = False,
               t_range: tuple[float, float] = (-np.inf, np.inf),
               ) -> tuple[FuncAnimation, np.ndarray, np.ndarray]:
    """[2D scan only]: animate the spatial voltage distribution over time
    Debug: you may have to assign the output of this function to a variable to prevent the animation closing automatically.
    Returns dictionary containing animation, t_samples, xlin, ylin, vdata_sampled."""
    #TODO: figure out why the animation is blocking; my MWE doesn't have issues
    the_vdata, the_tlin = _select_time_data(scan, config,
                                            crop_start_spike, crop_end,
                                            t_range)
    #animation update function
    def update_anim(frame):
        t_index = t_indices[frame]
        cur_t = the_tlin[t_index]
        ax.set_title(f"Spatial voltage distr.,\nt={cur_t:.3f} μs") #can't use blit because of this
        #change pcolormesh data
        singletime = sampled_data[:,frame,:]
        anim_mesh.set_array(singletime.ravel())
        return (anim_mesh,)
        
    t_indices = np.linspace(0, the_vdata.shape[1]-1, config.time_samples_anim, dtype=int)
    sampled_data = the_vdata[:,t_indices,:]
    vmin, vmax = np.min(sampled_data), np.max(sampled_data)
    fig, ax = plt.subplots()
    singletime = sampled_data[:,t_indices[0],:]
    anim_mesh = ax.pcolormesh(scan.xlin, scan.ylin, singletime, vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(anim_mesh, ax=ax, pad=0.01)
    cbar.ax.set_xlabel("  mV", labelpad=10)
    cbar.ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.set_xlabel("x position (mm)")
    ax.set_ylabel("y position (mm)")
    ax.set_title(f"Spatial voltage distr.,\nt={the_tlin[t_indices[0]]:.3f} μs")
    fig.tight_layout(pad=0.15)
    if config.print_progress:
        print("Loading animation...", end="")
    ani=FuncAnimation(fig, update_anim, frames=config.time_samples_anim, interval=1000/24)
    return_data = {"animation": ani,
                   "xlin": scan.xlin, "ylin": scan.ylin,
                   "t_samples": the_tlin[t_indices],
                   "vdata_sampled": sampled_data}
    if config.print_lists:
        _print_lists(return_data)
    #can't call _finalise_plot as we need ani.save, not fig.savefig
    if save:
        tstamp = _timestamp()
        filename = f"Anim {config.time_samples_anim} [{tstamp}].gif"
        loc = get_save_dir(scan, config) / filename
        loc.parent.mkdir(parents=True, exist_ok=True)
        ani.save(loc, writer=FFMpegWriter(fps=24))
    if config.show_figs:
        #plot reliably (taken from _finalise_plot)
        if not plt.isinteractive():
            #only necessary without plt.ion()
            plt.show(block=False)
        #these ensure the figure is actually rendered before the
        # next one starts loading, instead of just an empty window.
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        plt.close(fig)
    if config.print_progress:
        print("loaded. ")
    return return_data

def show_data(scan: Scan,
              config: Config,
              save: bool = False,
              crop_start_spike: bool = True,
              crop_end: bool = False,
              t_range: tuple[float, float] = (-np.inf, np.inf),
              show_analysis_range: bool = False
              ):
    """Present the data in full (although temporally subsampled for 1D or 2D spatial data) in a relevant format.
    Note the format of the return data will depend on the shape of the scan.
    For 0D (singlepoint) data, generate an A-scan.
    For 1D (x-scan, y-scan) data, generate an x-t or y-t plot.
    For 2D data, generate an animation.
    Returns return data of relevant plotting function."""
    if scan.is_point:
        out = ascan(scan, config, 0, 0, save, crop_start_spike, crop_end, t_range, show_analysis_range)        
    elif scan.is_x_scan:
        out = plot_xt(scan, config, save, crop_start_spike, crop_end, t_range, show_analysis_range)
    elif scan.is_y_scan:
        out = plot_yt(scan, config, save, crop_start_spike, crop_end, t_range, show_analysis_range)
    else:
        out = animate_xy(scan, config, save, crop_start_spike, crop_end, t_range)
    return out

"""---------------------Non-plotting functions---------------------"""

def moving_average(a: np.ndarray,
                    n: int
                    ) -> np.ndarray:
    """Small helper function to calculate a moving average.
    Used to smooth instantaneous frequency from Hilbert transform."""
    a_cum = np.cumsum(a)
    a_cum[n:] = a_cum[n:] - a_cum[:-n] #find sum in moving window; modify in place
    return a_cum[n - 1:] / n #divide by n to give windowed mean

def res(trace: np.ndarray
        ) -> tuple[float, float]:
    """[not used currently] find minimum and maximum differences in sorted voltage values.
    With enough data points, min_diff should be the instrument resolution.
    Importantly, this should not be confused with precision."""

    #find diffs between data points
    diff=np.diff(np.sort(trace))
    #remove zeroes
    diffpos=diff[diff>0]
    min_diff=np.min(diffpos)
    max_diff=np.max(diffpos)
    return (min_diff,max_diff)

def FWHM(xlin, data, plot=False):
    """Helper function to find row-wise full width at half-maximum (FWHM) of data,
    linearly interpolating between points. This is a simple algorithm, not designed
    to handle noisy data. In my work, fits were passed to avoid this problem.
    Returns an array containing [left HM x pos, peak xpos, right HM x pos].
    """
    #check if data is 1D, if so simply reshape it to 2D so the for loop works
    if len(data.shape)==1:
        data = data.reshape((1,data.shape[0]))
    #iterate through each row
    widths = np.empty((data.shape[0],3))
    for row in range(data.shape[0]):
        data_row=data[row]

        #maximum value
        M = np.max(data_row)
        #corresponding x value
        Mx = xlin[np.where(data_row==M)][0]

        #half-maximum value
        HM = M/2
        #find all values above HM
        datashift = data_row - HM
        pos = np.where(datashift > 0)[0]
        #find the indices sandwiching the left HM.
        # If data does not fall below HM, left_pos = 0 and left_neg = -1
        left_pos = pos[0]
        left_neg = left_pos-1
        #find the indices sandwiching the right HM (see above)
        right_pos = pos[-1]
        right_neg = right_pos+1

        #coordinates of left points
        try:
            #x values
            lnx = xlin[left_neg]
            lpx = xlin[left_pos]
            #data (y) values, with HM subtracted
            lny = datashift[left_neg]
            lpy = datashift[left_pos]
        except IndexError:
            [lnx, lpx, lny, lpy] = [xlin[0]-1, xlin[0], -1, 1]
            print(f"Error with left HM at line {row}")
        #coordinates of right points
        try:
            unx = xlin[right_neg]
            upx = xlin[right_pos]
            uny = datashift[right_neg]
            upy = datashift[right_pos]
        except IndexError:
            [unx, upx, uny, upy] = [xlin[-1], xlin[-1]+1, -1, 1]
            print(f"Error with right HM at line {row}")
        #interpolate to find left HM
        left_Weight = lny/(lny-lpy)
        left_x = left_Weight*lpx + (1-left_Weight)*lnx
        #interpolate to find right HM
        right_Weight = uny/(uny-upy)
        right_x = right_Weight*upx + (1-right_Weight)*unx
        #format return data
        width = np.array([left_x,Mx,right_x])
        widths[row] = width
        
        #debug plotting
        if plot:
            fig, ax = plt.subplots()
            ax.plot(xlin, datashift)
            ax.scatter(lpx,lpy,c="b")
            ax.scatter(lnx,lny,c="b")
            ax.scatter(upx,upy,c="orange")
            ax.scatter(unx,uny,c="orange")
            
            ax.hlines(0,xmin=xlin[0], xmax=xlin[-1],
                      color="k", linestyle="--")
            ax.vlines([left_x,right_x], ymin=-HM, ymax=HM,
                      color="k", linestyle="--")
            #plot reliably (taken from _finalise_plot)
            if not plt.isinteractive():
                #only necessary without plt.ion()
                plt.show(block=False)
            #these ensure the figure is actually rendered before the
            # next one starts loading, instead of just an empty window.
            fig.canvas.draw()
            fig.canvas.flush_events()
    return widths

def halfsine(x, A, w, x0, c):
    x = np.asarray(x)
    
    # condition: -w/2 < x - x0 < w/2
    mask = (x - x0 > -w/2) & (x - x0 < w/2)
    
    result = A * np.cos((np.pi / w) * (x - x0)) * mask + c
    
    return result

def fit_halfsine(t, data):
    t = np.asarray(t)
    data = np.asarray(data)
    
    # Initial parameter guesses: A, w, x0, c
    A0 = np.max(data) - np.min(data)
    c0 = np.min(data)
    x0_0 = t[np.argmax(data)]
    w0 = (t.max() - t.min()) / 4  # rough guess
    
    p0 = [A0, w0, x0_0, c0]
    
    popt, pcov = curve_fit(halfsine, t, data, p0=p0)
    
    return popt, pcov
