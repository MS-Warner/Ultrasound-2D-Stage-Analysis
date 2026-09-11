import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from dataclasses import dataclass
#handle imports correctly when run as part of the library or standalone
if __package__:
    from .data import Scan, Config
else:
    from data import Scan, Config

@dataclass(slots=True, frozen=True)
class SimGrid:
    """Stores details of the lattice over which to simulate"""
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    t_range: tuple[float, float]

    nx: int
    ny: int
    nt: int
    
    @property
    def dx(self):
        return (self.x_range[1] - self.x_range[0]) / (self.nx - 1)
    @property
    def dy(self):
        return (self.y_range[1] - self.y_range[0]) / (self.ny - 1)
    @property
    def dt(self):
        return (self.t_range[1] - self.t_range[0]) / (self.nt - 1)
    
    @property
    def xlin(self):
        return np.linspace(*self.x_range, self.nx)
    @property
    def ylin(self):
        return np.linspace(*self.y_range, self.ny)
    @property
    def tlin(self):
        return np.linspace(*self.t_range, self.nt)

@dataclass(slots=True)
class Wave:
    """Stores details about the wave used by a particular source.
    Expected to be shared by all sources for a given generator, if not all sources."""
    speed:       float #mm/μs
    wavelength:  float #mm
    n_cycles:    float
    phase_shift: float #rad
    min_dist:    float = 10 #mm
    @property
    def max_dist(self) -> float:
        return self.min_dist + self.n_cycles * self.wavelength
    @property
    def frequency(self) -> float:
        return self.speed / self.wavelength

@dataclass(slots=True, frozen=True)
class Source:
    """Stores details about a single point source"""
    wave: Wave
    x: float #mm
    y: float #mm
    A: float #mV
    delay: float #μs #should generally be zero, but can be modified while keeping a shared Wave instance across sources for some phasing applications.

@dataclass(slots=True)
class Sim:
    """Bundles all the simulation stuff together"""
    grid: SimGrid
    sources: list[Source]
    use_simpler_wave: bool = False #removes signal envelope and attenuation if True; useful for rapid determination of focal spots etc.

def line(start: tuple[float, float],
         end:   tuple[float, float],
         n_sources: int,
         wave: Wave,
         delays: float | np.ndarray = 0,       #μs. If ndarray, shape must be (n_sources,)
         amplitudes: float | np.ndarray = 1.0, #mV. If ndarray, shape must be (n_sources,)
         mirror_x: bool = False) -> list[Source]:
    """Return an array containing a line of point sources, sharing a common wave.
    delays and amplitudes can either be common or individually set.
    Used to model generators."""
    #Validate delays array
    if np.isscalar(delays):
        dels = np.full(n_sources, delays)
    else:
        dels = np.asarray(delays)
        if dels.shape != (n_sources,):
            raise ValueError(
                f"delays shape {dels.shape} must match n_sources ({n_sources},)."
                )
    #Validay amplitudes array
    if np.isscalar(amplitudes):
        amps = np.full(n_sources, amplitudes)
    else:
        amps = np.asarray(amplitudes)
        if amps.shape != (n_sources,):
            raise ValueError(
                f"amplitudes shape {amps.shape} must match n_sources ({n_sources},)."
                )
    xs = np.linspace(start[0], end[0], n_sources, endpoint=True, dtype=float)
    ys = np.linspace(start[1], end[1], n_sources, endpoint=True, dtype=float)
    
    sources=[]
    for i in range(n_sources):
        x = xs[i]
        y = ys[i]
        A = amps[i]
        delay = dels[i]
        sources.append(Source(wave,
                              x, y,
                              A,
                              delay,
                              )
                       )
        if mirror_x:
            sources.append(Source(wave,
                                  -x, y,
                                  A,
                                  delay,
                                  )
                           )
    return sources

def show_sources(sim: Sim):
    """Plot source distribution relative to simulation grid."""
    sources = sim.sources
    xs = np.empty(len(sources), float)
    ys = np.empty(len(sources), float)
    for i, source in enumerate(sources):
        xs[i] = source.x
        ys[i] = source.y
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, label="Sources")
    
    grid_xmin, grid_xmax = sim.grid.x_range
    grid_ymin, grid_ymax = sim.grid.y_range
    rect = Rectangle((grid_xmin, grid_ymin),
                     grid_xmax - grid_xmin,
                     grid_ymax - grid_ymin,
                     fill = True,
                     color="black",
                     alpha=0.5,
                     label="SimGrid")
    ax.add_patch(rect)
    ax.legend(borderaxespad=0.15,
              borderpad=0.25,
              handlelength=1,
              handletextpad=0.2,
              labelspacing=0.2
              )
    ax.invert_yaxis()
    plt.show()
    return xs, ys

def envelope(A, r_masked, rmin, k_env):
    """Apply a half-sine shaped envelope"""
    return A * np.sin(k_env*(r_masked-rmin))

@dataclass(slots=True)
class SourceCache:
    """Store expensive or frequently used derived values.
    For optimisation purposes only."""
    r: np.ndarray
    sqrt_r: np.ndarray
    k: float
    k_env: float
    omega: float
    
def calculate_source_frame(sim,source,source_cache,t_index):
    """Calculate the amplitude from a single source at a single point in time.
    Note that vectorising over time was not found to improve performance."""
    #define shorthand variables to make the maths actually readable
    t    = sim.grid.tlin[t_index]
    l, c = source.wave.wavelength,  source.wave.speed
    min_dist, n_cycles = source.wave.min_dist, source.wave.n_cycles
    r, sqrt_r          = source_cache.r, source_cache.sqrt_r
    k, omega, phase    = source_cache.k, source_cache.omega, source.wave.phase_shift
    
    #create mask to only calculate elements on the wavefront
    wave_age = t - source.delay
    rmin = max(c * wave_age, min_dist)
    rmax =     c * wave_age + n_cycles * l
    r_mask = (rmin < r ) & (r < rmax) #subset of lattice points to modify
    r_masked = r[r_mask]

    carrier = np.sin(k * r_masked - omega * wave_age - phase)
    if sim.use_simpler_wave:
        packet = source.A #apply no envelope or attenuation
    else:
        #pulse envelope with geometric attenuation
        packet = (envelope(source.A, #envelope, currently hardcoded as half-sine
                           r_masked,
                           rmin,
                           source_cache.k_env) 
                  / sqrt_r[r_mask] #geometric attenuation conserving energy
                  ) 
    values = packet * carrier
    return (values, r_mask)

def simulate(sim: Sim,
             name: Path | str | None = None
             ) -> Scan:
    """Perform simulation by calling calculate_source_frame for each source and time"""
    vdata = np.zeros((sim.grid.ny, sim.grid.nt, sim.grid.nx))
    x,y=np.meshgrid(sim.grid.xlin,sim.grid.ylin)
    for s in sim.sources:
        tlin = sim.grid.tlin #fetch once; uses a shorter reference for readability
        r = np.sqrt((x-s.x)**2+(y-s.y)**2) #calculate r(x,y) for this source
        source_cache = SourceCache(r = r,
                                   sqrt_r = np.sqrt(r),
                                   k = 2*np.pi/s.wave.wavelength,
                                   k_env = np.pi / (s.wave.n_cycles * s.wave.wavelength), #note absent factor of 2; env is half-wave
                                   omega = 2*np.pi*s.wave.frequency,
                                   )
        for t_index, t in enumerate(tlin): #note: vectorising over time did not improve performance
            values, r_mask = calculate_source_frame(sim, s, source_cache, t_index)
            frame = vdata[:, t_index, :]
            frame[r_mask] += values
        
    return Scan(vdata = vdata,
                xlin = sim.grid.xlin,
                ylin = sim.grid.ylin,
                dt = sim.grid.dt,
                has_generation_spike = False,
                post_clip_delay = None, #value unneeded as there is no spike
                folder = None,
                name = name,
                )
