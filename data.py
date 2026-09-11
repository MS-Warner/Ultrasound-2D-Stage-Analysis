import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True)
class Config:
    """
    Store variables relating to analysis, saving, and output.
    """
    #analysis config
    analysis_duration: float     #μs
    f_range: tuple[float, float] #kHz
    
    #sonogram config
    sono_win_time: float          #μs
    sono_overlap_time: float      #μs
    #Scale factors for sonogram nfft in various functions.
    # Higher values increase frequence resolution.
    sono_nfft_factor: float #
    sono_amp_nfft_factor: float #values >1 are likely slow, and probably not of much use. YMMV
    sono_int_nfft_factor: float #values >1 are likely slow and pointless, as frequency is integrated over

    #save config
    lvm_dir: Path
    npz_dir: Path
    save_root: Path

    #coordinate corrections
    shift_x: float = 0
    shift_y: float = 0
    
    #plot config
    dpi: float = 300
    show_figs: bool = True
    print_progress: bool = False
    show_scan_name: bool = False
    print_lists: bool = False
    time_samples_anim: int = 201
    time_samples_1D: int = 1001
    
    @property
    def sono_winstep(self) -> float:
        if self.sono_overlap_time >= self.sono_win_time:
            raise ValueError(
                "sono_win_time must exceed sono_overlap_time. "
                )
        return self.sono_win_time - self.sono_overlap_time

@dataclass(slots=True)
class Scan:
    """
    Store processed scan data.
    Raw files are stored in (V,mm,S) units and are converted to display units (mV, mm, μs) upon loading.
    """
    vdata: np.ndarray      #mV
    xlin: np.ndarray       #mm
    ylin: np.ndarray       #mm
    dt: float              #μs

    #if has_generation_spike==False, all references to *_post_spike, crop_start_spike, etc.
    # will have no cropping applied, as this variable indicates that
    # there is nothing to crop. See clip_end_index for more.
    has_generation_spike: bool
    post_clip_delay: float | None #μs
    
    #metadata
    folder: Path | None
    name: str | None
    #save_dir: Path | None = None
    
    @property
    def fileloc(self) -> Path | None:
        if self.folder is None or self.name is None:
            return None
        return self.folder / self.name

    def __post_init__(self):
        if self.has_generation_spike and self.post_clip_delay is None:
            raise ValueError(
                "post_clip_delay must be set when has_generation_spike=True."
            )
        #convert metadata to Path objects 
        if self.folder is not None:
            self.folder = Path(self.folder)
        if self.name is not None:
            self.name = Path(self.name)

    @property
    def ny(self) -> int:
        return self.vdata.shape[0]
    @property
    def nt(self) -> int:
        return self.vdata.shape[1]
    @property
    def nx(self) -> int:
        return self.vdata.shape[2]

    #tlin in range [0,(nt-1)*dt]
    @property
    def tlin(self) -> np.ndarray:
        return np.arange(self.nt) * self.dt

    #check data shape in human-readable terms
    @property
    def is_point(self) -> bool:
        return self.nx == 1 and self.ny == 1
    @property
    def is_x_scan(self) -> bool:
        return self.nx >  1 and self.ny == 1
    @property
    def is_y_scan(self) -> bool:
        return self.nx == 1 and self.ny >  1
    @property
    def is_xy_scan(self) -> bool:
        return self.nx >  1 and self.ny >  1

    #Locate end of clipped data    
    _clip_end_index: int | None = field(init=False, default=None)
    @property
    def clip_end_index(self) -> int:
        """
        Attempt to automatically identify generation noise, to allow removal from analysis.
        Real data from project starts with generation spike, which is clipped to a maximum/minimum value by the LabVIEW program.
        The two extremal values may not have the same magnitude, so maximum and minimum are handled separately to find the last clipped value.
        Data without clipping should be marked as such with has_generation_spike=False, in which case this function should not be called.
        If has_generation_spike==False, references to all *post_spike* variables will safely not call this method.
        """
        if not self.has_generation_spike:
            raise ValueError(
                "clip_end_index is meaningless when has_generation_spike=False"
                )
        if self._clip_end_index is None:
            min_data_t = np.min(self.vdata, axis=(0,2))
            max_data_t = np.max(self.vdata, axis=(0,2))
            #locate the final indices clipped negative or positive
            clip_min_end_index = (self.nt-1) - np.argmin(min_data_t[::-1])
            clip_max_end_index = (self.nt-1) - np.argmax(max_data_t[::-1])
            #Does not exclude ringdown.
            self._clip_end_index = max(clip_min_end_index, clip_max_end_index)
        return self._clip_end_index
    
    @property
    def post_spike_start_index(self) -> int:
        """Find first index after generation noise judged to be usable."""
        if not self.has_generation_spike:
            return 0
        if self.post_clip_delay is None:
            raise ValueError(
                "post_clip_delay must be set when has_generation_noise=True"
                )
        index = self.clip_end_index + 1 + round(self.post_clip_delay / self.dt)
        if index >= self.nt:
            raise ValueError(
                f"post_clip_delay={self.post_clip_delay} μs is too large or clip detection failed: "
                f"clip_end_index={self.clip_end_index}, dt={self.dt} μs, "
                f"giving post_spike_start_index={index} >= nt={self.nt}."
            )
        return index
    @property
    def post_spike_start_time(self) -> float:
        return self.tlin[self.post_spike_start_index]
    @property
    def vdata_post_spike(self) -> np.ndarray:
        """Return view of vdata with generation noise removed."""
        return self.vdata[:, self.post_spike_start_index:, :]
    @property
    def tlin_post_spike(self) -> np.ndarray:
        """Return view of tlin with generation noise removed."""
        return self.tlin[self.post_spike_start_index:]
