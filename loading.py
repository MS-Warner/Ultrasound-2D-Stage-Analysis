import numpy as np
import re
from pathlib import Path
#handle imports correctly when run as part of the library or standalone
if __package__:
    from .lvm2npz import lvm2npz
    from .data import Scan, Config
else:
    from lvm2npz import lvm2npz
    from data import Scan, Config


#error class for loading scan
class ScanLoadError(Exception):
    """Raised when scan data cannot be located or loaded."""

def _create_scan(raw_data: tuple[np.ndarray, np.ndarray, np.ndarray, float],
                source_dir: Path,
                config: Config,
                ) -> Scan:
    """Instantiate object from Scan class to store details and data from a scan."""
    vdata, xlin, ylin, dt = raw_data
    scan = Scan(
        vdata  = vdata * 1e3, #convert to mV
        xlin   = xlin  + config.shift_x,
        ylin   = ylin  + config.shift_y,
        dt     = dt    * 1e6, #convert to μs
        has_generation_spike = True,
        post_clip_delay = 8,
        folder = source_dir.parent,
        name   = source_dir.name
        )
    return scan

def get_save_dir(scan: Scan, config: Config) -> Path:
    """Return save Path, which separates scans by dimension."""
    if scan.name is None:
        print("Warning: this scan has no name. Plots will be saved under \"!noname\".")
        name = "!noname"
    else:
        #remove file extension
        name = Path(scan.name).stem
    
    #set save location based on scan dimensions
    if scan.is_point:
        return config.save_root / "0D scan (singlepoint)" / name
    elif scan.is_x_scan:
        return config.save_root / "1D scan" / name
    elif scan.is_y_scan:
        return config.save_root / "1D scan" / "y" / name
    elif scan.is_xy_scan:
        return config.save_root / "2D scan" / name
    raise ValueError(f"Invalid scan shape: {scan.vdata.shape}")

#check if array is either stricly increasing or strictly decreasing
def is_strictly_monotonic(arr: np.ndarray) -> bool:
    return np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)

def load_npz(filename: Path,
             config: Config
             ) -> Scan:
    """Produce scan from directory containing .lvm file(s)"""
    print("  Loading data... ")
    try:
        with np.load(filename) as data:
            if "vdata" in data:
                vdata = data["vdata"]
            else:
                vdata = data["alldata"]
            raw_data = (
                vdata,
                data["xlin"],
                data["ylin"],
                data["dt"],
                )
    except (KeyError, ValueError, OSError) as e:
        raise ScanLoadError(
            f"Could not load data from {filename}."
            ) from e
    scan = _create_scan(raw_data, filename, config)
    print(f"Loaded ((ny,nt,nx)={scan.vdata.shape}).")
    return scan

def load_lvm(folder: Path,
             config: Config
             ) -> Scan:
    """Produce scan from .npz file in accepted format.
    See lvm2npz for file format documentation."""
    print("  Loading data... ")
    #convert data (choosing not to handle error here if it fails)
    try:
        data = lvm2npz(folder, config.npz_dir)
    except Exception as e:
        raise ScanLoadError(
            f"Failed to load data from {folder}."
            ) from e
    #(else)
    scan = _create_scan(data, folder, config)
    print(f"Loaded ((ny,nt,nx)={scan.vdata.shape}).")
    return scan

def _search_directory(directory: Path,
                      scan_id: int| str
                      ) -> Path | None:
    """Locate file/folder in directory matching scan_id.
    If scan_id is int, search for number at start of each filename until a match is found.
    If scan_id is str, search for substring in each filename until a match is found."""
    filenames = sorted(directory.iterdir())

    if isinstance(scan_id, int):
        for filepath in filenames:
            #extract leading number
            match = re.match(r'^(\d+)(?!\d)', filepath.name)
            if match is None:
                continue
            #(else)
            
            filenum=int(match.group())
            #check if file is correct
            if scan_id == filenum:
                return filepath
    elif isinstance(scan_id, str):
        for filepath in filenames:
            if scan_id in filepath.name:
                return filepath
    return None


def load_scan(scan_id: int | str,
              config: Config
              ) -> Scan:
    """Instantiate and return a Scan object given only a scan_id and relevant directories to search."""
    print(f"\nscan_id: {scan_id}")
    print(f"  Searching {config.npz_dir}...")
    
    npz_path = _search_directory(config.npz_dir, scan_id)
    
    #if npz file exists
    if npz_path is not None:
        print(f"    Found {npz_path}.")
        scan = load_npz(npz_path, config)
        
        if not is_strictly_monotonic(scan.ylin):
            #Useful diagnostic information, particularly for scans with some issues
            print("Warning: ylin is not strictly monotonic, which will likely affect spatial plots.")
            print("This can happen when .lvm files from multiple scans are placed in the same folder.")
        return scan
    #(else)
    print(f"    No npz file found.")
    print(f"  Searching {config.lvm_dir}... ")
    lvm_path = _search_directory(config.lvm_dir, scan_id)
    #if data folder exists
    if lvm_path is None:
        raise ScanLoadError(
            f"No npz or lvm data found matching scan_id={scan_id}."
            )
    #(else)
    print(f"    Found {lvm_path}.")
    scan = load_lvm(lvm_path, config)
    
    if not is_strictly_monotonic(scan.ylin):
        print("Warning: ylin is not strictly monotonic, which will likely affect spatial plots.")
        print("This can happen when .lvm files from multiple scans are placed in the same folder.")
    return scan
