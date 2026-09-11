import numpy as np
from pathlib import Path

class LVMReadError(Exception):
    """Raised when an LVM scan cannot be parsed or converted."""
    pass

def _read_lvm_folder(scanfolder: Path
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Given the path to a folder containing .lvm files, process into single array vdata[y,t,x] (and xlin, ylin, dt)
    Assumes data are obtained on a rectangular grid.
    Note: to accept data on a non rectangular lattice, a significant rewrite would be required to use an array of x and y values, instead of the current approach of using independent coordinates and constructing the grid as required. Interactions with plt.pcolormesh would likely be more annoying, but to my knowledge this should not be an issue."""

    #print(f"Folder: {scanfolder}") #old debug, might be useful when tracking program flow
    scanfiles = sorted(scanfolder.glob("*.lvm"))
    if not scanfiles:
        raise FileNotFoundError(
            f"No .lvm files found in {scanfolder}"
            )
    try:
        trial_data = np.loadtxt(scanfiles[0], encoding="utf8")
    except (OSError, ValueError) as e:
        raise LVMReadError(
            f"Could not read {scanfiles[0]}"
            ) from e
    dt = trial_data[2,1] #find dt
    #find data shape
    ny = len(scanfiles)
    nt = trial_data.shape[0]-3 #(account for metadata rows)
    nx = trial_data.shape[1]-1 #(account for time column)

    xlin = trial_data[0,1:]
    ylin = np.empty((ny))
    
    #initialise array of correct size
    vdata = np.empty((ny, nt, nx))
    #loop through all files, filling vdata
    for i in range(len(scanfiles)):
        scanfile = scanfiles[i]
        print(f"    Reading {scanfile}...")
        try:
            #read data from file
            raw_data = np.loadtxt(scanfile, encoding="utf8")
        except (OSError, ValueError) as e:
            raise LVMReadError(
                f"Could not read {scanfile}"
                ) from e
        if raw_data.shape != trial_data.shape:
            raise LVMReadError(
                f"Inconsistent shape in {scanfile}: "
                f"{raw_data.shape} != {trial_data.shape}"
                )
        #remove time column
        xyv_values = raw_data[:,1:]
        
        ylin[i] = xyv_values[1,0]
        #save without x,y,dt rows
        vdata[i,:,:] = xyv_values[3:,:]
    print("  All files read.")
    return (vdata, xlin, ylin, dt)

#function to import
def lvm2npz(lvm_folder: Path,
            npz_folder: Path | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load .lvm data from lvm_folder, save it as an .npz file in npz_folder (if given), and return the data."""
    (vdata, xlin, ylin, dt) = _read_lvm_folder(lvm_folder)
    if npz_folder is not None:
        #take lvm folder name as npz file name
        npz_name = lvm_folder.name + ".npz"
        outfile  = npz_folder / npz_name
        #write data to file
        print(f"  Writing data to {outfile}\n    (This will override any existing data)...")
        np.savez(outfile, vdata=vdata, xlin=xlin, ylin=ylin, dt=dt)
    else:
        print("Not saving data, as no npz_folder supplied.")
    #return output for use without requiring additional read
    return (vdata, xlin, ylin, dt)
