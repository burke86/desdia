import os, subprocess
import numpy as np
from astropy import units as u
from astropy.io import fits
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def get_central_pointing(ra, dec):
    pointing_coords = np.array([ra, dec]).T
    # Find the central (median) pointing
    pointing_coord_med = np.median(pointing_coords, axis=0)
    print(pointing_coord_med)
    # Find nearest pointing to median
    idx = np.abs(pointing_coords - pointing_coord_med).argmin()
    return idx, pointing_coords[idx]

def single_header(filename):
    hdul = fits.open(filename,ignore_missing_end=True)
    hdu = fits.PrimaryHDU(data=hdul[1].data, header=hdul[1].header)
    hdu.writeto(filename,clobber=True,output_verify='ignore')
    return

def bash(command,print_out=True):
    if print_out: print(command)
    try: return subprocess.call(command.split())
    except: return -1

def clean_info(info_list):
    info_list_clean = [i for i in info_list if i is not None]
    return np.array(info_list_clean)

def clean_pool(pool_func, arg_tuple, num_threads):
    if num_threads == 1:
        out = [pool_func(arg) for arg in arg_tuple]
    else:
        pool = Pool(num_threads)
        out = pool.map(pool_func, arg_tuple)
        pool.close()
        pool.join()
    return clean_info(out)

def clean_tpool(pool_func, arg_tuple, num_threads):
    if num_threads == 1:
        out = [pool_func(arg) for arg in arg_tuple]
    else:
        pool = ThreadPool(num_threads)
        out = pool.map(pool_func, arg_tuple)
        pool.close()
        pool.join()
    return clean_info(out)

def safe_rm(file_path, debug_mode=False):
    if debug_mode: return
    try: os.remove(file_path)
    except: return
