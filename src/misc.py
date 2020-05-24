import os, subprocess
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from multiprocessing import Pool

def toIAU(ra, dec):
    c = SkyCoord(ra*u.degree, dec*u.degree)
    name = 'J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True), c.dec.to_string(sep='', precision=2, alwayssign=True, pad=True))
    return name

def fromIAU(name):
    name = name.strip()
    ra = name.split('J')[1].split('+')[0].split('-')[0]
    dec = name.split('J'+ra)[1]
    ra = "%sh%sm%ss" % (ra[0:2], ra[2:4], ra[4:])
    dec = "%sd%sm%ss" % (dec[0:3], dec[3:5], dec[5:])
    c = SkyCoord('%s %s' % (ra, dec))
    ra = c.ra.degree
    dec = c.dec.degree
    return ra, dec

def single_header(filename):
    hdul = fits.open(filename,ignore_missing_end=True)
    hdu = fits.PrimaryHDU(data=hdul[1].data,header=hdul[1].header)
    hdu.writeto(filename,clobber=True)
    return

def bash(command,print_out=True):
    if print_out: print(command)
    try: return subprocess.call(command.split())
    except: return -1

def clean_info(info_list):
    info_list_clean = [i for i in info_list if i is not None]
    return np.array(info_list_clean)

def clean_pool(pool_func, arg_tuple, num_threads):
    pool = Pool(num_threads)
    out = pool.map(pool_func, arg_tuple)
    pool.close()
    pool.join()
    return clean_info(out)

def safe_rm(file_path):
    try: os.remove(file_path)
    except: pass
