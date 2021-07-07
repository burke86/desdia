# For main function input, see line 720

# Last modified 7/7/2021
import os
import glob
import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.table import vstack, Table
from astropy.visualization import astropy_mpl_style, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling.models import Gaussian2D
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import DAOStarFinder, CircularAperture, aperture_photometry, SkyCircularAperture
from scipy.stats import norm, halfnorm
import warnings
from humanize import naturalsize

#ignore by message
warnings.filterwarnings("ignore", message="The kernel is not normalized")
warnings.filterwarnings("ignore", message="UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")

plt.style.use('seaborn-whitegrid')

#################################################################################################
# FUNCTIONS
# Check for variability: 
def myvarlc(fluxin,sigin):
    n = np.shape(fluxin)[0]
    #convergence threshold    tiny = 2.e-5
    tiny = 2.e-5
    #lower limit for extra variance
    #vmin = tiny * v0
    #trap no data
    good = 0.
    avg = 0.
    rms = 0.
    sigavg = -1.
    sigrms = -1.
    
    if( n <= 0 ):
        print('flux,sig arrays empty, no data',n)
        return(good, avg, rms, sigavg, sigrms)
    
    #only use data with non zero error bar
    idinc = np.where(sigin > 0)[0]
    flux = fluxin[idinc]
    flux2 = flux * flux
    sig  = sigin[idinc]
    sig2 = sig*sig
    ng = np.shape(sig)[0]
    
    #trap no valid data
    if( ng <= 0 ):
        print('** ERROR in avgrmsx. n', n, ' ng', ng)
        return(good, avg, rms, sigavg, sigrms)
    
    #average positive error bars
    e1 = np.mean(sig)
    #KDH : V1 MAY VANISH IF E1 TOO SMALL
    v1 = e1 * e1
    #initial estimate
    x = e1/sig
    x2 = x*x
    sum1 = np.sum(flux*x2)
    sum2 = np.sum(x2)
    varguess = np.std(flux) 
    #optimal average and its variance
    avg = sum1 / sum2
    sigavg = e1 / np.sqrt( sum2 )
    v0 = e1 * ( sigavg / sum2 )
    #scale factor ( makes <1/sig^2>=1 )
    v1 = ng * e1 * ( e1 / sum2 )
    e1 = np.sqrt( v1 )
    v0 = v0 / v1
    #convergence threshold
    tiny = 2.e-5
    #lower limit for extra variance
    vmin = tiny * v0
    #initial parameter guesses
    #avg = 10.0#np.mean(flux)
    #rms = 0.1#np.std(flux)
    var = rms*rms
    #max-likelihood estimate of mean and extra varian
    nloop = 1000
    for loop in range(nloop):
        #stow for convergence test
        oldavg = avg
        oldrms = rms
        oldvar = var
        #rescale
        d = flux/e1
        d2=d*d
        e = sig/e1
        #e2 = e*e
        a = avg/e1
        #weight
        w = 1./(v0 + e*e)
        #"goodness" of the data 
        g = v0 * w
        x = g * (d - a)
        xx = x*x
        sum1 = np.sum(g*d)
        sumg = np.sum(g)
        sum  = np.sum(g*g)
        sum2 = np.sum(xx)
        sum3 = np.sum(g*xx)  
        sum4 = np.sum(x*g)
        a = sum1 / sumg
        v0 = sum2 / sumg
        v0 = max( vmin, v0 )
        va = v0 / sumg
        #new avg and rms
        avg = a
        rms = np.sqrt( v0 )
        #hessian matrix
        Hmm = sumg/v0
        Hmv = sum4/v0/v0
        Hvv = sum3/v0/v0/v0 - 0.5*sum/v0/v0
        #correction for covariance
        c = 1. - Hmv*Hmv/Hmm/Hvv
        #error bars on avg and rms
        sigavg = np.sqrt( 1./Hmm/c )#np.sqrt( va )
        if ((1./Hvv/c) < 0.0): 
            print('Negative argument in sigv0')
            good = nan
            avg = nan
            rms = nan
            sigavg = nan
            sigrms = nan
            return good, avg, rms, sigavg, sigrms
        sigv0  = np.sqrt( 1./Hvv/c )
        #g = 2.0 * ( 2.0 * sum3 / sum2 * sumg - sum )
        sigrms = rms
        if( Hvv > 0.0 ): 
            sigrms = 1./2 / rms * sigv0 #combination of error formula for x^1/3
        #restore scaling
        avg = a * e1
        rms = rms * e1
        sigavg = sigavg * e1
        sigrms = sigrms * e1
        #"good" data points ( e.g. with sig < rms )
        if( sumg < 0.0 ):
            print('** ERROR in AVGRMSX. NON-POSITIVE SUM(G)=', gsum)
            print('** loop', loop, ' ndat', n, ' ngood', ng)
            print('** dat(:)', flux[:])
            print('** sig(:)', sig[:])
            good = 0
            return(good, avg, rms, sigavg, sigrms)
        #converge when test < 1
        if( loop > 10 ):
            safe = 0.9
            avg = oldavg * ( 1. - safe ) + safe * avg
            rms = oldrms * ( 1. - safe ) + safe * rms
            chiavg = ( avg - oldavg ) / sigavg
            chirms = ( rms - oldrms ) / sigrms
            test = max( abs( chiavg ), abs( chirms ) ) / tiny
            #report on last 5 iterations
            if( loop > nloop - 3 ):
                print('Loop', loop, ' of', nloop, ' in AVGRMSX')
                print('Ndat', n, ' Ngood', good, ' Neff', g)
                print(' avg', avg, ' rms', rms)
                print(' +/-', sigavg, ' +/-', sigrms)
                print(' chiavg', chiavg, ' chirms', chirms, ' test', test)
            #if( test < 1. ):
            #    print('** CONVERGED ** :))')
            #converged
            if( test < 1. ):
                good = 1
                return(good, avg, rms, sigavg, sigrms)
        #quit if v0 vanishes
        if( v0 <= 0. ): 
            print('no variance var <=0',var)
            return(good, avg, rms, sigavg, sigrms)
        #next loop
    #failed to converge
    print('** AVGRMSX FAILED TO CONVERGE :(((((')
    return(good, avg, rms, sigavg, sigrms)

# Check for empty FITS files from desdia code: 
def check_files(filenames_arr, empty_arr, ok_arr):
    for i in range(len(filenames_arr)):
        if os.stat(filenames_arr[i]).st_size == 0:
            empty_arr.append(filenames_arr[i])
        else:
            ok_arr.append(filenames_arr[i])

# Given array of fluxes and errors (floats), transform to array of magnitudes (floats):
def calc_mag(flux, flux_err):
    mag = 30 -2.5*np.log10(flux)
    mag_err = 2.5/np.log(10)*flux_err/flux
    return mag, mag_err

# Flag certain difference images for poor photometry:
def flag_check(photometry_table):
    apertures = ['ap_sum_3', 'ap_sum_4', 'ap_sum_5']
    poor_bool = []
    
    for i in range(len(apertures)):
        if photometry_table[apertures[i]] < 3e-28:
            poor_bool.append(True)
        else:
            poor_bool.append(False)

    photometry_table.add_column(np.all(poor_bool), name='Flag')
    return photometry_table

# Pull table data for non-flagged/flagged difference images:
def flag_data(photometry_table, flag_bool):
    idx = np.where(photometry_table['Flag']==flag_bool)[0]
    mjd = photometry_table['MJD'][idx]
    mag = [photometry_table['mag_3'][idx], photometry_table['mag_4'][idx], 
           photometry_table['mag_5'][idx]]
    merr = [photometry_table['merr_3'][idx], photometry_table['merr_4'][idx],
               photometry_table['merr_5'][idx]]
    return idx, mjd, mag, merr

# Compute variability statistics & add to arrays for source table:
def var_stats(flux, flux_err, arr_snr, arr_avg, arr_sigavg, arr_rms, arr_sigrms):
    good, avg, rms, sigavg, sigrms = myvarlc(flux, flux_err)
    snr = rms/sigrms 
    arr_snr.append(snr)
    arr_avg.append(avg)
    arr_sigavg.append(sigavg)
    arr_rms.append(rms)
    arr_sigrms.append(sigrms)
    return 

# Remove files with a specific extension once offset code is done:
def rem_files(filepaths_arr):
    for i in filepaths_arr:
        try:
            os.remove(i)
        except OSError as e:
            print('Error: %s : %s' % (f, e.strerror))
    return

# Convert ADUs to magnitudes, per SExtractor documentation:
def sx_mag(ap_sum, zero_point):
    if ap_sum > 0:
        mag = zero_point -2.5*np.log10(ap_sum)
    else:
        mag = 99.0
    return mag

def sx_mag_err(ap_sum, ap_sum_err, zero_point):
    if ap_sum > 0:
        merr = (2.5/np.log(10)) * (ap_sum_err/ap_sum)
    else:
        merr = 99.0
    return merr
    
# Main function:
def main(dia_dir_path, nsa_path, ccd, band='g'):
    start_time = time.time()
    print('---Beginning offset code...---')
    init_size = os.stat(dia_dir_path).st_size
    
    ##### READ FITS FILES:
    # Difference images:
    # diff_path = dia_dir_path + '/*_%s_c%_*_*proj_diff.fits' % (band, ccd)
    diff_path = dia_dir_path + '/*proj_diff.fits'
    diff_files = glob.glob(diff_path)
    empty_diff = []
    ok_diff = []
   
    # Difference image weights/error:
    dw_path = dia_dir_path + '/*proj_diff.weight.fits'
    dw_files = glob.glob(dw_path)
    empty_dw = []
    ok_dw = []

    print('\t # difference image files:', len(diff_files))        
    print('\t # difference image weights files:', len(dw_files), '\n')
    
    # Filter for empty diff img FITS files:
    print('\t Checking for empty difference image files...')
    check_files(diff_files, empty_diff, ok_diff)
    print('\t \t # difference image files:', len(ok_diff), '(OK),', len(empty_diff), '(empty) \n')
    
    # Filter for empty diff img weight FITS files:     
    print('\t Checking for empty difference image weight files...')
    check_files(dw_files, empty_dw, ok_dw)
    print('\t \t # difference image weights files:', len(ok_dw), '(OK),', len(empty_dw), '(empty) \n')
    
    # Template image:
    temp_path = dia_dir_path + '/template_c%d.fits' % ccd
    wtemp_path = dia_dir_path + '/template_c%d.weight.fits' % ccd
        
    ##### PULL DATA, HEADERS FROM FITS FILES: 
    # Difference images & weights:
    diff_data_set = []
    diff_hdr_set = []
    dw_data_set = []
    dw_hdr_set = []
    
    print('\t Extracting data from FITS files...')

    for i in range(len(ok_diff)):
        diff_data, diff_hdr = fits.getdata(ok_diff[i], header=True)
        dw_data, dw_hdr = fits.getdata(ok_dw[i], header=True)
        diff_data_set.append(diff_data)
        diff_hdr_set.append(diff_hdr)
        dw_data_set.append(dw_data)
        dw_hdr_set.append(dw_hdr)
    
    #Template images:
    # temp_data, temp_hdr = fits.getdata(str(temp_path[0]), header=True)
    temp_data, temp_hdr = fits.getdata(temp_path, header=True)
    wtemp_data, wtemp_hdr = fits.getdata(wtemp_path, header=True)

    ccd_path= dia_dir_path + '/' + str(temp_hdr['CCDNUM']) + 'des_offset'
    os.mkdir(ccd_path)
    
    ##### CO-ADDING & COMPARING:
    # Co-added difference image:
    abs_diff_data = np.abs(diff_data_set)
    co_add_data = np.zeros(np.shape(abs_diff_data[0]), dtype=float)
    print('\t Co-adding difference images... \n')
    for i in range(len(abs_diff_data)):
        co_add_data = co_add_data + abs_diff_data[i]
    
    # Co-added difference image:
    diff_fig, diff_ax = plt.subplots(figsize=(10,10))
    plt.style.use(astropy_mpl_style)
    diff_im = diff_ax.imshow(co_add_data, origin='lower', cmap='gray', vmin=0, vmax=5000)
    diff_fig.colorbar(diff_im, ax=diff_ax)
    diff_ax.set_title('Co-add difference image (multiple CCDs)')
    diff_ax.set_xlabel('Pixels')
    diff_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path + '/co_add.png'), facecolor='white', transparent=False)
    plt.show()

    # Template image:
    temp_fig, temp_ax = plt.subplots(figsize=(10,10))
    plt.style.use(astropy_mpl_style)
    temp_im = temp_ax.imshow(temp_data, origin='lower', cmap='gray', vmin=0, vmax=5000)
    temp_fig.colorbar(temp_im, ax=temp_ax)
    temp_ax.set_title('Template image (CCD ' + str(temp_hdr['CCDNUM']) + ')')
    temp_ax.set_xlabel('Pixels')
    temp_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path +'/temp.png'), facecolor='white', transparent=False)
    plt.show()
    
    ##### SOURCE DETECTION:
    # Search in co-added difference image:
    c_med = np.nanmedian(co_add_data)
    c_mean = np.nanmean(co_add_data)
    c_std = np.nanstd(co_add_data) 
    # print((c_mean, c_med, c_std)) 
    # What to do about the high stdev?

    daofind = DAOStarFinder(fwhm=3.0, threshold=3.*c_std) # When I used a detection threshold of 5, nothing showed up!
    
    print('\t Detecting sources in co-added difference image...')
    diff_sources = daofind(co_add_data - c_med)
    for col in diff_sources.colnames:
        diff_sources[col].info.format = '%.8g'  # for consistent table output

    # Search in template image:
    print('\t Detecting sources in template image... \n')
    temp_sources = daofind(temp_data - c_med)
    for col in temp_sources.colnames:
        temp_sources[col].info.format = '%.8g'  # for consistent table output

    # Co-added detections:
    pix_scale = diff_hdr_set[0]['PIXSCAL1'] * u.arcsec / u.pix # pixel scale is same along axis 1&2, same for all headers
    ap_pix = np.round(5*u.arcsec / pix_scale)

    dim_pos = np.transpose((diff_sources['xcentroid'], diff_sources['ycentroid']))
    dim_aps = CircularAperture(dim_pos, r=np.round(ap_pix.value/2))
    detect_fig, detect_ax = plt.subplots(figsize=(10,10))
    detect_im = detect_ax.imshow(co_add_data, cmap='gray', origin='lower', vmin=0, vmax=5000)
    dim_aps.plot(color='lime', lw=1.5, alpha=0.5)
    detect_fig.colorbar(detect_im, ax=detect_ax)
    detect_ax.set_title('Difference image detections')
    detect_ax.set_xlabel('Pixels')
    detect_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path + '/co_add_detections.png'), facecolor='white', transparent=False)
    plt.show()

    # Template detections:
    temp_pos = np.transpose((temp_sources['xcentroid'], temp_sources['ycentroid']))
    temp_aps = CircularAperture(temp_pos, r=np.round(ap_pix.value/2))
    temp_fig, temp_ax = plt.subplots(figsize=(10,10))
    temp_im = temp_ax.imshow(temp_data, cmap='gray', origin='lower', vmin=0, vmax=5000)
    temp_aps.plot(color='lime', lw=1.5, alpha=0.5)
    temp_fig.colorbar(temp_im, ax=temp_ax)
    temp_ax.set_title('Template image detections')
    temp_ax.set_xlabel('Pixels')
    temp_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path + '/temp_detections.png'), facecolor='white', transparent=False)
    plt.show()
    
    # OFFSET CANDIDATE SELECTION:
    # Convert pixel coordinates to SkyCoord coordinates: 
    w_diff = WCS(diff_hdr_set[0])
    diff_skycoord = SkyCoord.from_pixel(diff_sources['xcentroid'], diff_sources['ycentroid'], w_diff)
    diff_sources.add_column(diff_skycoord, name='SkyCoord')

    w_temp = WCS(temp_hdr)
    temp_skycoord = SkyCoord.from_pixel(temp_sources['xcentroid'], temp_sources['ycentroid'], w_temp)
    temp_sources.add_column(temp_skycoord, name='SkyCoord')

    # Get size of each extended source in template image:
    nsa_tab = Table.read(nsa_path) # make sure to import table!
    nsa_skycoord = SkyCoord(ra=nsa_tab['RA']*u.deg, dec=nsa_tab['DEC']*u.deg, frame='icrs')
    
    nsa_idx, temp_idx, d2d, d3d = temp_skycoord.search_around_sky(nsa_skycoord, 2*u.arcsec)
    if len(nsa_idx) > 0:
        temp_gal_p90 = nsa_tab['PETROTH90'][np.array(nsa_idx).astype(np.int)] # units in arcseconds
        temp_gal_p90 = np.array(temp_gal_p90).astype(np.int)
        temp_sources = temp_sources[np.asarray(temp_idx).astype(np.int)] # only keep sources in NSA
        nsa_tab = nsa_tab[np.asarray(nsa_idx).astype(np.int)]
    else:
        print('Error: No sources detected in the template image match the NASA-Sloan Atlas. Try another CCD.')
        print('Exiting offset analysis...')
        return
    
    temp_idcs = []
    diff_idcs = []
    dist = []
    # Calculate pair-wise separation:
    for i in range(len(temp_sources)):
        d2d = temp_sources[i]['SkyCoord'].separation(diff_sources['SkyCoord'])
        sep_bool = (d2d.to(u.arcsec)).value < temp_gal_p90[i]
        diff_idx = np.where(sep_bool)[0]
        diff_dist = d2d[diff_idx].to(u.arcsec)
        if len(diff_idx) > 0:
            for j in range(len(diff_idx)):
                temp_idcs.append(i)
                diff_idcs.append(diff_idx[j])
                dist.append(diff_dist[j].value)
    
    temp_idcs = np.array(temp_idcs).astype(np.int)
    diff_idcs = np.array(diff_idcs).astype(np.int)
    dist = np.array(dist).astype(np.int)
    
    # Visualize distribution of offsets:
    fig2, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10, 15), sharex=True, tight_layout=True)
    
    sns.histplot(data=dist, x=dist,   ax=ax1, stat='count', kde=True, color='green')
    ax1.set(xlabel="Separation ('')", ylabel="Counts", title="Raw data + KDE");
    
    sns.histplot(data=dist, x=dist,   ax=ax2, stat='probability', kde=True, color='blue')
    ax2.set(xlabel="Separation ('')", ylabel="Probability", title="Normed data + KDE");

    X2 = np.linspace(0, max(dist), 100)
    mu2, std2 = norm.fit(dist)
    mean2, var2, skew2, kurt2 = halfnorm.stats(moments='mvsk')
    p2 = norm.pdf(X2, mu2, std2)
    hn2 = halfnorm.pdf(X2)

    ax3.plot(X2, p2, 'k', linewidth=2, label='Gaussian')
    ax3.plot(X2, hn2, 'r-', linewidth=2, label='Half-norm')
    sns.histplot(data=dist, x=dist,   ax=ax3, stat='density')
    ax3.set(xlabel="Separation ('')", ylabel="Density", title="Offset Distribution")
    ax3.legend(loc='best')
    plt.savefig((ccd_path + '/offset_dist.jpg'), facecolor='white', transparent=False)

    gauss2 = "\t Gaussian fit (offset distribution): mu = %.2f '',  std = %.2f '' " % (mu2, std2)
    hnorm2 = "\t Half-norm fit (offset distribution): mu = %.2f '', std=%.2f '' \n" % (halfnorm.mean(), halfnorm.std())
    print(gauss2)
    print(hnorm2)
    
    # Pick a statistic, only include matches within certain bounds:
    cond_idx = np.where(dist > std2)
    diff_idcs = diff_idcs[cond_idx]
    temp_idcs = temp_idcs[cond_idx]

    dist = dist[dist > std]
    diff_sources = diff_sources[diff_idcs]
    temp_sources = temp_sources[temp_idcs]
    diff_sources.rename_column('id', 'detect_id')
    temp_sources.rename_column('id', 'detect_id')
    print('\t Found', len(diff_sources), 'pairs of candidate offset sources!')
 
    ##### OFFSET VISUALIZATION:
    # Make cutouts for each source in co-added difference image, template image:
    pix_scale = diff_hdr_set[0]['PIXSCAL1'] * u.arcsec / u.pix # pixel scale is same along axis 1/2, same for all headers
    ap_pix = np.round(5*u.arcsec / pix_scale)

    diff_cutouts = []
    temp_cutouts = []
    djnames = []
    tjnames = []
    match_ids = []

    tempcut_path = ccd_path + "/temp_cutouts"
    diffcut_path = ccd_path + "/diff_cutouts"
    os.mkdir(tempcut_path)
    os.mkdir(diffcut_path)

    print('\t Making and saving cutouts... \n')
    for i in range(len(diff_sources)):
        djrad = diff_sources[i]['SkyCoord'].ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
        djdec = diff_sources[i]['SkyCoord'].dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        djname = djrad+djdec
        djnames.append(djname)
        
        tjrad = temp_sources[i]['SkyCoord'].ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
        tjdec= temp_sources[i]['SkyCoord'].dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        tjname = tjrad+tjdec
        tjnames.append(tjname)
        
        match_id = i+1
        match_ids.append(match_id)
    
        diff_cut = Cutout2D(co_add_data, position=(diff_sources[i]['xcentroid'], diff_sources[i]['ycentroid']), 
                            size=(ap_pix, ap_pix))
        temp_cut = Cutout2D(temp_data, position=(temp_sources[i]['xcentroid'], temp_sources[i]['ycentroid']),
                            size=(ap_pix, ap_pix))
        diff_cutouts.append(diff_cut)
        temp_cutouts.append(temp_cut)
  
        ap_rad = np.int(ap_pix.value/2)
        
        # Co-added difference image detection:
        dcut_fig, dcut_ax = plt.subplots()
        dcut_ap = CircularAperture(((len(diff_cut.data)-1)/2, (len(diff_cut.data)-1)/2), r=ap_rad)
        normalize1 = ImageNormalize(stretch=SqrtStretch())
        dcut_ax.imshow(diff_cut.data, cmap='gray', origin='lower', norm=normalize1, interpolation='nearest')
        dcut_ax.set_xlabel('Pixels')
        dcut_ax.set_ylabel('Pixels')
        dcut_title = djname + "_dcut_matchid_" + str(match_id)
        dcut_ax.set_title(dcut_title)
        dcut_ax.text(x=15, y=1, s='r = '+str(np.int(5/2))+' as', color='lime')
        dcut_ap.plot(color='lime', lw=1.5, alpha=0.8)
        plt.savefig((diffcut_path + '/' + dcut_title + '.png'), facecolor='white', transparent=False)
    
        # Template image detection:
        tcut_fig, tcut_ax = plt.subplots()
        tcut_ap = CircularAperture(((len(temp_cut.data)-1)/2, (len(temp_cut.data)-1)/2), r=ap_rad)
        normalize2 = ImageNormalize(stretch=SqrtStretch())
        tcut_ax.imshow(temp_cut.data, cmap='gray', origin='lower', norm=normalize2, interpolation='nearest')
        tcut_ax.set_xlabel('Pixels')
        tcut_ax.set_ylabel('Pixels')
        tcut_title = tjname + "_tcut_matchid_" + str(match_id)
        tcut_ax.set_title(tcut_title)
        tcut_ax.text(x=15, y=1, s='r = '+str(np.int(5/2))+' as', color='lime')
        tcut_ap.plot(color='lime', lw=1.5, alpha=0.8)
        plt.savefig((tempcut_path + '/' + tcut_title + '.png'), facecolor='white', transparent=False)
        
        plt.close('all')

    diff_sources.add_column(match_ids, name='match_id')
    temp_sources.add_column(match_ids, name='match_id')
    new_order = ['match_id', 'detect_id', 'xcentroid','ycentroid','sharpness','roundness1',
                 'roundness2','npix','sky','peak','flux','mag','SkyCoord']
    diff_sources = diff_sources[new_order]
    temp_sources = temp_sources[new_order]
    diff_sources.write((ccd_path + '/diff_sources.csv'), format='ascii.csv')
    temp_sources.write((ccd_path + '/temp_sources.csv'), format='ascii.csv')
    
    ##### PHOTOMETRY:
    pdata_path = ccd_path + "/phot_data"
    os.mkdir(pdata_path)
    lc_path = ccd_path + "/light_curves"
    os.mkdir(lc_path)

    #snr3 = []
    #snr4 = []
    #snr5 = []
    #avg3 = []
    #avg4 = []
    #avg5 = []
    #sigavg3 = []
    #sigavg4 = []
    #sigavg5 = []
    #rms3 = []
    #rms4 = []
    #rms5 = []
    #sigrms3 = []
    #sigrms4 = []
    #sigrms5 = []

    # Calculate magnitude of each template image source using pixel counts:
    for i in range(len(temp_sources)):
        temp_phot_ap = SkyCircularAperture(positions=temp_sources['SkyCoord'][i], r=1.5*u.arcsec)
        temp_phot_ap = temp_phot_ap.to_pixel(wcs=w_temp)
        temp_phot_tab = aperture_photometry(temp_data, temp_phot_ap, error=wtemp_data)
        if i==0:
            tphot_tab = temp_phot_tab
        else:
            tphot_tab2 = temp_phot_tab
            tphot_tab = vstack([tphot_tab, tphot_tab2])
        
    zp_decam = 30 # arbitrarily defined here, but check hdul info
    temp_mags = []
    temp_merr = []
    for i in range(len(tphot_tab)):
        t_mag = sx_mag(tphot_tab['aperture_sum'][i], zp_decam)
        t_merr = sx_mag_err(tphot_tab['aperture_sum'][i], tphot_tab['aperture_sum_err'][i], zp_decam)
        temp_mags.append(t_mag)
        temp_merr.append(t_merr)
    tphot_tab.add_column(temp_mags, name='mag_des')
    tphot_tab.add_column(temp_err, name='mag_des_err')
    
    # Calibrate temp mags to NSA data to find the zero point between DES & NSA:
    fiber_mag = 22.5 - 2.5*np.log10(nsa_tab['FIBERFLUX'][3]) # check to make sure it's g-band!
    fiber_merr = 22.5 - 2.5*np.log10((nsa_tab['FIBERFLUX_IVAR'][3]**(-1/2)))
    des_to_nsa = fiber_mag - tphot_tab['mag_des']
    des_to_nsa_err = fiber_merr - tphot_tab['mag_des']
    
    # For each source, extract data from the difference images:
    print('\t Performing aperture photometry...')
    print('\t Making and saving light curves... \n')
    for i in range(len(diff_sources)):
        # Aperture photometry:
        radii = [np.round(3*u.arcsec / pix_scale).value/2., np.round(4*u.arcsec / pix_scale).value/2., 
                 np.round(5*u.arcsec / pix_scale).value/2.]
        diffim_aps = [CircularAperture([diff_sources[i]['xcentroid'], diff_sources[i]['ycentroid']], r=r) for r in radii]
        
        for j in range(len(abs_diff_data)):
            ptab = aperture_photometry(abs_diff_data[j], diffim_aps, error=dw_data_set[j])
            ap_names_og = ['aperture_sum_0', 'aperture_sum_1', 'aperture_sum_2']
            ap_names_new = ['ap_sum3', 'ap_sum4', 'ap_sum5']
            mag_names = ['mag_3', 'mag_4', 'mag_5']
            ap_err_og = ['aperture_sum_err_0', 'aperture_sum_err_1', 'aperture_sum_err_2']
            ap_err_new = ['ap_sum3_err', 'ap_sum4_err', 'ap_sum5_err']
            merr_names = ['merr_3', 'merr_4', 'merr_5']
            
            for k in range(len(ap_mag_og)):
                ptab.rename_column(ap_names_og[k], ap_names_new[k])
                ptab[mag_names[k]] = sx_mag(ptab[ap_names_new[k]], zp_decam) + des_to_nsa[i]
                ptab.rename_column(ap_err_og[k], ap_err_new[k])
                ptab[merr_names[k]] = sx_mag_err(ptab[ap_names_new[k]], ptab[ap_err_new[k]], zp_decam) + des_to_nsa_err[i]
            
            ptab['MJD'] = diff_hdr_set[j]['MJD-OBS']
            ptab = flag_check(ptab)
            ptab['diff_id'] = str(ok_diff[j])
            pt_order = ['MJD', 'xcenter', 'ycenter', 'ap_sum3', 'ap_sum3_err', 'mag_3', 'merr_3',
                        'ap_sum4', 'ap_sum4_err', 'mag_4', 'merr_4', 'ap_sum5', 'ap_sum5_err',
                        'mag_5', 'merr_5', 'Flag', 'diff_id']
            ptab = ptab[pt_order]
            
            if j==0:
               phot_table = ptab
            else:
               phot_table2 = ptab
               phot_table = vstack([phot_table, phot_table2])
        
        #var_stats(flux, flux_err, arr_snr, arr_avg, arr_sigavg, arr_rms, arr_sigrms)
        ptab_name = djnames[i] + "_photdata_matchid_" + str(match_ids[i])
        phot_table.write((pdata_path + '/' + ptab_name + '.csv'), format='ascii.csv')
        
        # Extract photometry data for non-flagged (norm_...) & flagged (flag_...) difference images:
        norm_idx, norm_mjd, norm_mag, norm_merr = flag_data(phot_table, False) 
        flag_idx, flag_mjd, flag_mag, flag_merr = flag_data(phot_table, True)
        
        # Plot light curves for each aperture:
        fig2, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        leg_labs = ['3', '4', '5']
        lc_name = djnames[i] + "_lc_matchid_" + str(match_ids[i])

        for k in range(len(leg_labs)):
            axes[k].errorbar(norm_mjd, norm_mag[k], yerr=norm_merr[k], ecolor='black', mfc='black', 
                            mec='black', marker='o', ls='', label='No Flag')
            if len(flag_idx > 0):
                axes[k].errorbar(flag_mjd, flag_mag[k], yerr=flag_merr[k], ecolor='darkgray', mfc='darkgray', 
                            mec='darkgray', marker='o', ls='', label='Flag')
            axes[k].legend(loc='best', frameon=True, title=leg_labs[k] + "'' Aperture")
            axes[k].set_ylabel('$g$ magnitude')
            if (k==0):
                axes[k].set_title('Source ' + djnames[i] + ' Light Curve')
            if (k==2):
                axes[k].set_xlabel('MJD')
    
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig((lc_path + '/' + lc_name + '.png'), facecolor='white', transparent=False)
    
    final_size = os.stat(dia_dir_path).st_size
    
    print('\t DATA PRODUCT SUMMARY:')
    print('\t ---------------------')
    print('\t \t Under', ccd_path, ':')
    print('\t \t \t Co-added difference image, template image, co-add/template detections,')
    print('\t \t \t offset distribution, list of co-add/template sources')
        
    print('\t \t Under', diffcut_path, ':')
    print('\t \t \t Co-added difference image cutouts')
        
    print('\t \t Under', tempcut_path, ':')
    print('\t \t \t Template image cutouts')
        
    print('\t \t Under', pdata_path, ':')
    print('\t \t \t Photometry tables')
        
    print('\t \t Under', lc_path, ':')
    print('\t \t \t Light curves \n')
        
    print('\t Removing FITS files', ('('+ naturalsize(init_size) + ') from directory'), dia_dir_path, '(total size', (naturalsize(final_size) + ')...'))
    all_fits = glob.glob((dia_dir_path + '/*.fits'))
    all_cat = glob.glob((dia_dir_path + '/*.cat'))
    all_head = glob.glob((dia_dir_path + '/*.head'))
    rem_files(all_fits)
    rem_files(all_cat)
    rem_files(all_head)
    postrem_size = os.stat(dia_dir_path).st_size
    print('\t FITS files removed from directory', dia_dir_path, '(total size', (naturalsize(postrem_size) + '). \n'))
    
    print('---Offset code executed in %.2f minutes--- \n' % float((time.time() - start_time)/60))
    
    return

###############################################################################
# Main function:
main('/data/des80.a/data/gtorrini/1', '/data/des80.a/data/cburke/nsa_v0_1_2.fits', 1, 'g')
