# For main function input, go to line 586

# Last modified 5/19/21
import os
import glob
import math
from math import nan
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.table import vstack, Table
from astropy.visualization import astropy_mpl_style, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
from scipy.stats import norm, halfnorm

sns.set_theme(style='whitegrid')
plt.style.use('seaborn-whitegrid')

#################################################################################################
# FUNCTIONS
# Check for variability: 
def myvarlc(fluxin,sigin):
    n = np.shape(fluxin)[0]
    #convergence threshold
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
    cts = [photometry_table['ap_sum_3'][idx], photometry_table['ap_sum_4'][idx], 
           photometry_table['ap_sum_5'][idx]]
    cts_err = [photometry_table['ap_sum_3_err'][idx], photometry_table['ap_sum_4_err'][idx],
               photometry_table['ap_sum_5_err'][idx]]
    return idx, mjd, cts, cts_err

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

# Main function:
def main(dia_dir_path, ccd, band='g'):
    des_cat_path = os.path.joing(dia_dir_path, 'template_%d' % ccd)
    
    ##### READ FITS FILES:
    # Difference images:
    diff_path = dia_dir_path + '/*_%s_c%d_*_*proj_diff.fits' % (band, ccd)
    diff_files = glob.glob(diff_path)
    empty_files = []
    ok_diff = []
    
    # Difference image weights/error:
    dw_path = dia_dir_path + '/*proj_diff.weight.fits'
    dw_files = glob.glob(dw_path)
    empty_dw = []
    ok_dw = []

    print(len(dw_files), len(diff_files))
    
    # Filter for empty FITS files
    for i in range(len(diff_files)): 
        if os.stat(diff_files[i]).st_size == 0: 
            empty_files.append(diff_files[i])
            empty_dw.append(dw_files[i])
        else:
            ok_diff.append(diff_files[i])
            ok_dw.append(dw_files[i])
    
    # Template image:
    temp_files= dia_dir_path + '/template_c%d.fits' % ccd
    temp_path = glob.glob(temp_files)
    for i in range(len(temp_path)):
        if temp_path[i].endswith('weight.fits'):
            temp_path.pop(i)
        
    ##### PULL DATA, HEADERS FROM FITS FILES: 
    # Difference images & weights:
    diff_data_set = []
    diff_hdr_set = []
    dw_data_set = []
    dw_hdr_set = []
    
    for i in range(len(ok_diff)):
        diff_data, diff_hdr = fits.getdata(ok_diff[i], header=True)
        dw_data, dw_hdr = fits.getdata(ok_dw[i], header=True)
        diff_data_set.append(diff_data)
        diff_hdr_set.append(diff_hdr)
        dw_data_set.append(dw_data)
        dw_hdr_set.append(dw_hdr)
    
    #Template images:
    temp_data, temp_hdr = fits.getdata(str(temp_path[0]), header=True)
    
    ccd_path= dia_dir_path + '/' + str(temp_hdr['CCDNUM']) + 'des_offset'
    os.mkdir(ccd_path)
    
    ##### CO-ADDING & COMPARING:
    # Co-added difference image:
    abs_diff_data = np.abs(diff_data_set)
    co_add_data = np.zeros(np.shape(abs_diff_data[0]), dtype=float)
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
    temp_im = temp_ax.imshow(temp_data, origin='lower', cmap='gray', 
                             vmin=0, vmax=5000)
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
    
    diff_sources = daofind(co_add_data - c_med)
    for col in diff_sources.colnames:
        diff_sources[col].info.format = '%.8g'  # for consistent table output
        # print(diff_sources)

    # Search in template image:
    temp_sources = daofind(temp_data - c_med)
    for col in temp_sources.colnames:
        temp_sources[col].info.format = '%.8g'  # for consistent table output
        # print(temp_sources)

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

    # Calculate pair-wise separation:
    diff_idx, temp_idx, d2d, d3d = temp_sources['SkyCoord'].search_around_sky(diff_sources['SkyCoord'], 5*u.arcsec)

    # Visualize distribution of offsets:
    fig1, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10, 15), sharex=True, tight_layout=True)
    
    sns.histplot(data=d2d.to(u.arcsec).value, x=d2d.to(u.arcsec).value, ax=ax1, stat='count', kde=True, color='green')
    ax1.set(xlabel="Separation ('')", ylabel="Counts", title="Raw data + KDE");
    
    sns.histplot(data=d2d.to(u.arcsec).value, x=d2d.to(u.arcsec).value, ax=ax2, stat='probability', kde=True, color='blue')
    ax2.set(xlabel="Separation ('')", ylabel="Probability", title="Normed data + KDE");

    X = np.linspace(0, 2.5, 100)
    mu, std = norm.fit(d2d.to(u.arcsec).value)
    mean, var, skew, kurt = halfnorm.stats(moments='mvsk')
    p = norm.pdf(X, mu, std)
    hn = halfnorm.pdf(X)

    ax3.plot(X, p, 'k', linewidth=2, label='Gaussian')
    ax3.plot(X, hn, 'r-', linewidth=2, label='Half-norm')
    sns.histplot(data=d2d.to(u.arcsec).value, x=d2d.to(u.arcsec).value, ax=ax3, stat='density')
    ax3.set(xlabel="Separation ('')", ylabel="Density", title="Offset Distribution")
    ax3.legend(loc='best')
    plt.savefig((ccd_path + '/offset_dist.png'), facecolor='white', transparent=False)

    gauss = "Gaussian results: mu = %.2f,  std = %.2f" % (mu, std)
    hnorm = "Half-norm results: mu = %.2f, std=%.2f" % (halfnorm.mean(), halfnorm.std())
    print(gauss, hnorm)
    
    # Pick a statistic, only include matches within certain bounds:
    cond_idx = np.where(d2d.to(u.arcsec).value > std)
    diff_idx = diff_idx[cond_idx]
    temp_idx = temp_idx[cond_idx]

    d2d = d2d[d2d.to(u.arcsec).value > std]
    diff_sources = diff_sources[diff_idx]
    temp_sources = temp_sources[temp_idx]
    diff_sources.rename_column('id', 'detect_id')
    temp_sources.rename_column('id', 'detect_id')

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
    diff_sources.write((ccd_path + '/diff_sources.csv'), format=ascii.csv)
    temp_sources.write((ccd_path + '/temp_sources.csv'), format=ascii.csv)
    
    ##### PHOTOMETRY:
    pdata_path = ccd_path + "/phot_data"
    os.mkdir(photdata_path)
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

    # For each source, extract data from the difference images:
    for i in range(len(diff_sources)):
        # Aperture photometry:
        radii = [np.round(3*u.arcsec / pix_scale).value/2., np.round(4*u.arcsec / pix_scale).value/2., 
                 np.round(5*u.arcsec / pix_scale).value/2.]
        
        for j in range(len(abs_diff_data)):
            diffim_aps = [CircularAperture([diff_sources[i]['xcentroid'], diff_sources[i]['ycentroid']], r=r) 
                          for r in radii]
            
            if j==0:
                phot_table = aperture_photometry(abs_diff_data[j], diffim_aps, error=dw_data_set[j])
                for col in phot_table.colnames:
                    phot_table[col].info.format = '%.8g'
                phot_table.rename_column('aperture_sum_0', 'ap_sum_3')
                phot_table.rename_column('aperture_sum_err_0', 'ap_sum_3_err')
                phot_table.rename_column('aperture_sum_1', 'ap_sum_4')
                phot_table.rename_column('aperture_sum_err_1', 'ap_sum_4_err')
                phot_table.rename_column('aperture_sum_2', 'ap_sum_5')
                phot_table.rename_column('aperture_sum_err_2', 'ap_sum_5_err')
                phot_table.add_column(diff_hdr_set[j]['MJD-OBS'], name='MJD')
                phot_table = flag_check(phot_table)
                phot_table['diff_id'] = str(ok_diff[j])
                pt_order = ['MJD', 'xcenter', 'ycenter', 'ap_sum_3', 'ap_sum_3_err', 
                            'ap_sum_4', 'ap_sum_4_err', 'ap_sum_5', 'ap_sum_5_err', 'Flag','diff_id']
                phot_table = phot_table[pt_order]
            
            else:
                ptab2 = aperture_photometry(abs_diff_data[j], diffim_aps, error=dw_data_set[j])
                for col in ptab2.colnames:
                    ptab2[col].info.format = '%.8g'
                ptab2.rename_column('aperture_sum_0', 'ap_sum_3')
                ptab2.rename_column('aperture_sum_err_0', 'ap_sum_3_err')
                ptab2.rename_column('aperture_sum_1', 'ap_sum_4')
                ptab2.rename_column('aperture_sum_err_1', 'ap_sum_4_err')
                ptab2.rename_column('aperture_sum_2', 'ap_sum_5')
                ptab2.rename_column('aperture_sum_err_2', 'ap_sum_5_err')
                ptab2.add_column(diff_hdr_set[j]['MJD-OBS'], name='MJD')
                ptab2 = flag_check(ptab2)
                ptab2['diff_id'] = str(ok_diff[j])
                ptab2 = ptab2[pt_order]
                phot_table = vstack([phot_table, ptab2])
        
        #var_stats(flux, flux_err, arr_snr, arr_avg, arr_sigavg, arr_rms, arr_sigrms)
        ptab_name = djnames[i] + "_photdata_matchid_" + str(match_ids[i])
        phot_table.write((pdata_path + '/' + ptab_name + '.csv'), format=ascii.csv)
        
        # Extract photometry data for non-flagged (norm_...) & flagged (flag_...) difference images:
        pix_cts = [phot_table['ap_sum_3'], phot_table['ap_sum_4'], phot_table['ap_sum_5']]
        pix_cts_err = [phot_table['ap_sum_3_err'], phot_table['ap_sum_4_err'], phot_table['ap_sum_5_err']]
       
        norm_idx, norm_mjd, norm_cts, norm_cts_err = flag_data(phot_table, False) 
        flag_idx, flag_mjd, flag_cts, flag_cts_err = flag_data(phot_table, True)
        
        # Plot light curves for each aperture:
        fig2, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        leg_labs = ['3', '4', '5']
        lc_name = djnames[i] + "_lc_matchid_" + str(match_ids[i])

        for k in range(len(pix_cts)):
            axes[k].errorbar(norm_mjd, norm_cts[k], yerr=norm_cts_err[k], ecolor='black', mfc='black', 
                            mec='black', marker='o', ls='', label='No Flag')
            if len(flag_idx > 0):
                axes[k].errorbar(flag_mjd, flag_cts[k], yerr=flag_cts_err[k], ecolor='darkgray', mfc='darkgray', 
                            mec='darkgray', marker='o', ls='', label='Flag')
            axes[k].legend(loc='best', frameon=True, title=leg_labs[k] + "'' Aperture")
            axes[k].set_ylabel(r'Brightness (e$^-$/s)')
            axes[k].set_yscale('log')
            if (k==0):
                axes[k].set_title('Source ' + djnames[i] + ' Light Curve')
            if (k==2):
                axes[k].set_xlabel('MJD')
    
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig((lc_path + '/' + lc_name + '.png'), facecolor='white', transparent=False)
    return

#################################################################################################
# Main function example usage:
# main('C:/Users/Gaby/Documents/DES/SP_2021/PHL_293B_validation/TARGET', 34)