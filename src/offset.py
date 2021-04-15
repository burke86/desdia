# For main function input, go to line 524

# Last modified 4/12/21
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from math import nan
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.visualization import astropy_mpl_style, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder, CircularAperture
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

# Create columns converted to magnitudes (flux_col, err_col are str): 
def lc_data(lc_df, flux_col, err_col):
    mag, mag_err = calc_mag(lc_df[flux_col], lc_df[err_col])
    return mag, mag_err

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
    
    # READ FITS FILES:
    # Difference images:
    diff_path = dia_dir_path + '/*_%s_c%d_*_*proj_diff.fits' % (band, ccd)
    diff_files = glob.glob(diff_path)
    empty_files = []
    ok_diff = []

    for i in range(len(diff_files)): 
        if os.stat(diff_files[i]).st_size == 0: # Filter for empty FITS files
            empty_files.append(diff_files[i])
        else:
            ok_diff.append(diff_files[i])

    # Template image:
    temp_files= dia_dir_path + '/template_c%d.fits' % ccd
    temp_path = glob.glob(temp_files)
    for i in range(len(temp_path)):
        if temp_path[i].endswith('weight.fits'):
            temp_path.pop(i)
        
    # EXTRACT DATA AND HEADERS FROM FITS FILES: 
    # Difference images:
    diff_data_set = []
    diff_hdr_set = []
    for i in range(len(ok_diff)):
        data, hdr = fits.getdata(ok_diff[i], header=True)
        diff_data_set.append(data)
        diff_hdr_set.append(hdr)

    #Template images:
    temp_data, temp_hdr = fits.getdata(str(temp_path[0]), header=True)
    ccd_path= dia_dir_path + '/' + str(temp_hdr['CCDNUM']) + 'des_offset'
    os.mkdir(ccd_path)
    
    # CO-ADD THE DIFFERENCE IMAGES:
    abs_diff_data = np.abs(diff_data_set)
    co_add_data = np.zeros(np.shape(abs_diff_data[0]), dtype=float)
    for i in range(len(abs_diff_data)):
        co_add_data = co_add_data + abs_diff_data[i]
    
    # COMPARE CO-ADDED DIFFERENCE & TEMPLATE IMAGES:
    # Co-added difference image:
    diff_fig, diff_ax = plt.subplots(figsize=(10,10))
    plt.style.use(astropy_mpl_style)
    diff_im = diff_ax.imshow(co_add_data, origin='lower', cmap='gray', 
                             vmin=0, vmax=5000)
    diff_fig.colorbar(diff_im, ax=diff_ax)
    diff_ax.set_title('Co-add difference image (multiple CCDs)')
    diff_ax.set_xlabel('Pixels')
    diff_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path + '/co_add.jpg'), facecolor='white', transparent=False)
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
    plt.savefig((ccd_path +'/temp.jpg'), facecolor='white', transparent=False)
    plt.show()
    
    # SEARCH FOR DETECTIONS:
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

    temp_sources = daofind(temp_data - c_med)
    for col in temp_sources.colnames:
        temp_sources[col].info.format = '%.8g'  # for consistent table output
        # print(temp_sources)

    # PLOT ALL DETECTIONS:
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
    plt.savefig((ccd_path + '/co_add_detections.jpg'), facecolor='white', transparent=False)
    plt.show()

    temp_pos = np.transpose((temp_sources['xcentroid'], temp_sources['ycentroid']))
    temp_aps = CircularAperture(temp_pos, r=np.round(ap_pix.value/2))
    temp_fig, temp_ax = plt.subplots(figsize=(10,10))
    temp_im = temp_ax.imshow(temp_data, cmap='gray', origin='lower', vmin=0, vmax=5000)
    temp_aps.plot(color='lime', lw=1.5, alpha=0.5)
    temp_fig.colorbar(temp_im, ax=temp_ax)
    temp_ax.set_title('Template image detections')
    temp_ax.set_xlabel('Pixels')
    temp_ax.set_ylabel('Pixels')
    plt.savefig((ccd_path + '/temp_detections.jpg'), facecolor='white', transparent=False)
    plt.show()
    
    # FIND OFFSETS:
    # Convert pixel coordinates to SkyCoord coordinates: 
    w_diff = WCS(diff_hdr_set[0])
    diff_skycoord = SkyCoord.from_pixel(diff_sources['xcentroid'], diff_sources['ycentroid'], w_diff)
    diff_sources.add_column(diff_skycoord, name='SkyCoord')

    w_temp = WCS(temp_hdr)
    temp_skycoord = SkyCoord.from_pixel(temp_sources['xcentroid'], temp_sources['ycentroid'], w_temp)
    temp_sources.add_column(temp_skycoord, name='SkyCoord')

    # Calculate pair-wise separation:
    diff_idx, temp_idx, d2d, d3d = temp_sources['SkyCoord'].search_around_sky(diff_sources['SkyCoord'], 5*u.arcsec)

    # Plots:
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
    plt.savefig((ccd_path + '/offset_dist.jpg'), facecolor='white', transparent=False)

    gauss = "Gaussian results: mu = %.2f,  std = %.2f" % (mu, std)
    hnorm = "Half-norm results: mu = %.2f, std=%.2f" % (halfnorm.mean(), halfnorm.std())
    print(gauss, hnorm)
    
    # CANDIDATE SELECTION: 
    # Pick a statistic, only include matches withing certain bounds:
    cond_idx = np.where(d2d.to(u.arcsec).value > std)
    diff_idx=diff_idx[cond_idx]
    temp_idx=temp_idx[cond_idx]

    d2d = d2d[d2d.to(u.arcsec).value > std]
    diff_sources=diff_sources[diff_idx]
    temp_sources=temp_sources[temp_idx]
    diff_sources.rename_column('id', 'detect_id')
    temp_sources.rename_column('id', 'detect_id')

    # Make cutouts:
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
        diff_cut = Cutout2D(co_add_data, position=(diff_sources[i]['xcentroid'], diff_sources[i]['ycentroid']), 
                            size=(ap_pix, ap_pix))
        temp_cut = Cutout2D(temp_data, position=(temp_sources[i]['xcentroid'], temp_sources[i]['ycentroid']),
                            size=(ap_pix, ap_pix))
        djrad = diff_sources[i]['SkyCoord'].ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
        djdec= diff_sources[i]['SkyCoord'].dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        djname = djrad+djdec
        djnames.append(djname)
        tjrad = temp_sources[i]['SkyCoord'].ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
        tjdec= temp_sources[i]['SkyCoord'].dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        tjname = tjrad+tjdec
        tjnames.append(tjname)
        match_id = i+1
        match_ids.append(match_id)
    
        diff_cutouts.append(diff_cut)
        temp_cutouts.append(temp_cut)
  
        ap_rad = np.int(ap_pix.value/2)
    
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
        plt.savefig((diffcut_path + '/' + dcut_title + '.jpg'), facecolor='white', transparent=False)
    
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
        plt.savefig((tempcut_path + '/' + tcut_title + '.jpg'), facecolor='white', transparent=False)
        plt.close('all')

    diff_sources.add_column(match_ids, name='match_id')
    temp_sources.add_column(match_ids, name='match_id')
    new_order = ['match_id', 'detect_id', 'xcentroid','ycentroid','sharpness','roundness1',
                 'roundness2','npix','sky','peak','flux','mag','SkyCoord']
    diff_sources = diff_sources[new_order]
    temp_sources = temp_sources[new_order]
    
    # Light curves:
    # Load catalog:
    files = glob.glob(des_cat_path) 
    df_lc_dia = pd.concat([pd.read_csv(f, sep='\s+', escapechar='#') for f in files])
    #df_lc_dia = df_lc_dia[np.isfinite(df_lc_dia['flux5'])] # Clean NaN
    coord_des_dia = SkyCoord(df_lc_dia['ra'], df_lc_dia['dec'], unit=u.deg)

    catmatch_path = ccd_path + "/cat_matches"
    os.mkdir(catmatch_path)
    lc_path = ccd_path + "/light_curves"
    os.mkdir(lc_path)

    snr3 = []
    snr4 = []
    snr5 = []
    avg3 = []
    avg4 = []
    avg5 = []
    sigavg3 = []
    sigavg4 = []
    sigavg5 = []
    rms3 = []
    rms4 = []
    rms5 = []
    sigrms3 = []
    sigrms4 = []
    sigrms5 = []

    # For each source, catalog match & plot light curves:
    for i in range(len(diff_sources)):
        # Using a separation constraint of [x] arcseconds:
        cat_dist = diff_sources[i]['SkyCoord'].separation(coord_des_dia)
        sep_cond = cat_dist < 5*u.arcsec
        
        # Convert boolean array above to indices:
        matched_idx = np.where(sep_cond)[0]
        
        # Create a dataframe only containining data from matches:
        matched_df = df_lc_dia.iloc[matched_idx, :]
        matched_df = matched_df.sort_values(by=['mjd_obs']) # Sort by MJD
        df_name = djnames[i] + "_catmatch_matchid_" + str(match_ids[i])
        matched_df.to_pickle(catmatch_path + '/' + df_name + '.pkl')
        
        fl_cols = ['flux3', 'flux4', 'flux5']
        flerr_cols = ['fluxerr3', 'fluxerr4', 'fluxerr5']

        # Create arrays of mags./mag. errs. for all aperture diameters:
        mag_set = []
        magerr_set = []
        for j in range(len(fl_cols)):
            mag_col, magerr_col = lc_data(matched_df, fl_cols[j], flerr_cols[j])
            mag_set.append(mag_col)
            magerr_set.append(magerr_col)

        var_stats(np.array(matched_df['flux3']), np.array(matched_df['fluxerr3']), snr3, avg3, sigavg3, rms3, sigrms3)
        var_stats(np.array(matched_df['flux4']), np.array(matched_df['fluxerr4']), snr4, avg4, sigavg4, rms4, sigrms4)
        var_stats(np.array(matched_df['flux5']), np.array(matched_df['fluxerr5']), snr5, avg5, sigavg5, rms5, sigrms5)
    
        fig2, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        labs = ['3', '4', '5']
        colors = ['blue', 'darkorange', 'green']
        lc_name = djnames[i] + "_lc_matchid_" + str(match_ids[i])
    
        # Plot light curves for each aperture diameter:
        for k in range(len(mag_set)):
            axes[k].errorbar(matched_df['mjd_obs'], mag_set[k], yerr=magerr_set[k], ecolor='black', mfc=colors[k], 
                            mec='black', marker='o', ls='')
            axes[k].legend(loc='best', labels=labs[k], frameon=True, title="Aperture ('')")
            axes[k].invert_yaxis()
            axes[k].set_ylabel('Magnitude ($g$)')
            if (k==0):
                axes[k].set_title('Source ' + djnames[i] + ' Light Curve')
            if (k==2):
                axes[k].set_xlabel('MJD')
    
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig((lc_path + '/' + lc_name + '.jpg'), facecolor='white', transparent=False)
        plt.close('all')
    
    diff_sources.add_columns([snr3, avg3, sigavg3, rms3, sigrms3], names=['snr3', 'avg3', 'sigavg3', 'rms3', 'sigrms3'])
    diff_sources.add_columns([snr4, avg4, sigavg4, rms4, sigrms4], names=['snr4', 'avg4', 'sigavg4', 'rms4', 'sigrms4'])
    diff_sources.add_columns([snr5, avg5, sigavg5, rms5, sigrms5], names=['snr5', 'avg5', 'sigavg5', 'rms5', 'sigrms5'])
        
    temp_df = temp_sources.to_pandas()
    diff_df = diff_sources.to_pandas()
    diff_df.to_pickle((ccd_path + '/diff_sources.pkl'))
    temp_df.to_pickle((ccd_path + '/temp_sources.pkl'))
    return

# Main function example usage:
# main('C:/Users/Gaby/Documents/DES/SP_2021/PHL_293B_validation/TARGET, 34)