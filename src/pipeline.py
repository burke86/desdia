import os, sys
import numpy as np
from multiprocessing import Pool as Pool
from astropy import units as u
from astropy.coordinates import SkyCoord
from misc import *

def difference(file_info):
    top_path = os.path.abspath(__file__)
    top_dir = '/'.join(os.path.dirname(top_path).split('/')[0:-1])
    hotpants_file = os.path.join(top_dir,'etc/DES.hotpants')
    local_path = str(file_info["path"])
    file_root = local_path[0:-5]
    path_root = os.path.dirname(local_path)
    file_sci = file_root + "_proj.fits"
    file_wgt = file_root + "_proj.weight.fits"
    outfile_sci = file_root + "_proj_diff.fits"
    outfile_wgt = file_root + "_proj_diff.weight.fits"
    template_sci = os.path.join(path_root,"template.fits")
    template_wgt = os.path.join(path_root,"template.weight.fits")
    hotpants_pars = ''.join(open(hotpants_file,'r').readlines())
    # HOTPANTS input parameters
    code = bash('hotpants -inim %s -ini %s -tmplim %s -tni %s -outim %s -oni %s -useWeight %s' % (file_sci,file_wgt,template_sci,template_wgt,outfile_sci,outfile_wgt,hotpants_pars))
    safe_rm(file_sci)
    safe_rm(file_wgt)
    # Handle HOTPANTS fatal error
    if code != 0:
        return None
    return file_info

class Pipeline:

    def __init__(self,bands,program,usr,psw,work_dir,out_dir,top_dir=None):
        self.bands = bands
        self.program = program
        self.usr = usr
        self.psw = psw
        self.tile_dir = work_dir
        self.out_dir = out_dir
        # setup directories
        top_path = os.path.abspath(__file__)
        # default paths
        if top_dir == None:
            self.top_dir = '/'.join(os.path.dirname(top_path).split('/')[0:-1])
            self.hotpants_file = os.path.join(self.top_dir,'etc/DES.hotpants')
            self.sex_file = os.path.join(self.top_dir,'etc/SN_diffim.sex')
            self.swarp_file = os.path.join(self.top_dir,'etc/SN_distemp.swarp')
            self.swarp_file_nite = os.path.join(self.top_dir,'etc/SN_nitecmb.swarp')
            self.sex_pars = ""
        # specified directory (FermiGrid)
        else:
            self.top_dir = top_dir
            self.hotpants_file = os.path.join(self.top_dir,'DES.hotpants')
            self.sex_file = os.path.join(self.top_dir,'SN_diffim.sex')
            self.swarp_file = os.path.join(self.top_dir,'SN_distemp.swarp')
            self.swarp_file_nite = os.path.join(self.top_dir,'SN_nitecmb.swarp')
            par_name = os.path.join(self.top_dir,"SN_diffim.sex.param")
            flt_name = os.path.join(self.top_dir,"SN_diffim.sex.conv")
            self.sex_pars = " -PARAMETERS_NAME %s -FILTER_NAME %s" % (par_name,flt_name)
        # get hotpants parameters
        self.hotpants_pars = ''.join(open(self.hotpants_file,'r').readlines())
        # make directories
        if not os.path.exists(self.tile_dir):
            os.makedirs(self.tile_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def download_image(self,archive_info):
        filepath = archive_info['path']
        filename = archive_info['filename']
        compression = archive_info['compression']
        archive_path = os.path.join(filepath,filename+compression)
        # download image from image archive server
    	url = os.path.join('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/',archive_path)
        bash('wget -nc -q --user %s --password %s %s -P %s' % (self.usr,self.psw,url,self.tile_dir),False)
        local_path = os.path.join(self.tile_dir,archive_path.split("/")[-1])
        dtype_info = [("path","|S200"),("mjd_obs",float),("nite",int),("psf_fwhm",float),("skysigma",float),("mag_zero",float),("sigma_mag_zero",float)]
        info_list = (local_path,archive_info["mjd_obs"],archive_info["nite"],archive_info["psf_fwhm"],archive_info["skysigma"],archive_info["mag_zero"],archive_info["sigma_mag_zero"])
        info_list = np.array(info_list,dtype=dtype_info)
        return info_list


    def make_weight(self,archive_info):
        # get reduced images ready for generating template
        local_path = archive_info["path"]
        # background, background variation
        file_root = local_path[0:-8]
        file_sci = file_root + ".fits"
        file_wgt = file_root + ".weight.fits"
        # make weight maps and mask
        code = bash('makeWeight -inFile_img %s -border 20 -outroot %s' % (local_path,file_root))
        if code != 0: return None
        # convert files to single-header format
        single_header(file_sci)
        single_header(file_wgt)
        # create final list
        dtype_info = [("path","|S200"),("mjd_obs",float),("nite",int),("psf_fwhm",float),("skysigma",float),("mag_zero",float),("sigma_mag_zero",float)]
        info_list = (file_sci,archive_info["mjd_obs"],archive_info["nite"],archive_info["psf_fwhm"],archive_info["skysigma"],archive_info["mag_zero"],archive_info["sigma_mag_zero"])
        info_list = np.array(info_list,dtype=dtype_info)
        safe_rm(local_path)
        return info_list


    def combine_night(self,file_info,tile_head,num_threads):
        # combine images taken on same night into single tile
        tiledir = os.path.dirname(file_info[0][0])
        ps = tile_head['PIXELSCALE'][0] # arcseconds/pixel
        size_x = tile_head['NAXIS1'][0] # pixels
        size_y = tile_head['NAXIS2'][0] # pixels
        ra_cent = tile_head['RA_CENT'][0] # arcseconds
        dec_cent = tile_head['DEC_CENT'][0] # arcseconds
        file_list_out = []
        # nite loop
        for nite in np.unique(file_info["nite"]):
            # get images on same taken night
            file_info_nite = file_info[file_info["nite"] == nite]
            # file names
            swarp_list = " ".join(file_info_nite["path"])
            weight_list = swarp_list.replace(".fits",".weight.fits")
            imageout_name = swarp_list.split()[-1]
            chip = imageout_name.split('_c')[1].split('_')[0]
            imageout_name = imageout_name.replace('_c'+chip+'_','_')
            weightout_name = weight_list.split()[-1].replace('_c'+chip+'_','_')
            # tile images taken on same night
            bash('swarp %s -c %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS %d -CENTER %s,%s -PIXEL_SCALE %f -IMAGE_SIZE %d,%d -RESAMPLE_DIR %s' % (swarp_list,self.swarp_file_nite,imageout_name,weightout_name,num_threads,ra_cent,dec_cent,ps,size_x,size_y,tiledir))
            for remove_path in swarp_list.split():
                safe_rm(remove_path)
            for remove_path in weight_list.split():
                safe_rm(remove_path)
            # new combined file info array
            mjd = np.median(np.unique(file_info_nite["mjd_obs"]))
            sky_noise = np.median(file_info_nite["skysigma"])
            psf_fwhm = np.median(file_info_nite["psf_fwhm"])
            mag_zero = np.median(file_info_nite["mag_zero"])
            sigma_mag_zero = np.median(file_info_nite["sigma_mag_zero"])
            file_list_out.append((imageout_name,mjd,psf_fwhm,sky_noise,mag_zero,sigma_mag_zero))
        dtype_info = [('path','|S200'),('mjd_obs',float),('psf_fwhm',float),('skysigma',float),('mag_zero',float),('sigma_mag_zero',float)]
        file_info_out = np.array(file_list_out,dtype_info)
        return file_info_out

    
    def align(self,filename_in):
        # project and align images to template
        file_root = filename_in[0:-5]
        path_root = os.path.dirname(filename_in)
        filename_out = file_root+"_proj.fits"
        file_header = file_root+"_proj.head"
        template_sci = os.path.join(path_root,"template.fits")
        # symbolic link for header geometry
        bash('ln -s %s %s' % (template_sci,file_header))
        bash('swarp %s -c %s -NTHREADS %d -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s.weight.fits -RESAMPLE_DIR %s' % (filename_in,self.swarp_file,1,filename_out,filename_out[0:-5],path_root))
        safe_rm(filename_in)
        safe_rm(filename_in[0:-5]+".weight.fits")
        safe_rm(file_header)
        return filename_in


    def generate_light_curves(self,cat_info):
        # generates light curve files from list of sextractor catalogs
        # with forced photometry on the template image
        # measure template flux to add to difference flux
        template_cat = os.path.join(self.tile_dir,'template.cat')
        lc_out = []
        # assumes all catalogs have same number of lines (detections)
        # true if doing forced photometry on template image
        with open(template_cat,'r') as cat_template:
            lines_template = cat_template.readlines()
            # for each difference catalog file
            catname = str(cat_info['file'])
            with open(catname,'r') as cat_diff:
                lines_diff = cat_diff.readlines()
                # loop though lines in catalog files
                for i in range(len(lines_template)):
                    s_template = lines_template[i].split()
                    if len(s_template) == 0 or str.startswith(lines_template[i],'#'):
                        continue
                    # astrometry
                    ra = float(s_template[18]); dec = float(s_template[19])
                    # fluxes (use FLUX_APER measurements in template for consistency)
                    Ftmp = float(s_template[5]); Ferr_tmp = float(s_template[6])
                    # check image flux
                    s_diff = lines_diff[i].split()
                    dF = float(s_diff[5])
                    # not in a completely blank or masked region
                    if dF != 0.0 and Ftmp > 1e-30 and np.isfinite(ra) and np.isfinite(dec):
                        # get J-name and create file
                        lc_name = '%s.dat' % toIAU(ra,dec)
                        lc_name = os.path.join(self.tile_dir,lc_name)
                        # open light curve file for this object
                        with open(lc_name,'a+') as lc:
                            dFerr = float(s_diff[6])
                            F = dF+Ftmp
                            if F > 0:
                                Ferr = np.sqrt(dFerr**2+Ferr_tmp**2)
                                m = self.template_mag_zero-2.5*np.log10(F)
                                m_err = 2.5/np.log(10)*Ferr/F
                                mjd_obs = cat_info['mjd']
                                lc.write('%f %f %f %f %f %f %f\n' % (mjd_obs,m,m_err))
                                lc_out.append((lc_name,ra,dec))
        dtype = [('filename','|S200')]
        return np.array(lc_out,dtype=dtype)


    def forced_photometry(self,file_info):
        local_path = str(file_info["path"])
        mjd = file_info["mjd_obs"]
        file_root = local_path[0:-5]
        path_root = os.path.dirname(local_path)
        outfile_sci = file_root + "_proj_diff.fits"
        outfile_wgt = file_root + "_proj_diff.weight.fits"
        template_sci = os.path.join(path_root,"template.fits")
        template_wgt = os.path.join(path_root,"template.weight.fits")
        outfile_cat = file_root + "_diff.cat"
        # SExtractor double image mode
        code = bash('sex %s,%s -WEIGHT_IMAGE %s,%s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT %f %s' % (template_sci,outfile_sci,template_wgt,outfile_wgt,outfile_cat,self.sex_file,self.template_mag_zero,self.sex_pars))
        safe_rm(outfile_sci)
        safe_rm(outfile_wgt)
        if code != 0 or not os.path.exists(outfile_cat): return None
        info_list = (outfile_cat, mjd)
        dtype = [('file','|S200'),('mjd',float)]
        return np.array(info_list,dtype=dtype)


    def run(self,image_list,num_threads,tile_head,fermigrid=False):
        # given list of single-epoch image filenames in same tile or region, execute pipeline
        print('Pooling %d single-epoch images to %d threads.' % (len(image_list),num_threads))
        print('Downloading images, making weight maps and image masks.')
        file_info = clean_pool(self.download_image, image_list, num_threads)
        print("Downloaded %d images" % len(file_info))
        file_info = clean_pool(self.make_weight, file_info, num_threads)
        # combine exposures with same MJD (tile mode only)
        print('Tiling CCD images.')
        file_info = self.combine_night(file_info,tile_head,num_threads)
        if len(file_info) < self.min_epoch:
            print("Not enough epochs in tile.")
            sys.exit(0)
        # make template
        file_info_Y1 = file_info[(file_info["mjd_obs"]>56400) & (file_info["mjd_obs"]<56830)]
        file_info_Y2 = file_info[(file_info["mjd_obs"]>56830) & (file_info["mjd_obs"]<57200)]
        file_info_Y3 = file_info[(file_info["mjd_obs"]>57200) & (file_info["mjd_obs"]<57550)]
        file_info_Y4 = file_info[(file_info["mjd_obs"]>57550) & (file_info["mjd_obs"]<57930)]
        file_info_Y5 = file_info[(file_info["mjd_obs"]>57930) & (file_info["mjd_obs"]<58290)]
        file_info_Y6 = file_info[(file_info["mjd_obs"]>58290)]
        # make template from season with lowest mean sky noise
        sky_sigma_Y1 = np.mean(file_info_Y1["skysigma"])
        sky_sigma_Y2 = np.mean(file_info_Y2["skysigma"])
        sky_sigma_Y3 = np.mean(file_info_Y3["skysigma"])
        sky_sigma_Y4 = np.mean(file_info_Y4["skysigma"])
        sky_sigma_Y5 = np.mean(file_info_Y5["skysigma"])
        sky_sigma_Y6 = np.mean(file_info_Y6["skysigma"])
        file_info_seasons = [file_info_Y1,file_info_Y2,file_info_Y3,file_info_Y4,file_info_Y5,file_info_Y6]
        sky_sigma_seasons = [sky_sigma_Y1,sky_sigma_Y2,sky_sigma_Y3,sky_sigma_Y4,sky_sigma_Y5,sky_sigma_Y6]
        template_season = np.nanargmin(sky_sigma_seasons)
        print("Making template with Y%d images." % template_season)
        file_info_template = file_info_seasons[template_season]
        if self.program == "supernova":
            # select sky noise < 2.5*(min sky noise), follows Kessler et al. (2015)
            file_info_template = file_info_template[file_info_template["skysigma"]<2.5*np.min(file_info_template["skysigma"])]
            # after this constraint, use up to 10 images with smallest PSF
            file_info_template = np.sort(file_info_template,order="psf_fwhm")
            if len(file_info_template) > 10:
                file_info_template = file_info_template[:10]
        template_sci = os.path.join(self.tile_dir,'template.fits')
        template_wgt = os.path.join(self.tile_dir,'template.weight.fits')
        # get lists for template creation and projection
        swarp_temp_list = " ".join(file_info_template["path"])
        swarp_all_list = " ".join(file_info["path"])
        self.template_mag_zero = np.median(file_info_template["mag_zero"])
        self.template_sigma_mag_zero = np.median(file_info_template["sigma_mag_zero"])
        # create template (coadd of best frames)
        s = swarp_temp_list.split()
        resample_dir = os.path.dirname(template_sci)
        bash('ln -s %s %s.head' % (s[0],template_sci[0:-5]))
        bash('swarp %s -c %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS %d -RESAMPLE_DIR %s' % (swarp_temp_list,self.swarp_file,template_sci,template_wgt,num_threads,resample_dir))
        # project (re-align) images onto template
        print('Aligning images.')
        clean_pool(self.align,swarp_all_list.split(),num_threads)
        # make difference images
        print('Differencing images.')
        file_info = clean_pool(difference,file_info,num_threads)
        # forced photometry
        print('%d HOTPANTS attempts failed.' % self.num_fail)
        print('Performing forced photometry.')
        cat_list = clean_pool(self.forced_photometry,file_info,num_threads)
        # get objects from template file
        template_cat = os.path.join(self.tile_dir,'template.cat')
        bash('sex %s -WEIGHT_IMAGE %s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT %f %s' % (template_sci,template_wgt,template_cat,self.sex_file,self.template_mag_zero,self.sex_pars))
        # write lightcurve data
        print('Generating light curves.')
        lc_list = clean_pool(self.generate_light_curves,cat_list,num_threads)
        # concatenate array and remove duplicates
        if lc_list is None:
            print("No light curves found.")
            sys.exit(0)
        lc_list = np.unique(np.concatenate(lc_list))
        # clean directory
        safe_rm(template_cat)
        for cat_file in cat_list:
            safe_rm(str(cat_file['file']))
            safe_rm('%s.head' % template_sci[0:-5])
        # remove template files
        safe_rm(template_sci)
        safe_rm(template_wgt)
        safe_rm(template_sci[0:-5]+".head")
        return lc_files
