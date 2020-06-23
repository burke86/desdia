import os, sys
import numpy as np
from multiprocessing import Pool as Pool
from astropy import units as u
from astropy.coordinates import SkyCoord
from misc import *

def difference(file_info):
    # Get DES hotpants files (TEMPORARY)
    top_path = os.path.abspath(__file__)
    top_dir = '/'.join(os.path.dirname(top_path).split('/')[0:-1])
    hotpants_file = os.path.join(top_dir,'etc/DES.hotpants')
    local_path = file_info["path"]
    ccd = file_info["ccd"]
    file_root = local_path[0:-5]
    path_root = os.path.dirname(local_path)
    file_sci = file_root + "_proj.fits"
    file_wgt = file_root + "_proj.weight.fits"
    outfile_sci = file_root + "_proj_diff.fits"
    outfile_wgt = file_root + "_proj_diff.weight.fits"
    template_sci = os.path.join(path_root,"template_c%d.fits" % ccd)
    template_wgt = os.path.join(path_root,"template_c%d.weight.fits" % ccd)
    hotpants_pars = ''.join(open(hotpants_file,'r').readlines())
    # HOTPANTS input parameters
    if not os.path.exists(outfile_sci):
        command = 'hotpants -inim %s -ini %s -tmplim %s -tni %s -outim %s -oni %s -useWeight %s'
        args = (file_sci,file_wgt,template_sci,template_wgt,outfile_sci,outfile_wgt,hotpants_pars)
        code = bash(command % args)
        #safe_rm(file_sci)
        #safe_rm(file_wgt)
        # Handle HOTPANTS fatal error
        if code != 0:
            print('***HOTPANTS FATAL ERROR**')
            return None
    return


class Pipeline:

    def __init__(self,bands,usr,psw,work_dir,out_dir,top_dir=None,min_epoch=5,debug_mode=False):
        self.bands = bands
        self.usr = usr
        self.psw = psw
        self.tile_dir = work_dir
        self.out_dir = out_dir
        self.min_epoch = min_epoch
        self.debug_mode = debug_mode
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

    def download_image(self,info_list):
        # download image from image archive server
    	url = info_list['path'] # Get URL
        local_path = os.path.join(self.tile_dir,os.path.basename(url))
        if not os.path.exists(local_path):
            command = 'wget -nc --no-check-certificate -q --user %s --password %s %s -P %s'
            args = (self.usr,self.psw,url,self.tile_dir)
            bash(command % args)
        # Change to local_path
        info_list['path'] = local_path
        return info_list


    def make_weight(self,info_list):
        # get reduced images ready for generating template
        try:
            local_path = info_list["path"]
            file_root = local_path[0:-8]
            file_sci = file_root + ".fits"
            file_wgt = file_root + ".weight.fits"
            # skip if file already exists (for debugging)
            if not os.path.exists(file_sci):
                # make weight maps and mask
                code = bash('makeWeight -inFile_img %s -border 20 -outroot %s' % (local_path,file_root))
                if code != 0:
                    safe_rm(local_path, self.debug_mode)
                    return None
                # convert files to single-header format
                single_header(file_sci)
                single_header(file_wgt)
            # Change to weight file
            info_list["path"] = file_sci
            safe_rm(local_path, self.debug_mode)
            return info_list
        except:
            return None

    def make_templates(self, info_list, num_threads):
        # Use Y3 images
        ccd = info_list["ccd"][0]
        info_list_template = info_list[(info_list["mjd_obs"]>57200) & (info_list["mjd_obs"]<57550)]
        # select sky noise < 2.5*(min sky noise), follows Kessler et al. (2015)
        info_list_template = info_list_template[info_list_template["skysigma"]<2.5*np.nanmin(info_list_template["skysigma"])]
        # after this constraint, use up to 10 images with smallest PSF
        info_list_template = np.sort(info_list,order="psf_fwhm")
        if len(info_list_template) > 10:
            info_list_template = info_list_template[:10]
        template_sci = os.path.join(self.tile_dir,'template_c%d.fits' % ccd)
        template_wgt = os.path.join(self.tile_dir,'template_c%d.weight.fits' % ccd)
        template_cat = os.path.join(self.tile_dir,'template_c%d.cat' % ccd)
        # get lists for template creation and projection
        swarp_all_list = " ".join(info_list["path"])
        swarp_temp_list = " ".join(info_list_template["path"])
        # create template (coadd of best frames)
        s = swarp_temp_list.split()
        resample_dir = os.path.dirname(template_sci)
        if True: #not os.path.exists(template_sci):
            bash('ln -s %s %s.head' % (s[0],template_sci[0:-5]))
            command = 'swarp %s -c %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS 1 -RESAMPLE_DIR %s'
            args = (swarp_temp_list,self.swarp_file,template_sci,template_wgt,resample_dir)
            bash(command % args)
        # Align
        clean_tpool(self.align,swarp_all_list.split(),num_threads)
        # Extract sources
        command = 'sex %s -WEIGHT_IMAGE %s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT 22.5 %s'
        args = (template_sci,template_wgt,template_cat,self.sex_file,self.sex_pars)
        bas(command % args)
        return
    
    def align(self,filename_in):
        # project and align images to template
        file_root = filename_in[0:-5]
        path_root = os.path.dirname(filename_in)
        filename_out = file_root+"_proj.fits"
        file_header = file_root+"_proj.head"
        template_sci = os.path.join(path_root,"template.fits")
        # If per-ccd
        s = os.path.basename(filename_out).split('_c')
        if len(s) > 0:
            ccd = int(s[1][:2])
            template_sci = os.path.join(path_root,"template_%d.fits"%ccd)
        # symbolic link for header geometry
        if not os.path.exists(filename_out):
            bash('ln -s %s %s' % (template_sci,file_header))
            bash('swarp %s -c %s -NTHREADS %d -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s.weight.fits -RESAMPLE_DIR %s' % (filename_in,self.swarp_file,1,filename_out,filename_out[0:-5],path_root))
        safe_rm(filename_in, self.debug_mode)
        safe_rm(filename_in[0:-5]+".weight.fits", self.debug_mode)
        safe_rm(file_header, self.debug_mode)
        return filename_in


    def generate_light_curves(self,cat_info):
        # generates light curve files from list of sextractor catalogs
        # with forced photometry on the template image
        # measure template flux to add to difference flux
        ccd = int(cat_info['ccd'])
        template_cat = os.path.join(self.tile_dir,'template_c%d.cat'%ccd)
        diff_cat = cat_info['file']
        mjds = cat_info['mjd']
        # assumes all catalogs have same number of lines (detections)
        # template catalog
        num,ra,dec,f3,f4,f5,ferr3,ferr4,ferr5 = np.loadtxt(template_cat, unpack=True)
        num_list = []; mjd_list = []
        ra_list = []; dec_list = []
        f3_list = []; ferr3_list = []
        f4_list = []; ferr4_list = []
        f5_list = []; ferr5_list = []
        print(np.shape(f3))
        # difference catalog
        for i, diff_cat_file in enumerate(diff_cat):
            num,ra,dec,df3,df4,df5,dferr3,dferr4,dferr5 = np.loadtxt(str(diff_cat_file), unpack=True)
            mjd = np.full(len(num), mjds[i])
            # bad photometry
            df3[np.abs(df3)<1e-29] = np.nan
            df4[np.abs(df4)<1e-29] = np.nan
            df5[np.abs(df5)<1e-29] = np.nan
            # save light curves
            print(np.shape([f3,df3]))
            f3 = np.sum([f3,df3],axis=0)
            f4 = np.sum([f4,df4],axis=0)
            f5 = np.sum([f5,df5],axis=0)
            ferr3 = np.sqrt(np.sum(ferr3**2,dferr3**2),axis=0)
            ferr4 = np.sqrt(np.sum(ferr4**2,dferr4**2),axis=0)
            ferr5 = np.sqrt(np.sum(ferr5**2,dferr5**2),axis=0)
            # append arrays
            num_list.append(num)
            mjd_list.append(mjd)
            ra_list.append(ra)
            dec_list.append(dec)
            f3_list.append(f3)
            f4_list.append(f4)
            f5_list.append(f5)
            ferr3_list.append(ferr3)
            ferr4_list.append(ferr4)
            ferr5_list.append(ferr5)
        # flatten and save data
        num_list = np.array(num_list).flatten()
        mjd_list = np.array(mjd_list).flatten()
        ra_list = np.array(ra_list).flatten()
        dec_list = np.array(dec_list).flatten()
        f3_list = np.array(f3_list).flatten()
        f4_list = np.array(f4_list).flatten()
        f5_list = np.array(f5_list).flatten()
        ferr3_list = np.array(ferr3_list).flatten()
        ferr4_list = np.array(ferr4_list).flatten()
        ferr5_list = np.array(ferr5_list).flatten()
        # save
        path_root = os.path.dirname(diff_cat[0])
        path_dat = os.path.join(path_root,'cat.dat')
        dat = [num_list, mjd_list, ra_list, dec_list, m3_list, m4_list, m5_list, merr3_list, merr4_list, merr5_list]
        hdr = 'num mjd ra dec m3 m4 m5 merr3 merr4 merr5'
        np.savetxt(path_dat,np.array(dat).T,fmt='%d %f %f %f %f %f %f %f %f %f',header=hdr)
        safe_rm(template_cat, self.debug_mode)
        [safe_rm(str(i), self.debug_mode) for i in diff_cat]
        return


    def forced_photometry(self,info_list):
        local_path = info_list["path"]
        mjd = info_list["mjd_obs"]
        ccd = info_list["ccd"]
        file_root = local_path[0:-5]
        path_root = os.path.dirname(local_path)
        outfile_sci = file_root + "_proj_diff.fits"
        outfile_wgt = file_root + "_proj_diff.weight.fits"
        template_sci = os.path.join(path_root,"template_%d.fits"%ccd)
        template_wgt = os.path.join(path_root,"template_%d.weight.fits"%ccd)
        outfile_cat = file_root + "_diff.cat"
        # SExtractor double image mode
        command = 'sex %s,%s -WEIGHT_IMAGE %s,%s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT 22.5 %s'
        args = (template_sci,outfile_sci,template_wgt,outfile_wgt,outfile_cat,self.sex_file,self.sex_pars)
        bash(command % args)
        safe_rm(outfile_sci, self.debug_mode)
        safe_rm(outfile_wgt, self.debug_mode)
        if code != 0 or not os.path.exists(outfile_cat): return None
        # Update to catalog file
        info_list["path"] = outfile_cat
        return info_list

    def run_ccd(self,image_list,num_threads,fermigrid=False):
        # given list of single-epoch image filenames in same tile or region, execute pipeline
        print('Pooling %d single-epoch images to %d threads.' % (len(image_list),num_threads))
        print('Downloading images, making weight maps and image masks.')
        file_info = clean_tpool(self.download_image, image_list, num_threads)
        print("Downloaded %d images" % len(file_info))
        file_info = clean_tpool(self.make_weight, file_info, num_threads)
        print('Making templates and aligning frames.')
        # CCD loop
        for ccd in np.sort(np.unique(file_info['ccd'])):
            print('Running CCD %d.' % ccd)
            file_info = file_info[file_info['ccd']==ccd]
            self.make_templates(file_info,num_threads)
            # make difference images
            print('Differencing images.')
            clean_pool(difference,file_info,num_threads)
            # forced photometry
            print('Performing forced photometry.')
            file_info = clean_tpool(self.forced_photometry,file_info,num_threads)
            # write lightcurve data
            print('Generating light curves.')
            self.generate_light_curves(file_info)
            # clean directory
            safe_rm(template_cat, self.debug_mode)
            for cat_file in cat_list:
                safe_rm(str(cat_file['file']))
                safe_rm('%s.head' % template_sci[0:-5])
            # remove template files
            #safe_rm(template_sci, self.debug_mode)
            #safe_rm(template_wgt, self.debug_mode)
            #safe_rm(template_sci[0:-5]+".head", self.debug_mode)
            return

    def run(self,image_list,num_threads,tile_head,fermigrid=False):
        # given list of single-epoch image filenames in same tile or region, execute pipeline
        print('Pooling %d single-epoch images to %d threads.' % (len(image_list),num_threads))
        print('Downloading images, making weight maps and image masks.')
        file_info = clean_tpool(self.download_image, image_list, num_threads)
        print("Downloaded %d images" % len(file_info))
        file_info = clean_tpool(self.make_weight, file_info, num_threads)
        # combine exposures with same MJD (tile mode only)
        print('Tiling CCD images.')
        file_info = self.combine_night(file_info,tile_head,num_threads)
        if len(file_info) < self.min_epoch:
            print("Not enough epochs in tile.")
            sys.exit(0)
        # SPLIT INTO REGIONS
        # make template from Y3
        # for i in regions:
        file_info_template = file_info[(file_info["mjd_obs"]>57200) & (file_info["mjd_obs"]<57550)]
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
        clean_tpool(self.align,swarp_all_list.split(),num_threads)
        # make difference images
        print('Differencing images.')
        clean_pool(difference,file_info,num_threads)
        # forced photometry
        print('Performing forced photometry.')
        cat_list = clean_tpool(self.forced_photometry,file_info,num_threads)
        # get objects from template file
        template_cat = os.path.join(self.tile_dir,'template.cat')
        bash('sex %s -WEIGHT_IMAGE %s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT %f %s' % (template_sci,template_wgt,template_cat,self.sex_file,self.template_mag_zero,self.sex_pars))
        # write lightcurve data
        print('Generating light curves.')
        self.generate_light_curves(cat_list)
        # clean directory
        safe_rm(template_cat, self.debug_mode)
        for cat_file in cat_list:
            safe_rm(str(cat_file['file']))
            safe_rm('%s.head' % template_sci[0:-5])
        # remove template files
        safe_rm(template_sci, self.debug_mode)
        safe_rm(template_wgt, self.debug_mode)
        safe_rm(template_sci[0:-5]+".head", self.debug_mode)
        return
