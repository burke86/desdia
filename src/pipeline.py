import os, sys
import numpy as np
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
    file_sci = file_root + ".fits"
    file_wgt = file_root + ".weight.fits"
    outfile_sci = file_root + "_template_c%d" % ccd + "_diff.fits"
    outfile_wgt = file_root + "_template_c%d" % ccd + "_diff.weight.fits"
    template_sci = os.path.join(path_root,"template_c%d.fits" % ccd)
    template_wgt = os.path.join(path_root,"template_c%d.weight.fits" % ccd)
    hotpants_pars = ''.join(open(hotpants_file,'r').readlines())
    # HOTPANTS input parameters
    if not os.path.exists(outfile_sci):
        command = 'hotpants -inim %s -ini %s -tmplim %s -tni %s -outim %s -oni %s -useWeight %s'
        args = (file_sci,file_wgt,template_sci,template_wgt,outfile_sci,outfile_wgt,hotpants_pars)
        code = bash(command % args)
        # Handle HOTPANTS fatal error
        if code != 0:
            print('***HOTPANTS FATAL ERROR**')
            # Create a dummy file if it failed
            with open(outfile_sci, 'w'): pass
            return None
    else:
        print('***Difference image exists**')
    return


class Pipeline:

    def __init__(self,bands,usr,psw,work_dir,top_dir=None,debug_mode=False):
        self.bands = bands
        self.usr = usr
        self.psw = psw
        self.tile_dir = work_dir
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
    

    def download_image(self,info_list):
        # download image from image archive server
        url = info_list['path'] # Get URL
        local_path = os.path.join(self.tile_dir,os.path.basename(url))
        if not os.path.exists(local_path[:-3]):
            command = 'wget -nc --no-check-certificate -q --user %s --password %s %s -P %s'
            args = (self.usr,self.psw,url,self.tile_dir)
            bash(command % args)
        else:
            print('***Image exists***')
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
            else:
                print('***Weight image exists***')
            # Change to weight file
            info_list["path"] = file_sci
            safe_rm(local_path, self.debug_mode)
            return info_list
        except:
            return None
    
        
    def make_coadd_diff(self, info_list, num_threads=1):
        # Use Y3 images
        ccd = info_list["ccd"][0]
        if len(info_list) == 0:
            print("No images to generate coadd!")
            return 1
        coadd_sci = os.path.join(self.tile_dir,'coadd_diff_c%d.fits' % ccd)
        coadd_wgt = os.path.join(self.tile_dir,'coadd_diff_c%d.weight.fits' % ccd)
        # get lists for template creation and projection
        info_list_abs = []
        # Take absolute value of image
        for i, f in enumerate(info_list["path"]):
            try:
                with fits.open(f[:-4]+".fits") as hdul:
                    hdu = hdul[0]
                    hdu = fits.PrimaryHDU(np.abs(hdu.data), hdu.header)
                    f_abs = f[:-4] + "_abs.fits"
                    hdu.writeto(f_abs, clobber=True)
                    info_list_abs.append(f_abs)
                    bash('ln -s %s %s' % (f[:-4]+".weight.fits", f[:-4]+"_abs.weight.fits"))
            except:
                pass
        # Create swarp list
        swarp_all_list = " ".join(info_list_abs)
        # create coadd
        s = swarp_all_list.split()
        resample_dir = os.path.dirname(coadd_sci)
        if not os.path.exists(coadd_sci):
            bash('ln -s %s %s.head' % (s[0], coadd_sci[0:-5]))
            command = 'swarp %s -c %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS %d -RESAMPLE_DIR %s'
            args = (swarp_all_list,self.swarp_file,coadd_sci,coadd_wgt,num_threads,resample_dir)
            bash(command % args)
        return 0
    

    def make_template(self, info_list, sn=True, season=6, num_threads=1):
        # Use Y3 images
        ccd = info_list["ccd"][0]
        t0_Y6 = 58250
        t1_Y6 = 58615
        t0 = t0_Y6 + 365*(season - 6)
        t1 = t1_Y6 + 365*(season - 6)
        info_list_template = info_list[(info_list["mjd_obs"] > t0) & (info_list["mjd_obs"] < t1)] # Y6
        if len(info_list_template) ==  0:
            print("No Y%d images to generate template!" % season)
            return 1
        if sn:
            # select sky noise < 2.5*(min sky noise), follows Kessler et al. (2015)
            info_list_template = info_list_template[info_list_template["skysigma"]<2.5*np.nanmin(info_list_template["skysigma"])]
        # after this constraint, use up to 10 images with smallest PSF
        info_list_template = np.sort(info_list,order="psf_fwhm")
        if len(info_list_template) > 10:
            info_list_template = info_list_template[:10]
        elif len(info_list_template) ==  0:
            print("Insufficient images to generate template!")
            return 2
        template_sci = os.path.join(self.tile_dir,'template_c%d.fits' % ccd)
        template_wgt = os.path.join(self.tile_dir,'template_c%d.weight.fits' % ccd)
        template_cat = os.path.join(self.tile_dir,'template_c%d.cat' % ccd)
        # get lists for template creation and projection
        swarp_all_list = " ".join(info_list["path"])
        swarp_temp_list = " ".join(info_list_template["path"])
        # create template (coadd of best frames)
        s = swarp_temp_list.split()
        resample_dir = os.path.dirname(template_sci)
        if not os.path.exists(template_sci):
            bash('ln -s %s %s.head' % (s[0],template_sci[0:-5]))
            command = 'swarp %s -c %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS %d -RESAMPLE_DIR %s'
            args = (swarp_temp_list,self.swarp_file,template_sci,template_wgt,num_threads,resample_dir)
            bash(command % args)
        # Align
        info_list_swarp = info_list
        info_list_swarp['path'] = swarp_all_list.split()
        clean_tpool(self.align,info_list_swarp,num_threads)
        # Extract sources
        command = 'sex %s -WEIGHT_IMAGE %s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT 22.5 %s'
        args = (template_sci,template_wgt,template_cat,self.sex_file,self.sex_pars)
        bash(command % args)
        return 0
    
    
    def align(self,info_list):
        filename_in = info_list["path"]
        ccd = info_list["ccd"]
        # project and align images to template
        file_root = filename_in[0:-5]
        path_root = os.path.dirname(filename_in)
        filename_out = file_root+"_proj.fits"
        file_header = file_root+"_proj.head"
        template_sci = os.path.join(path_root,"template_c%d.fits"%ccd)
        # symbolic link for header geometry
        if not os.path.exists(filename_out):
            bash('ln -s %s %s' % (template_sci,file_header))
            # This filename_in.head file should be the ouput WCS!!
            bash('swarp %s -c %s -NTHREADS %d -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s.weight.fits -RESAMPLE_DIR %s' % (filename_in,self.swarp_file,1,filename_out,filename_out[0:-5],path_root), True)
            # Should get a warning that it is using .head
        else:
            print('out file exists')
        info_list["path"] = filename_out
        return info_list
    

    def generate_light_curves(self,info_list):
        # generates light curve files from list of sextractor catalogs
        # with forced photometry on the template image
        # measure template flux to add to difference flux
        ccd = info_list['ccd'][0]
        template_cat = os.path.join(self.tile_dir,'template_c%d.cat'%ccd)
        diff_cat = info_list['path']
        mjds = info_list['mjd_obs']
        # assumes all catalogs have same number of lines (detections)
        # template catalog
        # Yes, this code is horrifying...
        num,ra,dec,f3_temp,f4_temp,f5_temp,ferr3_temp,ferr4_temp,ferr5_temp = np.loadtxt(template_cat, unpack=True)
        num_list = []; mjd_list = []
        ra_list = []; dec_list = []
        f3_list = []; ferr3_list = []
        f4_list = []; ferr4_list = []
        f5_list = []; ferr5_list = []
        # difference catalog
        for i, diff_cat_file in enumerate(diff_cat):
            try:
                num,ra,dec,df3,df4,df5,dferr3,dferr4,dferr5 = np.loadtxt(str(diff_cat_file), unpack=True)
            except:
                continue
            mjd = np.full(len(num), mjds[i])
            # Bad photometry
            df3[np.abs(df3)<1e-29] = np.nan
            df4[np.abs(df4)<1e-29] = np.nan
            df5[np.abs(df5)<1e-29] = np.nan
            # Save light curves
            f3 = np.sum([f3_temp,df3],axis=0)
            f4 = np.sum([f4_temp,df4],axis=0)
            f5 = np.sum([f5_temp,df5],axis=0)
            # Add errors in quadrature
            ferr3 = np.sqrt(np.sum([ferr3_temp**2,dferr3**2],axis=0))
            ferr4 = np.sqrt(np.sum([ferr4_temp**2,dferr4**2],axis=0))
            ferr5 = np.sqrt(np.sum([ferr5_temp**2,dferr5**2],axis=0))
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
            safe_rm(diff_cat, self.debug_mode)
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
        path_dat = os.path.join(path_root,'cat_c%d.dat' % ccd)
        dat = [num_list, mjd_list, ra_list, dec_list, f3_list, f4_list, f5_list, ferr3_list, ferr4_list, ferr5_list]
        hdr = 'num mjd_obs ra dec flux3 flux4 flux5 fluxerr3 fluxerr4 fluxerr5'
        print('Saving catalog as %s' % path_dat)
        np.savetxt(path_dat,np.array(dat).T,fmt='%d %f %f %f %f %f %f %f %f %f',header=hdr)
        safe_rm(template_cat, self.debug_mode)
        safe_rm('%s.head' % template_cat[0:-5], self.debug_mode)
        return


    def forced_photometry(self,info_list):
        local_path = info_list["path"]
        mjd = info_list["mjd_obs"]
        ccd = info_list["ccd"]
        file_root = local_path[0:-5]
        path_root = os.path.dirname(local_path)
        outfile_sci = file_root + "_template_c%d" % ccd + "_diff.fits"
        outfile_wgt = file_root + "_template_c%d" % ccd + "_diff.weight.fits"
        template_sci = os.path.join(path_root,"template_c%d.fits" % ccd)
        template_wgt = os.path.join(path_root,"template_c%d.weight.fits" % ccd)
        outfile_cat = file_root + "_template_c%d" % ccd + "_diff.cat"
        code = 0
        # SExtractor double image mode
        if not os.path.exists(outfile_cat):
            command = 'sex %s,%s -WEIGHT_IMAGE %s,%s  -CATALOG_NAME %s -c %s -MAG_ZEROPOINT 30.753 %s'
            args = (template_sci,outfile_sci,template_wgt,outfile_wgt,outfile_cat,self.sex_file,self.sex_pars)
            code = bash(command % args)
        if code != 0 or not os.path.exists(outfile_cat): return None
        # Update to catalog file
        info_list["path"] = outfile_cat
        return info_list
    

    def run_ccd_sn(self,image_list,num_threads=1,template_season=6,fermigrid=False):
        # given list of single-epoch image filenames in same tile or region, execute pipeline
        print('Pooling %d single-epoch images to %d threads.' % (len(image_list),num_threads))
        print('Downloading images, making weight maps and image masks.')
        file_info_all = clean_tpool(self.download_image, image_list, num_threads)
        print("Downloaded %d images" % len(file_info_all))
        file_info_all = clean_tpool(self.make_weight, file_info_all, num_threads)
        print('Making templates and aligning frames.')
        # CCD loop
        for ccd in np.sort(np.unique(file_info_all['ccd'])):
            print('Running CCD %d.' % ccd)
            file_info = file_info_all[file_info_all['ccd']==ccd]
            if len(file_info) == 0: continue
            code = self.make_template(file_info,sn=True,season=template_season,num_threads=num_threads)
            if code != 0: continue
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
        return
    

    def run_ccd_survey(self,image_list,query_sci,num_threads=1,template_season=6,fermigrid=False,band='g',coadd_diff=False,offset=False):
        # given list of single-epoch image filenames in same pointing, execute pipeline
        print('Pooling %d single-epoch images to %d threads.' % (len(image_list),num_threads))
        print('Downloading images, making weight maps and image masks.')
        file_info_all = clean_tpool(self.download_image, image_list, num_threads)
        print("Downloaded %d images" % len(file_info_all))
        file_info_all = clean_tpool(self.make_weight, file_info_all, num_threads)
        print('Making templates and aligning frames.')
        # CCD loop in template list
        print(file_info_all['ccd'])
        print(np.sort(np.unique(file_info_all['ccd'])))
        for ccd in np.sort(np.unique(file_info_all['ccd'])):
            print('Running CCD %d.' % ccd)
            file_info_template = file_info_all[file_info_all['ccd']==ccd]
            if len(file_info_template) == 0: continue
            print('Making template')
            code = self.make_template(file_info_template,sn=False,season=template_season,num_threads=num_threads)
            if code != 0: continue
            print('Querying overlapping CCD images.')
            # Now query images which overlap with this template CCD
            # (dithering prevents us from using same CCD ID as in SN fields)
            image_list = query_sci.get_image_info_overlap(file_info_template['ramin'][0],file_info_template['ramax'][0],
                                                         file_info_template['decmin'][0],file_info_template['decmax'][0],
                                                         band=band)
            if image_list is None:
                print('***Error: No overlapping images found.***')
                return
            print('Downloading images, making weight maps and image masks.')
            file_info = clean_tpool(self.download_image,image_list,num_threads)
            print("Downloaded %d images" % len(file_info))
            file_info = clean_tpool(self.make_weight,file_info,num_threads)
            # Small hack to set the image CCD to the template's CCD
            file_info['ccd'] = np.full(len(file_info), ccd)
            print('Aligning frames.')
            info_list = clean_tpool(self.align,file_info,num_threads=num_threads)
            # make difference images
            # we should be good with this! No need to combine ones taken on the same night anymore ;)
            # they will come through nicely by sorting the catalogs afterwords
            print('Differencing images.')
            clean_pool(difference,file_info,num_threads)
            # forced photometry
            print('Performing forced photometry.')
            file_info = clean_tpool(self.forced_photometry,file_info,num_threads)
            # write lightcurve data
	    # seems to be different numbers of detections in template and diff images for some reason
            if offset==False:
                print('Generating light curves.')
                self.generate_light_curves(file_info)
                # TODO clean directory
            # Coadd difference frames
            if coadd_diff:
                print('Making coadd of difference frames.')
                self.make_coadd_diff(file_info,num_threads=num_threads)
        return
    
