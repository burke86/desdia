import os
import despydb.desdbi as desdbi
import numpy as np
import argparse

class Query:

    def __init__(self,section="db-dessci",user="local"):
        # read des services file for database connection
        self.desdmfile = os.environ["DES_SERVICES"]
        self.user = user.lower()
        with open(self.desdmfile,"r") as f:
            contents = f.read()
            s = contents.split('[%s]' % section)[1].replace("\n"," ")
            self.usr = s.split("user")[1].split("=")[1].split()[0]
            self.psw = s.split("passwd")[1].split("=")[1].split()[0]
        if user.lower() == 'decade':
            s = contents.split('[%s]' % "db-decade")[1].replace("\n"," ")
            self.usr = s.split("user")[1].split("=")[1].split()[0]
            self.psw = s.split("passwd")[1].split("=")[1].split()[0]
            self.base_url = "https://decade.ncsa.illinois.edu/deca_archive/"
        else:
            self.base_url = "https://desar2.cosmology.illinois.edu/DESFiles/desarchive/"
        # start connection
        self.con = desdbi.DesDbi(self.desdmfile,section)
        self.cur = self.con.cursor()
        

    def get_filenames_from_object(self,ra,dec,band,window_radius=10):
        # get reduced filename from RA, DEC of object
        dec_radian = dec*np.pi/180.
        ra_upper = (ra+window_radius/3600./np.cos(dec_radian))
        ra_lower = (ra-window_radius/3600./np.cos(dec_radian))
        dec_upper = dec+window_radius/3600.
        dec_lower = dec-window_radius/3600.
        # get Y1-Y5 filenames
        get_list = "select distinct filename from Y6A1_FINALCUT_OBJECT where RA between :ra_lower and :ra_upper and DEC between :dec_lower and :dec_upper and band=:band and flags=0 and RA between :ra_lower and :ra_upper and DEC between :dec_lower and :dec_upper and band=:band and flags=0 order by filename"
        self.cur.execute(get_list,ra_lower=ra_lower,ra_upper=ra_upper,dec_lower=dec_lower,dec_upper=dec_upper,band=band)
        info_list = self.cur.fetchall()
        # get SV filenames
        get_list = "select distinct filename from Y4A1_FINALCUT_OBJECT where RA between :ra_lower and :ra_upper and DEC between :dec_lower and :dec_upper and band=:band and flags=0 and RA between :ra_lower and :ra_upper and DEC between :dec_lower and :dec_upper and band=:band and flags=0 order by filename"
        self.cur.execute(get_list,ra_lower=ra_lower,ra_upper=ra_upper,dec_lower=dec_lower,dec_upper=dec_upper,band=band)
        info_list += self.cur.fetchall()
        if len(info_list) > 0:
            # create np stuctured array
            dtype = [("filename","|S41")]
            filename_list = np.array(info_list,dtype=dtype)
            return filename_list
        else:
            return None

    
    def get_pointing_coord(self,ra,dec,band='g',season=6,field=None):
        # Get list of unique Y? pointing of target given RA and dec in the main DES survey
        t0_Y6 = 58250
        t1_Y6 = 58615
        t0 = t0_Y6 + 365*(season - 6)
        t1 = t1_Y6 + 365*(season - 6)
        if self.user == 'local':
            get_list = "select unique e.tradeg, e.tdecdeg, e.mjd_obs, i.ccdnum from y6a1_image i, y6a1_exposure e where e.mjd_obs>:t0 and e.mjd_obs<:t1 and e.PROGRAM='survey' and i.expnum=e.expnum and (:ra between i.racmin and i.racmax) and (:dec between i.deccmin and i.deccmax) and i.band=:band"
            self.cur.execute(get_list,ra=ra,dec=dec,t0=t0,t1=t1,band=band)
        elif self.user == 'decade':
            get_list = "select unique e.tradeg, e.tdecdeg, e.mjd_obs, i.ccdnum from DECADE.IMAGE i, DECADE.EXPOSURE e where e.mjd_obs>:t0 and e.mjd_obs<:t1 and e.OBJECT=:field and i.expnum=e.expnum and (:ra between i.racmin and i.racmax) and (:dec between i.deccmin and i.deccmax) and i.band=:band"
            self.cur.execute(get_list,field=field,ra=ra,dec=dec,t0=t0,t1=t1,band=band)
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            # Use the first pointing if there are multiple overlapping
            #if len(info_list) > 1:
            #    info_list = info_list[0]
            dtype = [("TRADEG",float),("TDECDEG",float),("mjd_obs",float),('ccd',float)]
            info_list = np.array(info_list,dtype=dtype)
            return info_list
        else:
            return None
        
    
    def get_image_info_pointing(self,tra,tdec,mjd_obs,band='g',field=None):
        # Get image archive info (URL) from specific pointing and MJD (to generate templates)
        # Get Y1-Y6 images
        if self.user == 'local':
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, i.racmin, i.racmax, i.deccmin, i.deccmax from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.TRADEG=:tra and e.TDECDEG=:tdec and i.band=:band and e.mjd_obs=:mjd_obs and z.version=:version"
            self.cur.execute(get_list,tra=tra,tdec=tdec,band=band,mjd_obs=mjd_obs,version="y6a1_v2.1")
        elif self.user == 'decade':
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, i.racmin, i.racmax, i.deccmin, i.deccmax from DECADE.FILE_ARCHIVE_INFO f, DECADE.IMAGE i, DECADE.EXPOSURE e, DECADE.QA_SUMMARY s where i.filetype='red_immask' and f.filename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.TRADEG=:tra and e.TDECDEG=:tdec and i.band=:band and e.mjd_obs=:mjd_obs and e.OBJECT=:field"
            self.cur.execute(get_list,tra=tra,tdec=tdec,band=band,mjd_obs=mjd_obs,field=field)
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            dtype = [("filename","|S41"),("path","|S200"),("compression","|S4"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),("ramin",float),("ramax",float),("decmin",float),("decmax",float)]
            info_list = np.array(info_list,dtype=dtype)
            # Form URL and data type
            ccd_list = [f["filename"].split('_c')[1][:2] for f in info_list]
            url_list = [self.base_url+f["path"]+"/"+f["filename"]+f["compression"] for f in info_list]
            dtype = [("path","|S300"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),("ccd",int),("ramin",float),("ramax",float),("decmin",float),("decmax",float)]
            info_list = list(zip(url_list,info_list["psf_fwhm"],info_list["skysigma"],info_list["mjd_obs"],ccd_list,info_list["ramin"],info_list["ramax"],info_list["decmin"],info_list["decmax"]))
            info_list = np.array(info_list,dtype=dtype)
            return info_list
        else:
            return None
    
    
    def get_image_info_overlap(self,ramin,ramax,decmin,decmax,band='g'):
        # TODO: EDGE CASE WHEN  RACROSS0
        # Get image archive info (URL) at main survey telescope pointing
        # Get Y1-Y6 images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and \
        (i.racmin between :ramin and :ramax or i.racmax between :ramin and :ramax) and \
        (i.deccmin between :decmin and :decmax or i.deccmax between :decmin and :decmax) \
        and i.band=:band and z.version=:version and e.mjd_obs>56400"
        self.cur.execute(get_list,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax,band=band,version="y6a1_v2.1")
        info_list = self.cur.fetchall()
        # get SV images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and \
        (i.racmin between :ramin and :ramax or i.racmax between :ramin and :ramax) and \
        (i.deccmin between :decmin and :decmax or i.deccmax between :decmin and :decmax) \
        and i.band=:band and z.version=:version and e.mjd_obs>56400"
        self.cur.execute(get_list,ramin=ramin,ramax=ramax,decmin=decmin,decmax=decmax,band=band,version="v2.0")
        info_list += self.cur.fetchall()
        if len(info_list) > 0:
            dtype = [("filename","|S41"),("path","|S200"),("compression","|S4"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float)]
            info_list = np.array(info_list,dtype=dtype)
            # Form URL and data type
            ccd_list = [f["filename"].split('_c')[1][:2] for f in info_list]
            url_list = [self.base_url+f["path"]+"/"+f["filename"]+f["compression"] for f in info_list]
            dtype = [("path","|S300"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),("ccd",int)]
            info_list = list(zip(url_list,info_list["psf_fwhm"],info_list["skysigma"],info_list["mjd_obs"],ccd_list))
            info_list = np.array(info_list,dtype=dtype)
            return info_list
        else:
            return None
            

    def get_image_info_field(self,field,band='g'):
        # get image archive info (URL) from SN-field name or COSMOS
        if field.lower() == "cosmos":
            decmin, decmax = 1.22, 3.20
            ramin, ramax = 149.03, 151.21
            # get Y1-Y6 images
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and i.dec_cent between :decmin and :decmax and i.ra_cent between :ramin and :ramax and i.band=:band and z.version=:version and e.mjd_obs>56400"
            self.cur.execute(get_list,decmin=decmin,decmax=decmax,ramin=ramin,ramax=ramax,band=band,version="y6a1_v2.1")
            info_list = self.cur.fetchall()
            # get SV images
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y4a1_file_archive_info f, y4a1_image i, y4a1_exposure e, y4a1_qa_summary s, y4a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and i.dec_cent between :decmin and :decmax and i.ra_cent between :ramin and :ramax and i.band=:band and z.version=:version and e.mjd_obs<56400"
            self.cur.execute(get_list,decmin=decmin,decmax=decmax,ramin=ramin,ramax=ramax,band=band,version="v2.0")
            info_list += self.cur.fetchall()
        elif field.startswith('SN-'): # Should start with "SN-*"
            # get Y1-Y6 images
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.field=:field and e.program='supernova' and i.band=:band and z.version=:version and e.mjd_obs>56400"
            self.cur.execute(get_list,field=field,band=band,version="y6a1_v2.1")
            info_list = self.cur.fetchall()
            # get SV images
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from y4a1_file_archive_info f, y4a1_image i, y4a1_exposure e, y4a1_qa_summary s, y4a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.field=:field and e.program='supernova' and i.band=:band and z.version=:version and e.mjd_obs<56400"
            self.cur.execute(get_list,field=field,band=band,version="v2.0")
            info_list += self.cur.fetchall()
        else: # Yue's follow up programs of special fields
            # get images
            get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs from DECADE.FILE_ARCHIVE_INFO f, DECADE.IMAGE i, DECADE.EXPOSURE e, DECADE.QA_SUMMARY s, DECADE.CATALOG c where c.expnum in (select expnum from DECADE.EXPOSURE where propid in ('2019A-0065','2019B-0219','2019B-0304','2019B-1005','2019B-1011','2019B-0910','2021A-0037','2021A-0113','2021B-0038') and obstype='object') and i.filetype='red_immask' and f.filename=i.filename and c.expnum=i.expnum and i.expnum=e.expnum and e.expnum=s.expnum and i.band=:band and e.OBJECT=:field"
            self.cur.execute(get_list,band=band,field=field)
            info_list = self.cur.fetchall()
        # TODO: Search misc fields
        if len(info_list) > 0:
            dtype = [("filename","|S41"),("path","|S200"),("compression","|S4"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float)]
            info_list = np.array(info_list,dtype=dtype)
            # Form URL and data type
            ccd_list = [f["filename"].split('_c')[1][:2] for f in info_list]
            info_list["compression"] = ["" if f["compression"]=='None' else f["compression"] for f in info_list]
            url_list = [self.base_url+f["path"]+"/"+f["filename"]+f["compression"] for f in info_list]
            dtype = [("path","|S300"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),("ccd",int)]
            info_list = list(zip(url_list,info_list["psf_fwhm"],info_list["skysigma"],info_list["mjd_obs"],ccd_list))
            info_list = np.array(info_list,dtype=dtype)
            return info_list
        else:
            return None
