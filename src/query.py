import os
import despydb.desdbi as desdbi
import numpy as np
import argparse

class Query:

    def __init__(self,section="db-dessci"):
        # read des services file doe database connection
        self.desdmfile = os.environ["DES_SERVICES"]
        f = open(self.desdmfile,"r")
        contents = f.read()
        s = contents.split('[%s]' % section)[1].replace("\n"," ")
        self.usr = s.split("user")[1].split("=")[1].split()[0]
        self.psw = s.split("passwd")[1].split("=")[1].split()[0]
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


    def get_filenames_from_tile(self,tilename,band):
        # get filenames from tilename
        band = '%_'+str(band)+'_%'
        get_list = "select distinct filename from Y6A1_IMAGE_TO_TILE where tilename=:tilename and filename like :band"
        self.cur.execute(get_list,tilename=tilename,band=band)
        info_list = self.cur.fetchall()
        # SV
        get_list = "select distinct filename from Y4A1_IMAGE_TO_TILE where tilename=:tilename and filename like :band"
        self.cur.execute(get_list,tilename=tilename,band=band)
        info_list += self.cur.fetchall()
        if len(info_list) > 0:
            # create np stuctured array
            dtype = [("filename","|S41")]
            filename_list = np.array(info_list,dtype=dtype)
            filename_list = np.unique(filename_list)
            return filename_list
        else:
            return None


    def get_all_tilenames(self):
        # get tilenames for whole survey
        get_list = "select distinct tilename from Y6A1_IMAGE_TO_TILE"
        self.cur.execute(get_list)
        info_list = self.cur.fetchall()
        get_list = "select distinct tilename from Y4A1_IMAGE_TO_TILE"
        self.cur.execute(get_list)
        info_list += self.cur.fetchall()
        dtype = [("tilename","|S41")]
        tilename_list = np.array(info_list,dtype=dtype)
        tilename_list = np.unique(tilename_list)
        # save for future use
        return tilename_list


    def get_image_info_field(self,band='g',field='SN-C3'):
        # get Y1-Y6 images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, e.nite, z.mag_zero, z.sigma_mag_zero from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.field=:field and e.program='supernova' and i.band=:band and z.version=:version and e.mjd_obs>56400"
        self.cur.execute(get_list,field=field,band=band,version="y6a1_v2.1")
        info_list = self.cur.fetchall()
        # get SV images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, e.nite, z.mag_zero, z.sigma_mag_zero from y4a1_file_archive_info f, y4a1_image i, y4a1_exposure e, y4a1_qa_summary s, y4a1_zeropoint z where i.filetype='red_immask' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and e.field=:field and e.program='supernova' and i.band=:band and z.version=:version and e.mjd_obs<56400"
        self.cur.execute(get_list,field=field,band=band,version="v2.0")
        info_list += self.cur.fetchall()
        # TODO: Search misc fields
        if len(info_list) > 0:
            dtype_info = [("filename","|S41"),("path","|S200"),("compression","|S4"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),('nite',int),('mag_zero',float),('sigma_mag_zero',float)]
            info_list = np.array(info_list,dtype=dtype_info)
            return info_list
        else:
            return None


    def get_image_info(self,file_info,program="any"):
        # get image archive info from filename from program
        if program == "survey":
            get_program = "and e.program!='supernova'"
        elif program == "supernova":
            get_program = "and e.program='supernova'"
        elif program == "any":
            get_program = ""
        # get Y1-Y6 images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, e.nite, z.mag_zero, z.sigma_mag_zero from y6a1_file_archive_info f, y6a1_image i, y6a1_exposure e, y6a1_qa_summary s, y6a1_zeropoint z where "
        for item in file_info:
            filename = item["filename"]
            s = filename.split('_')
            masked_filename = s[0]+"_"+s[1]+"_"+s[2]+"_"+s[3]+"_immasked.fits"
            get_list += ("(f.filename='%s' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and z.version=:version and e.mjd_obs>56400 %s) or " % (masked_filename,get_program))
        get_list = get_list[0:-4]
        self.cur.execute(get_list,version="y6a1_v2.1")
        info_list = self.cur.fetchall()
        # get SV images
        get_list = "select f.filename, f.path, f.compression, s.psf_fwhm, i.skysigma, e.mjd_obs, e.nite, z.mag_zero, z.sigma_mag_zero from y4a1_file_archive_info f, y4a1_image i, y4a1_exposure e, y4a1_qa_summary s, y4a1_zeropoint z where "
        for item in file_info:
            filename = item["filename"]
            s = filename.split('_')
            masked_filename = s[0]+"_"+s[1]+"_"+s[2]+"_"+s[3]+"_immasked.fits"
            get_list += ("(f.filename='%s' and f.filename=i.filename and z.imagename=i.filename and i.expnum=e.expnum and e.expnum=s.expnum and z.version=:version and e.mjd_obs<56400 %s) or " % (masked_filename,get_program))
        get_list = get_list[0:-4]
        self.cur.execute(get_list,version="v2.0")
        info_list += self.cur.fetchall()
        if len(info_list) > 0:
            dtype_info = [("filename","|S41"),("path","|S200"),("compression","|S4"),("psf_fwhm",float),("skysigma",float),("mjd_obs",float),('nite',int),('mag_zero',float),('sigma_mag_zero',float)]
            info_list = np.array(info_list,dtype=dtype_info)
            return info_list
        else:
            return None

    def get_tile_head(self,tilename,band):
        get_list = "select distinct CD1_1, CD1_2, CD2_1, CD2_2, CRPIX1, CRPIX2, CRVAL1, CRVAL2, CTYPE1, CTYPE2, PIXELSCALE, NAXIS1, NAXIS2, RA_CENT, DEC_CENT, RA_SIZE, DEC_SIZE from Y3A2_COADDTILE_GEOM where tilename=:tilename"
        self.cur.execute(get_list,tilename=tilename)
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            dtype = [('CD1_1',float),('CD1_2',float),('CD2_1',float),('CD2_2',float),('CRPIX1',float),('CRPIX2',float),('CRVAL1',float),('CRVAL2',float),('CTYPE1','|S8'),('CTYPE2','|S8'),('PIXELSCALE',float),('NAXIS1',int),('NAXIS2',int),('RA_CENT',float),('DEC_CENT',float),('RA_SIZE',float),('DEC_SIZE',float)]
            tilehead_list = np.array(info_list,dtype=dtype)
            return tilehead_list
        else:
            return None

    def get_coadd_info(self,tilename):
        get_list = 'select c.FILENAME, f.PATH, f.COMPRESSION from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f where prod.PROCTAG.TAG=:tag and c.PFW_ATTEMPT_ID=PROCTAG.PFW_ATTEMPT_ID and f.FILENAME=c.FILENAME and c.TILENAME=:tilename and (c.band=:bandr or c.band=:bandg or c.band=:bandb)'
        self.cur.execute(get_list,tag='Y3A1_COADD',tilename=tilename,bandr='i',bandg='r',bandb='g')
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            dtype_info = [("filename","|S41"),("path","|S200"),("compression","|S4")]
            info_list = np.array(info_list,dtype=dtype_info)
            return info_list
        else:
            return None

    def get_tilename_from_object(self,ra,dec):
        # use objects then get image
        # ra dec in deg
        window_radius = 10 # arcseconds
        dec_radian = dec*np.pi/180.
        ra_upper = (ra+window_radius/3600./np.cos(dec_radian))
        ra_lower = (ra-window_radius/3600./np.cos(dec_radian))
        dec_upper = dec+window_radius/3600.
        dec_lower = dec-window_radius/3600.
        get_list = 'select tilename from Y3A2_COADD_OBJECT_BAND_G where ra between :ra_lower and :ra_upper and dec between :dec_lower and :dec_upper'
        self.cur.execute(get_list,ra_lower=ra_lower,ra_upper=ra_upper,dec_lower=dec_lower,dec_upper=dec_upper)
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            dtype_info = [('tilename',"|S41")]
            info_list = np.array(info_list,dtype=dtype_info)
            return info_list
        else:
            return None

    def get_supernova_tilenames(self):
        get_list = "select distinct t.tilename from Y6A1_IMAGE_TO_TILE t, y6a1_image i, y6a1_exposure e where t.filename=i.filename and e.expnum=i.expnum and e.program='supernova'"
        self.cur.execute(get_list)
        info_list = self.cur.fetchall()
        if len(info_list) > 0:
            dtype_info = [('tilename',"|S41")]
            info_list = np.array(info_list,dtype=dtype_info)
            return info_list
        else:
            return None
