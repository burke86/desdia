import os, sys, time
import numpy as np
import argparse
import commands
from random import randint
from astropy import units as u
import multiprocessing as mp
import query, pipeline
from misc import bash
from misc import clean_pool

def start_tile(tilename,ccd=None,band='g',work_dir='./work',out_dir=None,threads=1,debug_mode=False):
    # run pipeline on a tile
    max_threads = 32
    top_dir = None
    fermigrid = False
    xfer_dir = "/pnfs/des/persistent/cburke"
    # if on grid
    if work_dir == '_CONDOR_SCRATCH_DIR':
        # See if tile already exists
        xfer_dir = "/pnfs/des/persistent/cburke"
        exists_path = os.path.join(xfer_dir,"%s.tar.gz" % tilename)
        s=commands.getstatusoutput('ifdh ls %s' % exists_path)
        if exists_path in s[1].splitlines():
            print("Tile already analyzed, quitting.")
            return
        work_dir = os.environ[work_dir]
        top_dir = os.environ['CONDOR_DIR_INPUT']
        fermigrid = True
        max_threads = threads # match to requested number of CPUs
        time.sleep(randint(1,10)) # be less harsh on database
    # create directory for tile
    tile_dir = os.path.join(work_dir,tilename)
    if out_dir is None:
        out_dir = os.path.join(tile_dir,band)
    # set up database
    query_sci = query.Query('db-dessci')
    print("Querying single-epoch images for tile/field %s." % tilename)
    # get reduced filenames
    # DEPRICATED
    #if tilename.startswith('DES'):
    #    # get archive urls and other info
    #    image_list = query_sci.get_image_info_tile(tilename,band)
    #    print("Querying tile geometery.")
    #    tile_head = query_sci.get_tile_head(tilename,band)
    # Supernova field
    if tilename.startswith('SN-') or tilename.lower() == "cosmos": # tilename is the fieldname in this case
        image_list = query_sci.get_image_info_field(tilename,band)
    # Main survey [1-2180]
    elif tilename.startswith('SURVEY-'):
        # Load pointings table
        num = int(tilename.split('-')[1])
        dtype = [('tra',float),('tdec',float),('mjd_obs',float)]
        data = np.genfromtxt('etc/y3point.csv',delimiter=',',skip_header=1,dtype=dtype)
        data = data[num] # Get the pointing number
        image_list = query_sci.get_image_info_Y3_pointing(data['tra'],data['tdec'],data['mjd_obs'],band='g')
    else:
        print("Pointing/field '%s' not recognized!" % tilename)
    if image_list is None:
        print("***No images found in field/tile %s!***" % tilename)
        return
    # if ccd is specified, run in single-CCD mode
    if ccd is not None:
        image_list = image_list[image_list['ccd']==ccd]
    # run pipeline
    des_pipeline = pipeline.Pipeline(band,query_sci.usr,query_sci.psw,tile_dir,out_dir,top_dir,debug_mode)
    num_threads = np.clip(threads,0,max_threads)
    print("Running pipeline.")
    # Supernova field
    if tilename.startswith('SN-'):
        des_pipeline.run_ccd_sn(image_list,num_threads,fermigrid)
    elif tilename.startswith('SURVEY-'):
        des_pipeline.run_ccd_survey(image_list,query_sci,num_threads,fermigrid)
    # plot summary statistics and save data
    print('Compressing and transfering files.')
    # compress and transfer files
    if fermigrid == True:
        os.chdir(work_dir)
        bash("tar czf %s.tar.gz %s" % (tilename,tilename))
        bash("ifdh cp -D %s.tar.gz %s" % (tilename,xfer_dir))
    return

def main():
    print('main')
    # set up arguments
    parser = argparse.ArgumentParser(description='Find AGN from photometric variability in surveys.')
    parser.add_argument('pointing',nargs='+',type=str,help='tile or field name (e.g. SN-C3 or SURVEY-[1-2180])')
    parser.add_argument('-c','--ccd',type=int,default=None,help="which CCD to use (default is all)")
    parser.add_argument('-w','--work_dir',nargs='+',type=str,default='./work',help='work directory')
    parser.add_argument('-o','--out_dir',nargs='+',type=str,default=None,help='output directory')
    parser.add_argument('--grid',action='store_true',help='run for all tiles on fermigrid')
    parser.add_argument('--debug',action='store_true',help='run with debug mode (enhanced persistency)')
    parser.add_argument('--nowarn',action='store_true',help='supress warnings')
    parser.add_argument('-f','--filter',type=str,default='g',help='filter to use')
    parser.add_argument('-n','--threads',type=int,default=1,help='number of threads')
    args = parser.parse_args()
    tile = np.asscalar(np.asarray(args.pointing))
    band = np.asscalar(np.asarray(args.filter))
    work_dir = np.asscalar(np.asarray(args.work_dir))
    out_dir = np.asscalar(np.asarray(args.out_dir))
    threads = np.asscalar(np.asarray(args.threads))
    print("==============================================")
    print("On grid:        %s" % args.grid)
    print("Pointing/fiel   %s" % tile)
    print("Band:           %s" % band)
    if args.ccd is None:
        print("CCD:         all")
    else:
        print("CCD:            %d" % args.ccd)
    print("Work directory: %s" % work_dir)
    print("Threads:        %s" % threads)
    print("Debug mode:     %s" % args.debug)
    print("==============================================")
    if args.nowarn == True:
        import warnings
        warnings.filterwarnings("ignore")
    # wide survey mode (in FermiGrid environment)
    #if args.grid == True:
    #    # get all tile names
    #    tile_info = np.load(os.path.join(os.environ["CONDOR_DIR_INPUT"],"tile_info.npy"))
    #    # use process number to select tile
    #    num_proc = int(os.environ["PROCESS"])
    #    if args.tile == "all_survey": # 12,966 tiles
    #        # Note: this is too many to submit to the grid (current limit is 10k)
    #        tile_list = query_sci.get_all_tilenames()
    #        tile = tile_info[num_proc][0]
    #    elif args.tile == "stripe82": # 652 tiles
    #        select_dec = (abs(tile_info["dec_cent"])-tile_info["dec_size"])<1.266
    #        select_ra = ((tile_info["ra_cent"]-tile_info["ra_size"]) < 60) | ((tile_info["ra_cent"]+tile_info["ra_size"]) > 300.5)
    #        tile_info = tile_info[select_dec & select_ra]
    #        tile = tile_info[num_proc][0]
    #    start_tile(tile,args.ccd,band,work_dir,out_dir,threads,args.debug)
    # single-tile mode
    start_tile(tile,args.ccd,band,work_dir,out_dir,threads,args.debug)
    return

if __name__ == "__main__":
    # main function
    print('main')
    start_time = time.time()
    main()
    print("--- Done in %.2f minutes ---" % float((time.time() - start_time)/60))
