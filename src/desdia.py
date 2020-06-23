import os, sys, time
import numpy as np
import argparse
import commands
from random import randint
from astropy.coordinates import SkyCoord
from astropy import units as u
import multiprocessing as mp
import query, pipeline
from misc import toIAU
from misc import bash
from misc import clean_pool

def start_tile(tilename,program='survey',band='g',work_dir='./work',out_dir=None,threads=1,debug_mode=False):
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
    if program == 'survey':
        file_list = query_sci.get_filenames_from_tile(tilename,band)
        if file_list is None:
            print("No images found.")
            return
        # get archive urls and other info
        image_list = query_sci.get_image_info(file_list,program)
    elif program == 'supernova': # tilename is the fieldname in this case
        image_list = query_sci.get_image_info_field(band,tilename)
    if image_list is None:
        print("No images found.")
        return
    # get coadd objects within tile geometry
    print("Querying tile geometry.")
    tile_head = query_sci.get_tile_head(tilename,band)
    des_pipeline = pipeline.Pipeline(band,program,query_sci.usr,query_sci.psw,tile_dir,out_dir,top_dir,debug_mode)
    num_threads = np.clip(threads,0,max_threads)
    print("Running pipeline.")
    lc_files = des_pipeline.run_ccd(image_list,num_threads,tile_head,fermigrid)
    #lc_files = des_pipeline.run(image_list,num_threads,tile_head,fermigrid)
    # plot summary statistics and save data
    print('Compressing and transfering files.')
    # compress and transfer files
    os.chdir(work_dir)
    if debug_mode == False:
        bash("tar czf %s.tar.gz %s" % (tilename,tilename))
    if fermigrid == True:
        bash("ifdh cp -D %s.tar.gz %s" % (tilename,xfer_dir))
    return

def main():
    # set up arguments
    parser = argparse.ArgumentParser(description='Find AGN from photometric variability in surveys.')
    parser.add_argument('-p','--program',nargs='+',type=str,default='survey',help="'supernova', 'survey', or 'legacy'")
    parser.add_argument('-t','--tile',nargs='+',type=str,help='tilename')
    parser.add_argument('-w','--work_dir',nargs='+',type=str,default='./work',help='work directory')
    parser.add_argument('-o','--out_dir',nargs='+',type=str,default=None,help='output directory')
    parser.add_argument('--grid',action='store_true',help='run for all tiles on fermigrid')
    parser.add_argument('--debug',action='store_true',help='run with debug mode (enhanced persistency)')
    parser.add_argument('--nowarn',action='store_true',help='supress warnings')
    parser.add_argument('-f','--filter',type=str,default='g',help='filter to use')
    parser.add_argument('-n','--threads',type=int,default=1,help='number of threads')
    args = parser.parse_args()
    program = np.asscalar(np.asarray(args.program))
    band = np.asscalar(np.asarray(args.filter))
    work_dir = np.asscalar(np.asarray(args.work_dir))
    out_dir = np.asscalar(np.asarray(args.out_dir))
    threads = np.asscalar(np.asarray(args.threads))
    print("==============================================")
    print("On grid:        %s" % args.grid)
    print("Program:        %s" % program)
    print("Band:           %s" % band)
    print("Work directory: %s" % work_dir)
    print("Threads:        %s" % threads)
    print("Debug mode:     %s" % args.debug)
    print("==============================================")
    if args.nowarn == True:
        import warnings
        warnings.filterwarnings("ignore")
    # wide survey mode (in FermiGrid environment)
    if args.grid == True:
        # get all tile names
        tile_info = np.load(os.path.join(os.environ["CONDOR_DIR_INPUT"],"tile_info.npy"))
        # use process number to select tile
        num_proc = int(os.environ["PROCESS"])
        if program == "survey": # 12,966 tiles
            # Note: this is too many to submit to the grid (current limit is 10k)
            tile_list = query_sci.get_all_tilenames()
            tile = tile_info[num_proc][0]
        elif program == "stripe82": # 652 tiles
            select_dec = (abs(tile_info["dec_cent"])-tile_info["dec_size"])<1.266
            select_ra = ((tile_info["ra_cent"]-tile_info["ra_size"]) < 60) | ((tile_info["ra_cent"]+tile_info["ra_size"]) > 300.5)
            tile_info = tile_info[select_dec & select_ra]
            tile = tile_info[num_proc][0]
            program = "survey"
        start_tile(tile,program,band,work_dir,out_dir,threads,args.debug)
    # single-tile mode
    elif args.tile is not None:
        # start tile
        tile = np.asscalar(np.asarray(args.tile))
        start_tile(tile,program,band,work_dir,out_dir,threads,args.debug)
    else:
        # none specified
        parser.error('Must specify --tile.')
    return

if __name__ == "__main__":
    # main function
    start_time = time.time()
    main()
    print("--- Done in %.2f minutes ---" % float((time.time() - start_time)/60))
