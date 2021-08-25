import os, sys, time
import numpy as np
import argparse
import commands
from random import randint
import query, pipeline
from misc import bash

def start_desdia(pointing,ccd=None,targetra=None,targetdec=None,template_season=6,band='g',work_dir='./work',out_dir=None,threads=1,debug_mode=False,offset=False):
    # Start
    max_threads = 32
    top_dir = None
    fermigrid = False
    xfer_dir = "/pnfs/des/persistent/${USER}"
    # If code is running on grid
    if work_dir == '_CONDOR_SCRATCH_DIR':
        # See if tile already exists
        xfer_dir = "/pnfs/des/persistent/${USER}"
        exists_path = os.path.join(xfer_dir,"%s.tar.gz" % pointing)
        s = commands.getstatusoutput('ifdh ls %s' % exists_path)
        if exists_path in s[1].splitlines():
            print("Tile already analyzed, quitting.")
            return
        work_dir = os.environ[work_dir]
        top_dir = os.environ['CONDOR_DIR_INPUT']
        fermigrid = True
        max_threads = threads # Match to requested number of CPUs
        time.sleep(randint(1,10)) # Be less harsh on database
    # Create directory for tile
    tile_dir = os.path.join(work_dir,pointing)
    #if out_dir is None:
    #    out_dir = os.path.join(tile_dir,band)
    # Set up database
    query_sci = query.Query('db-dessci')
    print("Querying single-epoch images for %s." % pointing)
    # Get reduced filenames and related info
    # Supernova field
    if pointing.startswith('SN-') or pointing.lower() == "cosmos":
        # pointing is the fieldname in this case
        image_list = query_sci.get_image_info_field(pointing,band)
    # Main survey (pointing is a value from 0-2038)
    else:
        if targetra is None and targetdec is None:
            # Load pointings table
            dtype = [('tra',float),('tdec',float),('mjd_obs',float)]
            data = np.genfromtxt('./etc/y%dpoint.csv' % template_season,delimiter=',',skip_header=1,dtype=dtype)
            data = data[int(pointing)] # Get the pointing number
            # Get template filename info at requested pointing
            image_list = query_sci.get_image_info_pointing(data['tra'],data['tdec'],data['mjd_obs'],band=band)
        else:
            # Get template pointing at target
            pointing_list = query_sci.get_pointing_coord(targetra,targetdec,band,template_season)
            # Get template filename info at requested pointing
            image_list = query_sci.get_image_info_pointing(pointing_list['TRADEG'][0],pointing_list['TDECDEG'][0],pointing_list['mjd_obs'][0],band=band)
            ccd = image_list['ccd'][0]
    if image_list is None:
        print("***No images found in field/tile %s!***" % pointing)
        return
    
    # If ccd is specified, run in single-CCD mode
    if ccd is not None:
        image_list = image_list[image_list['ccd']==ccd]
    
    # Run pipeline
    des_pipeline = pipeline.Pipeline(band,query_sci.usr,query_sci.psw,tile_dir,top_dir,debug_mode)
    num_threads = np.clip(threads,0,max_threads)
    
    print("Running pipeline.")
    # Supernova field
    if pointing.startswith('SN-') or pointing.lower() == "cosmos":
        # Here image_list is all image info
        des_pipeline.run_ccd_sn(image_list,num_threads,template_season,fermigrid)
    # Main survey
    else:
        # Here image_list is just the template image info
        des_pipeline.run_ccd_survey(image_list,query_sci,num_threads,template_season,fermigrid,band,coadd_diff=offset,offset=offset)
    # Save data to out_dir
    b = np.vstack(map(list, image_list))
    np.savetxt(os.path.join(tile_dir,'image_list.csv'), b, fmt=','.join(['%s']*b.shape[1]))
    print('Compressing and transfering files.')
    # Compress and transfer files
    if fermigrid == True:
        os.chdir(work_dir)
        bash("tar czf %s.tar.gz %s" % (pointing,pointing))
        bash("ifdh cp -D %s.tar.gz %s" % (pointing,xfer_dir))
    return

def main():
    # Set up arguments
    parser = argparse.ArgumentParser(description='DES difference imaging pipeline.')
    parser.add_argument('--survey',default=None,help='Survey pointing number (1-2180) or target name')
    parser.add_argument('--field',type=str,default=None,help='Field name (e.g. SN-C3 or COSMOS)')
    parser.add_argument('--ra',type=float,default=None,help="Target RA [deg.]")
    parser.add_argument('--dec',type=float,default=None,help="Target dec [deg.]")
    parser.add_argument('-c','--ccd',type=int,default=None,help="Which CCD to use (default is all; ignored if pointing=TARGET)")
    parser.add_argument('-w','--work_dir',nargs='+',type=str,default='./work',help='Work directory')
    parser.add_argument('-o','--out_dir',nargs='+',type=str,default=None,help='Output directory')
    parser.add_argument('--grid',action='store_true',help='Run for all tiles on fermigrid')
    parser.add_argument('--debug',action='store_true',help='Run with debug mode (enhanced persistency)')
    parser.add_argument('--offset',action='store_true',help='Compute offsets in DIA image')
    parser.add_argument('--nowarn',action='store_true',help='Supress warnings')
    parser.add_argument('-f','--filter',type=str,default='g',help='Filter to use')
    parser.add_argument('-n','--threads',type=int,default=1,help='Number of threads')
    parser.add_argument('-s','--season',type=int,default=6,help='Template season to use Y[0-6]')
    args = parser.parse_args()
    band = np.asscalar(np.asarray(args.filter))
    work_dir = np.asscalar(np.asarray(args.work_dir))
    out_dir = np.asscalar(np.asarray(args.out_dir))
    print("==============================================")
    print("On grid:        %s" % args.grid)
    if args.survey is None and args.field is None:
        print("Must specify --survey or --field.")
        return
    if args.survey is not None and args.field is not None:
        print("Must only specify either --survey or --field.")
        return
    # Main survey
    if args.survey is not None:
        pointing = args.survey
        print("Pointing:       SURVEY-%s" % pointing)
    # SN field
    elif args.field is not None:
        pointing = args.field
        print("Field:      %s" % pointing)
    # Single-target mode
    if args.ra is not None or args.dec is not None:
        if args.ra is None or args.dec is None:
            print('Both RA and dec required in TARGET mode!')
            return
        print("RA:             %f [deg.]" % args.ra)
        print("dec:            %f [deg.]" % args.dec)
        print("CCD:            auto")
    elif args.ccd is None:
        print("CCD:         all")
    else:
        print("CCD:            %d" % args.ccd)
    print("Band:           %s" % band)
    print("Template:      Y%d" % args.season)
    print("Work directory: %s" % work_dir)
    print("Threads:        %d" % args.threads)
    print("Debug mode:     %d" % args.debug)
    print("offset    :     %d" % args.offset)
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
    #        tile_list = query_sci.get_all_pointings()
    #        tile = tile_info[num_proc][0]
    #    elif args.tile == "stripe82": # 652 tiles
    #        select_dec = (abs(tile_info["dec_cent"])-tile_info["dec_size"])<1.266
    #        select_ra = ((tile_info["ra_cent"]-tile_info["ra_size"]) < 60) | ((tile_info["ra_cent"]+tile_info["ra_size"]) > 300.5)
    #        tile_info = tile_info[select_dec & select_ra]
    #        tile = tile_info[num_proc][0]
    #    start_tile(tile,args.ccd,band,work_dir,out_dir,threads,args.debug)
    # single-tile mode
    start_desdia(pointing,args.ccd,args.ra,args.dec,args.season,band,work_dir,out_dir,args.threads,args.debug,args.offset)
    return

if __name__ == "__main__":
    # main function
    print('main')
    start_time = time.time()
    main()
    print("--- Done in %.2f minutes ---" % float((time.time() - start_time)/60))
