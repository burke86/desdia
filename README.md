# desdia
### Dark Energy Survey Difference Imaging Code

![example](https://user-images.githubusercontent.com/13906989/85915997-e4c08b00-b811-11ea-8093-5f2df0d15962.png)

Produce difference imaging light curves for the dark energy survey in an automated fashion.

### Usage:

Needs to be run using the [EUPS](https://opensource.ncsa.illinois.edu/confluence/display/DESDM/The+Impatient%27s+Guide+to+DESDM+EUPS+installation) environment after installing packages in the `source.me` file. Note this assumes the paths to the setup and config files are `${HOME}/eups/desdm_eups_setup.sh` and `${HOME}/.desservices.ini`.


#### Supernova/COSMOS Fields

The `-c` flag can be used to specify the CCD. The `-n` flag specifies the number of threads to use. Due to invalid keywords in image headers, supressing warnings with `--nowarn` is recommended. Example:

`./desdia --field SN-C3 -n 30 -c 1 -w /data/des80.a/data/${USER}/ --nowarn`

`./desdia --field COSMOS -n 30 -c 1 -w /data/des80.a/data/${USER}/ --nowarn`


#### Wide-Area (Main Survey)

For the wide-area survey, specify the pointing number (value from 0-2038; see `y3point.csv` file). Each pointing will be split into regions (or a CCD if specified with the `-c` flag) using a template constructed from Y3 pointings:

`./desdia --survey 1 -n 30 -w /data/des80.a/data/${USER}/ --nowarn`

If you are only interested in a single target, the code will find the pointing and CCD number for you:

`./desdia --survey TARGET --ra 337.653325476 --dec -0.110275781 -n 30 -w /data/des80.a/data/${USER}/ --nowarn`

### Output:

The difference images and catalogs will be saved in the output directory. The reduced images will be downloaded and processed in the work directory. The output is a list of sextractor-like catalogs in the field for each CCD `SN-C3/cat_c?.dat`. Each line corresponds to a measurement, so they need to be combined and matched using external software to construct light curves. The catalogs are constructed using forced photometry from the template image with a 5" aperture. Additionally, detection can be done on the difference images to discover transients.
