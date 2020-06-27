# desdia
### Dark Energy Survey Difference Imaging Code

![example](https://user-images.githubusercontent.com/13906989/85915997-e4c08b00-b811-11ea-8093-5f2df0d15962.png)

Produce difference imaging light curves for the dark energy survey in an automated fashion.

### Usage:

Needs to be run using the EUPS environment after installing packages in the `source.me` file. The input is a DES field name (e.g. SN-C3) or, for the wide-area field, tile name (e.g. DES0251+0043):

`./desdia field/tilename -n threads -w work directory -o output directory`

If you want to run a particular source, you will need to first query the field/tile name first using easyaccess.

#### Supernova Fields

The `-c` flag can be used to specify the CCD. Due to invalid keywords in image headers, supressing warnings is recommeded. Example:

`./desdia SN-C3 -n 30 -c 1 -w /data/des80.a/data/user/ --nowarn`

#### Wide-area Tile (coming soon)

For the wide-area survey, specify the tile. Each tile will be split into four regions.

`./desdia DES0251+0043 -n 30 -w /data/des80.a/data/user/ --nowarn`

### Output:

The difference images and catalogs will be saved in the output directory. The reduced images will be downloaded and processed in the work directory. The output is a list of sextractor-like catalogs in the field for each CCD `SN-C3/cat_c?.dat`. Each line corresponds to a measurement, so they need to be combined and matched using external software to construct light curves. The catalogs are constructed using forced photometry from the template image with a 5" aperture. Additionally, detection can be done on the difference images to discover transients.
