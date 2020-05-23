#!/bin/bash
source /home/s1/cburke/eups/desdm_eups_setup.sh
export DES_SERVICES=/home/s1/cburke/.desservices.ini
setup despydb
setup despymisc
setup diffimg
setup swarp
setup sextractor
setup astropy
setup numpy
