#!/bin/bash
source /home/s1/cburke/eups/desdm_eups_setup.sh
export DES_SERVICES=/home/s1/cburke/.desservices.ini
setup despydb 2.0.6+0
setup despymisc
setup diffimg
setup swarp
setup sextractor
setup astropy
setup numpy
