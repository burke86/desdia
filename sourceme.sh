#!/bin/bash
source /data/des80.a/data/cburke/eups/desdm_eups_setup.sh
export DES_SERVICES=${HOME}/.desservices.ini
setup despydb 2.0.6+0
setup despymisc
setup diffimg
setup swarp
setup sextractor
setup astropy
setup numpy
