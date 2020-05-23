source /cvmfs/des.opensciencegrid.org/eeups/startupcachejob21i.sh
export EUPS_USERDATA=$CONDOR_DIR_INPUT
export DES_SERVICES=$CONDOR_DIR_INPUT/.desservices.ini
chmod 600 $CONDOR_DIR_INPUT/.desservices.ini
setup extralibs 1.0
setup despydb 2.0.5+0
setup despymisc 1.0.4+2
setup cxOracle 5.2.1+0
setup diffimg
setup swarp
setup sextractor
setup matplotlib
setup astropy 1.1.2+4
setup numpy 1.9.1+11
python $CONDOR_DIR_INPUT/findseeds.py "$@"
