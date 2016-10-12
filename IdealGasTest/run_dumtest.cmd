parfile $PROTOMSHOME/parameter/amber14SB.ff
parfile $PROTOMSHOME/parameter/solvents.ff
parfile $PROTOMSHOME/parameter/amber14SB-residues.ff
parfile $PROTOMSHOME/parameter/gaff14.ff
solvent1 water.pdb
outfolder out_gcmc
streamheader off
streamdetail off
streamwarning warning
streaminfo info
streamfatal fatal
streamresults results
streamaccept accept
cutoff 10.0
feather 0.5
temperature 25.0
ranseed 1288827
boundary solvent
pdbparams on
#  GCMC specific parameters
gcmc 0
parfile gcmc_dum4p.tem
grand1 dummybath.pdb
potential 0.000
originx 30.024
originy 1.952
originz 8.033
x 5.3
y 4.731
z 4.645
#  End of GCMC specific parameters
dump 1 results write results
dump 1 pdb all solvent=all file=all.pdb standard
dump 1 restart write restart
dump 1 averages reset
chunk simulate 10 solvent=0 protein=00 solute=0 insertion=167 deletion=167 gcsolute=167
