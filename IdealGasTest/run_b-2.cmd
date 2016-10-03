parfile $PROTOMSHOME/parameter/amber14SB.ff
parfile $PROTOMSHOME/parameter/solvents.ff
parfile $PROTOMSHOME/parameter/amber14SB-residues.ff
parfile $PROTOMSHOME/parameter/gaff14.ff
solvent1 solvent.pdb
outfolder out-2
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
ranseed 5445237
pdbparams on
boundary solvent
#  GCMC specific parameters
gcmc 0
parfile dummy.tem
grand1 dummy2.pdb
potential -2.000
originx 0.0
originy 0.0
originz 0.0
x 5.0
y 5.0
z 5.0
#  End of GCMC specific parameters
dump 20000 results write results
dump 20000 pdb all solvent=all file=all.pdb standard
dump 20000 restart write restart
dump 20000 averages reset
chunk simulate 10000000 solvent=0 protein=0 solute=0 insertion=167 deletion=167 gcsolute=167
