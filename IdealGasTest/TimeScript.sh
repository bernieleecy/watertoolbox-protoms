start=`date +%s`
$PROTOMSHOME/protoms3 run_gcmc.cmd
end=`date +%s`

runtime=$((end-start))
echo $runtime

