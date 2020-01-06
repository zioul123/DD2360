FILENAME="inputfiles/GEM_2D_increased.inp"
LINEBREAK="**********************************"
EXPTNAME="O4"
CORRECTFILE="slow2d.vtk"

printf "Correctness of all files\n" > "${EXPTNAME}/Correctness.txt"
for nStreams in 2 4 8 16 32 64
do
    for i in {1..5}
    do
        srun -n 1 nvprof --csv --log-file "${EXPTNAME}/nvprof-nStreams_${nStreams}_${i}.csv" ./bin/sputniPIC.out $FILENAME -t 128 -s "$nStreams" -c | tee "${EXPTNAME}/output-nStreams_${nStreams}_${i}.txt"
        mv data/rho_net_10.vtk "${EXPTNAME}/rho_net-nStreams_${nStreams}_${i}.vtk"
        ./DiffCheck.out "${EXPTNAME}/rho_net-nStreams_${nStreams}_${i}.vtk" "$CORRECTFILE" >> "${EXPTNAME}/Correctness.txt"
        echo "$LINEBREAK"
        echo "nStreams ${nStreams} iteration ${i} done."
        echo "$LINEBREAK"
    done
done


