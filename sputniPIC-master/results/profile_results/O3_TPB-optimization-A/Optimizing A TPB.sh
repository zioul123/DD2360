FILENAME="inputfiles/GEM_2D.inp"
LINEBREAK="**********************************"
EXPTNAME="O3"
CORRECTFILE="CPU_2D.vtk"

printf "Correctness of all files\n" > "${EXPTNAME}/Correctness.txt"
for TPB in 16 32 64 128 256 512
do
    for i in {1..5}
    do
        srun -n 1 nvprof --csv --log-file "${EXPTNAME}/nvprof-TPB${TPB}_${i}.csv" ./bin/sputniPIC.out $FILENAME -t "$TPB" -c | tee "${EXPTNAME}/output-TPB_${TPB}_${i}.txt"
        mv data/rho_net_10.vtk "${EXPTNAME}/rho_net-TPB${TPB}_${i}.vtk"
        ./DiffCheck.out "${EXPTNAME}/rho_net-TPB${TPB}_${i}.vtk" "$CORRECTFILE" >> "${EXPTNAME}/Correctness.txt"
        echo "$LINEBREAK"
        echo "TPB ${TPB} iteration ${i} done."
        echo "$LINEBREAK"
    done
done


