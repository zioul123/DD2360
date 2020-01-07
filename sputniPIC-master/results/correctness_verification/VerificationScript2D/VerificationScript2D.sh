FILENAME="inputfiles/GEM_2D.inp"
PREFIX="2D"
LINEBREAK="**********************************"
EXPTNAME="VerificationScript2D"

srun -n 1 ./bin/sputniPIC.out $FILENAME | tee "${EXPTNAME}/${PREFIX}_CPU.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_CPU.vtk"
echo "$LINEBREAK"
echo CPU Done
echo "$LINEBREAK"

srun -n 1 ./bin/sputniPIC.out $FILENAME -t 128 -m -i | tee "${EXPTNAME}/${PREFIX}_MI.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_MI.vtk"
echo "$LINEBREAK"
echo MI Done
echo "$LINEBREAK"

srun -n 1 ./bin/sputniPIC.out $FILENAME -t 128 -m -i -s 1 | tee "${EXPTNAME}/${PREFIX}_MIS1.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_MIS1.vtk"
echo "$LINEBREAK"
echo MIS1 Done
echo "$LINEBREAK"

srun -n 1 ./bin/sputniPIC.out $FILENAME -t 128 -m -i -s 10 | tee "${EXPTNAME}/${PREFIX}_MIS10.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_MIS10.vtk"
echo "$LINEBREAK"
echo MIS10 Done
echo "$LINEBREAK"

srun -n 1 ./bin/sputniPIC.out $FILENAME -t 128 -m -i -s 1 -c | tee "${EXPTNAME}/${PREFIX}_MIS1C.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_MIS1C.vtk"
echo "$LINEBREAK"
echo MIS10C Done
echo "$LINEBREAK"

srun -n 1 ./bin/sputniPIC.out $FILENAME -t 128 -m -i -s 10 -c | tee "${EXPTNAME}/${PREFIX}_MIS10C.txt"
mv data/rho_net_10.vtk "${EXPTNAME}/${PREFIX}_MIS10C.vtk"
echo "$LINEBREAK"
echo MIS10C Done
echo "$LINEBREAK"

echo Done running.

./DiffCheck.out "${EXPTNAME}/${PREFIX}_MI.vtk" "${EXPTNAME}/${PREFIX}_CPU.vtk" > "${EXPTNAME}/${PREFIX}_results.txt"
./DiffCheck.out "${EXPTNAME}/${PREFIX}_MIS1.vtk" "${EXPTNAME}/${PREFIX}_CPU.vtk" >> "${EXPTNAME}/${PREFIX}_results.txt"
./DiffCheck.out "${EXPTNAME}/${PREFIX}_MIS10.vtk" "${EXPTNAME}/${PREFIX}_CPU.vtk" >> "${EXPTNAME}/${PREFIX}_results.txt"
./DiffCheck.out "${EXPTNAME}/${PREFIX}_MIS1C.vtk" "${EXPTNAME}/${PREFIX}_CPU.vtk" >> "${EXPTNAME}/${PREFIX}_results.txt"
./DiffCheck.out "${EXPTNAME}/${PREFIX}_MIS10C.vtk" "${EXPTNAME}/${PREFIX}_CPU.vtk" >> "${EXPTNAME}/${PREFIX}_results.txt"
echo "Output to ${EXPTNAME}/results.txt done."
