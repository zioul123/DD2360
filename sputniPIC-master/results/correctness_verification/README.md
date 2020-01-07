# Correctness Verification
In this directory, you can find the output files of our program that can be used for correctness verification.

The folders are organized with two categories,
- By input file used
- By grade

In each grade's folder, you will find the output files, diff files, and nvprof files from running the program with the mode specified by that grade's requirements.
- Grade d/e: Run on `GEM_2D.inp` and `GEM_3D.inp`
- Grade c: Run on `GEM_2D_increased.inp` and `GEM_3D_increased.inp`
- Grade b: Similar to grade c, with `nStreams=1` and `nStreams=3`
- Grade a: Similar to grade b, but with combined kernels used

In the "VerificationScript" folders, you will find the verification script run on all four files:
- 2D - `GEM_2D.inp`
- 2Di - `GEM_2D_increased.inp` (Batching logic is tested)
- 3D - `GEM_3D.inp`
- 3Di - `GEM_3D_increased.inp` (Batching logic is tested)

The verification script used will be the `.sh` file within the folder, and you can find the `.vtk` files (renamed from `rho_net_10.vtk` for each run), with the names:
- CPU - the CPU version (no flags)
- MI - GPU Mover and Interpolator version (`-m -i -t 128`)
- MIS1 - GPU Mover and Interpolator with 1 stream (`-m -i -s 1 -t 128`)
- MIS1C - Combined kernels with 1 stream (`-c -s 1 -t 128`)
- MIS10 - GPU Mover and Interpolator with 10 streams (`-m -i -s 10 -t 128`)
- MIS10C - Combined kernels with 10 streams (`-c -s 10 -t 128`)
