# Diff Checker Tool
This is the tool we used to verify the correctness of our program.

To compile the program, you can run
```
g++ DiffChecker.cpp -o DiffChecker.out
```

To verify correctness, you will need to have a reference output file (.vtk), and the file you wish to check. Perform the checking like so:
```
./DiffChecker.out checkedFile.vtk correctFile.vtk
```

If the values in both files are all within 10e-6 of each other, the message "All differences less than 10e-6." will be displayed. 

If there is a big discrepancy, it will be flagged.
