#include <cstdio>
#include <iostream>
#include <cstring>
#include <fstream>
#include <queue>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cout << "Need to provide the input file for diff checker." << std::endl;
        std::cout << "Usage: ./DiffChecker.out correct.vtk testing.vtk" << std::endl;
        return (-1);
    }

    std::string inputFile1 = argv[1], inputFile2 = argv[2];
    std::ifstream in1(inputFile1.c_str()); std::ifstream in2(inputFile2.c_str());
  
    std::string inLine;
    double inDouble; 
    std::queue<double> in1Doubles, in2Doubles;

    // Read file 1
    if (in1.is_open())
    {
        while ( in1.good() )
        {            
            getline (in1, inLine); 
            if (in1.eof()) break;

            if (sscanf(inLine.c_str(), "%lf", &inDouble) == 1) {
                in1Doubles.push(inDouble);
            } else {
                continue;
            }
        }
        in1.close();
        std::cout << "All input1 read." << std::endl;
    } 
    else 
    {
        std::cout << "Unable to open file 1." << std::endl;
        return -1;
    }

    // Read file 2
    if (in2.is_open())
    {
        while ( in2.good() )
        {            
            getline (in2, inLine); 
            if (in2.eof()) break;

            if (sscanf(inLine.c_str(), "%lf", &inDouble) == 1) {
                in2Doubles.push(inDouble);
            } else {
                continue;
            }
        }
        in2.close();
        std::cout << "All input2 read." << std::endl;
    } 
    else 
    {
        std::cout << "Unable to open file 2." << std::endl;
        return -1;
    }

    // Compare the two
    while (in1Doubles.size() > 0) 
    {
        // std::cout << in1Doubles.front() << ", " << in2Doubles.front() << std::endl;
        
        if (fabs(in1Doubles.front() - in2Doubles.front()) > 10e-6) {
            std::cout << "Big discrepancy in " << inputFile1 << ", " << inputFile2 << " with number " << in1Doubles.front() << " against " << in2Doubles.front() << std::endl;
            return 0;
        }
        in1Doubles.pop(); in2Doubles.pop();
    }
    std::cout << "All differences less than 10e-6." << std::endl;
 
    return 0;

}