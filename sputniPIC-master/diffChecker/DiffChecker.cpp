#include <cstdio>
#include <iostream>
#include <cstring>
#include <fstream>
#include <queue>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << "Need to provide the input file for diff checker." << std::endl;
        std::cout << "Usage: ./DiffChecker.out abc.diff" << std::endl;
        return (-1);
    }
    std::string inputFile = argv[1];
    std::ifstream in(inputFile.c_str());

    
    if (in.is_open())
    {
        std::string inLine;
        double inDouble; 
        std::queue<double> inLines;
        std::queue<double> outLines;

        while ( in.good() )
        {            
            getline (in, inLine); 
            if (in.eof()) break;
            // std::cout << in << std::endl;
            if (sscanf(inLine.c_str(), "< %lf", &inDouble) == 1) {
                inLines.push(inDouble);

            } else if (sscanf(inLine.c_str(), "> %lf", &inDouble) == 1) {
                outLines.push(inDouble);
            } else {
                continue;
            }
        }
        in.close();
        std::cout << "All input read." << std::endl;
        while (inLines.size() > 0) 
        {
            // std::cout << inLines.front() << ", " << outLines.front() << std::endl;
            if (fabs(outLines.front() - inLines.front()) > 10e-6) {
                std::cout << "Big discrepancy in " << inputFile << " with number " << outLines.front() << " against " << inLines.front() << std::endl;
                return 0;
            }
            inLines.pop(); outLines.pop();
        }
        std::cout << "All differences less than 10e-6." << std::endl;
    } 
    else 
    {
        std::cout << "Unable to open file." << std::endl;
    }
    return 0;

}