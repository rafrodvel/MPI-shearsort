# PDP Project

All executables are runned with the same command line arguments, following the format `./snake input_file n output`, where:

- `input_file` is the path to the file containing the matrix to be sorted. The program expects a .txt file with a sequence of numbers separated by spaces and new lines, representing the matrix row-wise.
- `n` is the size of the matrix, i.e., the number of rows and/or columns.
- `output` is an integer that determines the output of the program, allowing it to switch between simple debugging or performance testing modes. The possible values are:
  - 0: Only prints the execution time of the algorithm to the console.
  - 1: Reads the specified matrix from the input file, sorts it, prints the output matrix to the console and checks if the matrix is sorted correctly.
  - 2: Ignores the input file and generates a random matrix of size `n x n`, sorts it, prints the output matrix to the console and checks if the matrix is sorted correctly.

A Python script `matrixgen.py` is provided to generate random matrices and write them to a file, which can be used as input for the program.