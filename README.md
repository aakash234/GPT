The data loader is implimented in C++.  
 
Since we have .npy shard files, we need cnpy module. `git clone https://github.com/rogersce/cnpy.git`  

First run the binding code to integrate c++ code to python by running:  This requires `pip install pybind11`  
`python binding_setup.py build_ext --inplace` this gives you additional .so file in the current dir.  

Now you can import the module to python scripts.  

If you wanted to run the C++ file individually and look at the output, you are required to point to the cnpy,eigen3 module while executing C++ code.  
`g++ -I /path/to/eigen3 -I /path/to/cnpy -o DataLoader DataLoader.cpp -lz`  
`./DataLoader`  

