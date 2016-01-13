Welcome to the Kepler model analysis routine
============================================

Author: Nathanael Lampe

Note: This code should work, but handle it with care. Asking 
questions is encouraged.

What is it?

KEPLER is a 1-D hydrodynamics code that is used to simulate type 1
x-ray bursts. The analysis routines here opens KEPLER lightcurves
(in ASCII format) and analyses them, producing separated x-ray 
bursts from the burst trains provided as an input. It then returns
summary tables of individual burst properties, as well as global 
properties for each burst.


How do I use it?

This script requires three inputs. A master file containing the initial
conditions of each run, a directory containing the ASCII formatted
bursts the master initial conditions file refers to, and the desired 
output directory. The format of the initial conditions file and the 
ascii file are described below. Ideally, the output directory should
be empty to avoid any accidental file overwrites (As yet, I haven't
implemented an automatic overwrite prevention).

Prerequisites?

This script was designed for *Python 2.7*

You'll need the following Python packages too
 * lmfit
 * Scipy
 * Numpy
 * Bottleneck


How should I format my input?

 * Model files should be in ASCII format seperated by whitespace. They should 
   have the filename specified by there identifier in the initial conditions
   file, with the suffix ".data". They should all be located in the same 
   directory.
 * The initial conditions file is a bit more tricky. The file
   *kepler-analysis.py* will need to be edited as this files format changes.
   It requires the following columns:
   - name : model filename without .data extension
   - acc  : accretion rate
   - z    : metallicity
   - h    : hydrogen fraction
   - pul  : comment line describing the lightcurve (may be empty, but must 
     exist). The comments on this line will be carried through to the output table
   - cyc  : Number of numerical steps simulated 
   - comm : area for comments


How will I get my output?

Output is placed in the specified output directory in the *bursts* subdirectroy.
The file *bursts/summ.csv* summarises each model, and the file
*bursts/db.csv* contains the information for each separated burst

Show me an example.

The script kepler-analyser-example.py has been configured to work with the
files in example\_data, and will output the results to example\_output.
