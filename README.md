Welcome to the Kepler model analysis routine
============================================

Author: Nathanael Lampe

Note: This is currently a work in progress

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




How will I get my output?



