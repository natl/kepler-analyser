Welcome to the Kepler model analysis routine
============================================

Author: Nathanael Lampe

Note: This code should work, but handle it with care. Asking
questions is encouraged.

**I don't want to read things yet, show me the code**

OK, well if you have your burst lightcurves as ASCII files, try this:

```
import kepler_analyser as k
grid = k.ModelGrid("input_dir", "filename_prefix", "init_conditions_file",
    "output_dir")
grid.analyse_all()
```

If you have binary files from KEPLER as your lightcurves, try:

```
import kepler_analyser as k
grid = k.ElmoGrid("input_dir", "filename_prefix", "init_conditions_file",
    "output_dir")
grid.analyse_all()
```

Do you have lightcurves in a different format? You can write your own reader for them by inheriting from the
`kepler_analysis.ModelGrid` class

**What is it?**

KEPLER is a 1-D hydrodynamics code that is used to simulate type 1
x-ray bursts. The analysis routines here opens KEPLER lightcurves
(in ASCII format) and analyses them, producing separated x-ray
bursts from the burst trains provided as an input. It then returns
summary tables of individual burst properties, as well as global
properties for each burst.

**Prerequisites?**

This script was designed for *Python 2.7*

You'll need the following Python packages too
 * lmfit
 * Scipy
 * Numpy
 * Bottleneck

**Using the ModelGrid class (for ASCII data).**

The four arguments to instantiate the ModelGrid class are:
  1. input_dir: the directory where the lightcurve files are located
  2. filename_prefix: any prefix that is appended to the lightcurve files (e.g. xrba, xrbh, etc.)
  3. init_conditions_file: A file specifying the initial conditions of each lightcurve (see below for detailed instructions)
  4. output_dir: The output directory where lightcurves will be saved.

Keyword Arguments:
  * `twinpeaks=list` currently, the lightcurve analysis
routines cannot robustly identify twin peaked bursts, so these can be specified via the keyword argument `twinpeaks=[xrba001, xrba002, ...]` (where the prefix for files is xrba)
  * `notAnalysable=list` specified similarly to `twinpeaks`, this is a list of bursts that will not be analysed

**Using the ElmoGrid class (for KEPLER binary data).**
The four arguments to instantiate the ModelGrid class are:
  1. input_dir: the directory where the lightcurve files are located
  2. filename_prefix: any prefix that is appended to the lightcurve files (e.g. xrba, xrbh, etc.)
  3. init_conditions_file: A file specifying the initial conditions of each lightcurve (see below for detailed instructions)
  4. output_dir: The output directory where lightcurves will be saved.

Keyword Arguments:
  * `twinpeaks=list` currently, the lightcurve analysis
routines cannot robustly identify twin peaked bursts, so these can be specified via the keyword argument `twinpeaks=[xrba001, xrba002, ...]` (where the prefix for files is xrba)
  * `notAnalysable=list` specified similarly to `twinpeaks`, this is a list of bursts that will not be analysed

*NOTE:*

This class is particular to a set of model runs with fixed
compositions. Accordingly, this class reimplements the
method to load the initial condition file in a non-generic
way. It is recommended that users of this class ensure that
reimplement or overwrite the ElmoGrid.load_key_table method


**Format for the initial conditions file**
The initial conditions file requires by default the following columns
- name : model filename without .data extension
- acc  : accretion rate
- z    : metallicity
- h    : hydrogen fraction
- pul  : comment line describing the lightcurve (may be empty, but must
  exist). The comments on this line will be carried through to the output table
- cyc  : Number of numerical steps simulated
- comm : area for comments

The structure of this file is dependent upon the position of each column in the text file. For this reason, it is suggested that for your own implementation, you overwrite the method to read in the table with a method suited to your specific text file, as follows:

```
class MyGrid(ModelGrid):
    def load_key_table(self):
        table = dict()
        table['name'] = method_to_load_burst_names
        table['acc'] = method_to_load_accretions
        table['z'] = method_to_load_metallicities
        ...
        table['comm'] = method_to_load_burst_comments
        return table
```

**Default ASCII file format for lightcurves**

Model files should be in ASCII format seperated by whitespace. They should
have the filename specified by there identifier in the initial conditions
file, with the suffix ".data". They should all be located in the same
directory.


**How will I get my output?**

Output is placed in the specified output directory in the *bursts* subdirectroy.
The file *bursts/summ.csv* summarises each model, and the file
*bursts/db.csv* contains the information for each separated burst
