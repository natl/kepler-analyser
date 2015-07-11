#!/bin/python
"""
kepler-analyser.py
A wrapper file for model_analysis.py, with the instructions on how to 
complete the analysis.

Author: Nathanael Lampe
        nathanael@natlampe.com
        
Created: June 27, 2015

How to use me:
  This script requires some tinkering to set directories. Due to the variety of
  ways a system may be formatted, I've decided to this all inside this wrapper
  script which then calls the model_analysis.
  
  Essentially, you just need to set a few global variables for directories and 
  data table formats. This is all explained in comments. Once these are set, 
  type on the command line:
  
  python kepler-analyser.py
  
  The script will not allow you to create the directory structure if one already
  exists unless you add the command line argument "-w", to prevent careless
  overwriting of data. In this case type
  
  python kepler-analyser.py -w

"""
import numpy as np
import sys
import os   

from model_analysis import analysis

# Set this variable to be the directory where you would like the output
# to go. The directroy structure will be made to accomodate all the bursts.
# All data will go in a subdirectory called "bursts"
# It will resemble:
#  outputDir
#   |>bursts
#     |>runId1
#       |>1.data #burst1
#       |>2.data #burst2
#       |>3.data #burst3
#       |>mean.data #mean lightcurve
#     |>runId2
#       |>1.data #burst1
#       |>2.data #burst2
#       |>3.data #burst3
#       |>mean.data #mean lightcurve
#     ...
#     |>db.csv    #information for each separated burst
#     |>summ.csv  #mean burst information
outputDir = "./output/" #include trailing slash


#Set this variable to be the directory where the models will be found
#Models should be in ASCII format with their filenames ending as .data
#The name of the file should be identical to the name they are given in 
#the text file containing initial conditions
inputDir = "./modelfiles/" #include trailing slash


#Here is where we read in the initial conditions table. I have used numpy's
#genfromtxt routine, specifying the columns where new data starts
#You may change this routine as you see fit, the important thing is that the
#table contains named columns with the following names:
# name : model filename without .data extension
# acc  : accretion rate
# z    : metallicity
# h    : hydrogen fraction
# pul  : comment line describing the lightcurve (may be empty, but must exist)
#      : The comments on this line will be carried through to the output table
# cyc  : Number of numerical steps simulated 
# comm : area for comments

dt = [ (str('name'), 'S7'), (str('acc'), float), (str('z'), float),
         (str('h'), float), (str('lAcc'), float), (str('pul'), 'S20'),
         (str('cyc'), int), (str('comm'), 'S200') ]

keyTable = np.genfromtxt('./modelfiles/MODELS.txt', dtype = dt, 
  delimiter=[7,15,8,8,10,20,10,200] )

#Finally you can set any models to ignore or flag. There are three arrays for 
#this and each requires a string with the model name
# notAnalysable : Model will be skipped for any reason
# twinPeaks     : Model will be flagged as a twin peaked burst
# stableTrans   : Use this if a transition to stable burning leads to 
#                 analysis difficulties

noExist = [1,2,228,233,235,257,258,259,260,261,331]

notAnalysable = ['a324', 'a325','a326'] + ['a%i' % ii for ii in noExist]
twinPeaks     = ['a76','a281','a282','a362','a363','a364','a366','a387','a400',
    'a401','a402','a403','a408'] + ['a%i' % ii for ii in xrange(410,422)]
stableTrans   = ['a232']


def main( overwrite = False ):
    buildDirectoryStructure(overwrite = overwrite)
    analysis(keyTable, inputDir, outputDir, notAnalysable = notAnalysable,
        twinPeaks = twinPeaks, stableTrans = stableTrans)
    

def buildDirectoryStructure(overwrite = False):
    """
    """
    #Set whether or not paths can be overwritten, risking overwrites
    mkdir = os.mkdir if overwrite == True else safeMakeDir
    mkdir(outputDir+"bursts")
    for name in keyTable["name"]:
        mkdir(outputDir + r"bursts/" + name)
    return True
    

def safeMakeDir(path):
    assert not os.path.exists(path),("Continuing would use an already existing"+
        " directory, aborting. Use the -w flag to use existing paths (risks "+
        "overwriting files.")
    os.mkdir(path)
    return True


if __name__ == "__main__":
    if "-w" in sys.argv:
        overwrite = True
    else:
        overwrite = False
        
    main(overwrite = overwrite)
