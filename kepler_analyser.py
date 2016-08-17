#!/bin/python
"""
kepler_analyser.py
Provides classes that collect information about a grid of models and use the
routines in model_analysis.py to analyse the light curves.

Author: Nathanael Lampe
        nathanael@natlampe.com

Created: June 27, 2015

How to use me:
import kepler_analyser
grid = kepler_analyser.ModelGrid('example_data', 'xrba', 'MODELS.txt', 'output_data')
number_of_runs = len(grid)
grid.analyse_all()

The ModelGrid class expects a directory structure (see for example the 'example_data'
directory) and a file with a "key table" describing the parameters of each model run
("MODELS.txt" in the example). Different directory structures and key tables can be
accommodated by making a subclass: see for example ElmoGrid.

TODO:
- cut out accretion dips?
- Work out why lmfit doesn't work on certain configurations
"""

from __future__ import division, unicode_literals, print_function
import numpy as np
import bottleneck as bn
import sys
import os
import logging

try:
    from utils import kepler # For reading binary light curve files
except:
    pass
import model_analysis

# You can set any models to ignore or flag. There are three arrays for
# this and each requires a string with the model name
# notAnalysable : Model will be skipped for any reason
# twinPeaks     : Model will be flagged as a twin peaked burst

notAnalysable = ['xrba324', 'xrba325','xrba326']
twinPeaks = ['xrba076','xrba281','xrba282','xrba362','xrba363','xrba364',
    'xrba366','xrba387','xrba400','xrba401','xrba402','xrba403',
    'xrba408'] + ['xrba%i' % ii for ii in xrange(410,422)]

class ModelGrid:
    """
    A class to facilitate the automated analysis of the light curves from a series
    of KEPLER runs.
    This class is setup for the grid from A. Heger, and reads ascii versions of the
    light curve files. For a new model grid, make a subclass and modify the necessary
    parts. See ElmoAnalyser as an example.

    All data will go in a specified output directory. It will have the following structure:
    outputDir
     |>modelId1
       |>0.data # burst 1
       |>1.data # burst 2
       |>2.data # burst 3
       |>mean.data # mean lightcurve
     |>modelId2
       |>0.data # burst 1
       |>1.data # burst 2
       |>2.data # burst 3
       |>mean.data # mean lightcurve
     ...
     |>db.csv    # information for each separated burst
     |>summ.csv  # mean burst information

    Example:
    grid = ModelGrid('model_files', 'xrba', 'MODELS.txt', 'burst_analysis')
    grid.analyse_all()
    """

    def __init__(self, base_dir, base_name, parameter_filename, output_dir, notAnalysable=[], twinPeaks=[]):
        """
        base_dir: all files are in (subdirectories of) this directory
        base_name: all problem names are base_name followed by a number
        parameter_filename: name of the file in base_dir that contains the
                            parameter values for each run
        output_dir: location of output files
        notAnalysable: manually categorize models
        twinPeaks: manually categorize models
        """
        self.base_dir = base_dir
        self.base_name = base_name
        self.parameter_filename = parameter_filename
        self.output_dir = output_dir
        self.notAnalysable = notAnalysable
        self.twinPeaks = twinPeaks

        self.dbVals = {}
        self.summVals = {}

        self.key_table = self.load_key_table()

    def load_key_table(self):
        """
        Return a dictionary with arrays that lists the problem names as well as some information
        on them. Fields are:
        name : model filename without .data extension
        acc  : accretion rate
        z    : metallicity
        h    : hydrogen fraction
        pul  : comment line describing the lightcurve (may be empty, must exist)
             : The comments on this line will be printed in the output table
        cyc  : Number of numerical steps simulated
        comm : area for comments
        """
        path = os.path.join(self.base_dir, self.parameter_filename)
        dt = [ (str('name'), 'S7'), (str('acc'), float), (str('z'), float),
               (str('h'), float), (str('lAcc'), float), (str('pul'), 'S20'),
               (str('cyc'), int), (str('comm'), 'S200') ]
        data = np.genfromtxt(path, dtype = dt, delimiter=[7,15,8,8,10,20,10,200] )

        table = {}
        for name in data.dtype.names:
            table[name] = data[name]
        table['name'] = [name.strip() for name in data['name']]
        #table['cyc'] = [cyc.strip() for cyc in data['cyc']]
        table['pul'] = [pul.strip() for pul in data['pul']]
        table['qb'] = np.ones(len(table['name']))*0.15
        return table

    def __len__(self):
        """
        Return the number of runs listed in the key_table
        """
        return len(self.key_table['name'])

    def build_output_directories(self):
        """
        Create the directories where the output will be placed in
        """
        self.safe_mkdir(self.output_dir)
        for name in self.key_table["name"]:
            self.safe_mkdir(os.path.join(self.output_dir, name.strip()))

    def safe_mkdir(self, path):
        """
        Create a directory only if it does not exist yet
        """
        if not os.path.exists(path):
            os.mkdir(path)
        elif not os.path.isdir(path):
            logging.warn(
                'Path already exists, but is not a directory: {}'.format(path))

    def load_light_curve(self, modelID):
        """
        Load the light curve from filename, and return time, luminosity and
        radius
        """
        filename = os.path.join(self.base_dir, '{}.data'.format(modelID))
        return np.loadtxt(filename, skiprows=1, unpack=True)

    def get_model_id(self, index):
        """
        For given index, return the model id
        """
        return self.key_table['name'][index]

    def is_completed(self, index):
        """
        Return whether the analysis of the run with given index is completed
        """
        modelID = self.get_model_id(index)
        if (modelID in self.dbVals) and (modelID in self.summVals):
            return True
        return False

    def analyse_all(self, step=1, skip_completed=False):
        """
        Iterate over all model runs and analyse the light curves
        step: analyse every 'step' run. Useful for debugging.
        skip_completed: whether to skip runs that have already been analysed.
                        Useful for resuming.
        """
        self.build_output_directories()

        for i in xrange(0, len(self), step):
            if not (skip_completed and self.is_completed(i)):
                self.analyse_one(i)

        self.save_burst_data()

    def analyse_one(self, index):
        """
        Analyse a single light curve with given index. Results are returned as
        well as stored in self.dbVals[modelID] and self.summVals[modelID].
        """
        modelID = self.get_model_id(index)
        try:
            burstTime, burstLum, burstRad = self.load_light_curve(modelID)
        except: # Problem reading light curve; file may not exist or be corrupt
            logging.warn(
                'Problem loading light curve {}. Skipping.'.format(modelID))
            return None, None
        # If the number of cycles was unknown, get it from the light curve
        if self.key_table['cyc'][index]==0:
            self.key_table['cyc'][index] = len(burstTime)

        ma = model_analysis.ModelAnalysis(modelID, burstTime, burstLum,
                                          burstRad)
        if modelID not in self.notAnalysable:
             # Locate bursts
            ma.separate(os.path.join(self.output_dir, modelID))
        x = ma.separated
        flag = ma.get_flag(modelID in self.twinPeaks,
                           modelID in self.notAnalysable)
        print('model: %s; flag: %i' % (modelID, flag))

        # For 1.4Msun, 10km NS (rest frame) and solar composition
        # TODO: make more generic?
        lAccErgs = float(1.17e46*self.key_table['acc'][index])
        alphas = ma.get_alpha(lAccErgs)

        dbVals = ma.get_burst_values()
        ma.get_mean_lcv(self.output_dir)
        summVals = ma.get_mean_values(
            {k: v[index] for k, v in self.key_table.items()})

        self.dbVals[modelID] = dbVals
        self.summVals[modelID] = summVals
        return dbVals, summVals

    def save_burst_data(self):
        """
        Save the results from the analyses to CSV files
        """
        # Collect data from all runs and bursts
        dbVals = []
        summVals = []
        for i in xrange(0, len(self)):
            modelID = self.get_model_id(i)
            if self.is_completed(i):
                db = self.dbVals[modelID]
                summ = self.summVals[modelID]
                if db!=None:
                    dbVals = dbVals + db
                    summVals.append(summ)

        summHead = ['model', 'num', 'acc', 'z', 'h', 'lAcc', 'pul', 'cyc',
                    'burstLength', 'uBurstLength', 'peakLum', 'uPeakLum',
                    'persLum', 'uPersLum', 'fluence', 'uFluence', 'tau',
                    'uTau', 'tDel', 'uTDel', 'conv', 'uConv', 'r1090',
                    'uR1090', 'r2590', 'uR2590', 'singAlpha', 'uSingAlpha',
                    'singDecay', 'uSingDecay', 'alpha', 'uAlpha', 'flag', 'Qb']

        dbHead = ['model', 'burst', 'bstart', 'btime', 'fluence', 'peakLum',
                  'persLum', 'tau', 'tdel', 'conv', 't10', 't25', 't90',
                  'fitAlpha', 'fitDecay', 'alpha']

        ofile = open(os.path.join(self.output_dir, 'db.csv'), 'w')
        ofile.write('# ' + ','.join(dbHead) + '\n')
        for line in dbVals:
            ofile.write(','.join(map(repr, line)) + '\n')
        ofile.close()

        ofile = open(os.path.join(self.output_dir, 'summ.csv'), 'w')
        ofile.write('# ' + ','.join(summHead) + '\n')
        for line in summVals:
            ofile.write(','.join(map(repr, line)) + '\n')
        ofile.close()


class ElmoGrid(ModelGrid):
    """
    Setup for analysing the "elmo" grid from L. Keek. Directly reads KEPLER's
    binary lc files.

    Example:
    grid = ElmoGrid('example_data', 'xrbh', 'params.txt', 'elmo_analysis')
    grid.analyse_all()
    """

    def load_key_table(self):
        """
        Alternate version that loads from a simpler file format
        """
        path = os.path.join(self.base_dir, self.parameter_filename)
        X = 0.71  # Accretion composition is solar
        Z = 0.02

        data = np.loadtxt(path, unpack=True)
        number = np.array(data[0], dtype=np.int)
        mdot = data[1]
        qb = data[2]

        table = {}
        table['name'] = ['{base}{number}'.format(base=self.base_name, number=i) for i in number]
        table['lAcc'] = mdot
        table['acc'] = mdot*1.75e-8
        table['z'] = np.ones_like(mdot)*Z
        table['h'] = np.ones_like(mdot)*X
        # Why is this a string? Because sometimes it uses characters to
        # describe the train
        table['pul'] = np.zeros(len(mdot), dtype=np.str)
        # This should be pulled from the lcv, not from any table..
        table['cyc'] = np.zeros_like(mdot)
        table['comm'] = np.zeros(len(mdot), dtype=np.str)
        table['qb'] = qb
        return table

    def load_light_curve(self, modelID):
        """
        Load the light curve from filename, and return time, luminosity and
        radius
        """
        filename = os.path.join(self.base_dir, modelID,
                                '{}.lc'.format(modelID))
        lc = kepler.LcFile(filename)
        return lc.time, lc.lum, lc.radius
