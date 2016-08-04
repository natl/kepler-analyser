# Routines for Binary Evolution Code
# 2006-2008 Laurens Keek

from pylab import *
import numpy as n
from numpy import rec
import struct
from math import pi
import types
try:
    import asciidata as ad
except:
    print('Warning: asciidata not available.')
import os, gzip
import copy

# Define some constants
year = 31557600.0 # yr/sec. using 365.25 days per year.
xmsun= 1.9879261e33 # 1 solar mass
lsun = 3.83e33 # Solar luminosity
lightspeed = 2.998e10 # Speed of light
Grav = 6.673E-8 # Gravitational constant
a = 7.5646e-15 # Radiation constant
kb = 1.3807e-16 # Boltzmann constant erg/K
#isotopes = load_isotopes() # Isotopes used by the code.
mproton = 1.67262158e-24 # Proton mass (g)
keV = 1.602e-9 # erg

def fixline(line):
    """fix a string containing a line from the various
    ASCII output files of BEC"""
    #can't read e.g. 1D2, but I can read 1E2
    line = line.replace('D', 'E')
    #crappy hack to fix reading crappy files
    line = line.replace('0-', 'E-')
    line = line.replace('0+', 'E-')

    return line

def read_ascii(filename):
    """read columns from an ascii file as numarrays"""
    f = open(filename)

    line = f.readline()
    f.seek(0) #reset file pointer
    parts = line.split()
    num_columns = len(parts) #get the number of columns

    columns = [] #initialize arrays
    for i in range(num_columns):
        columns += [[]] 

    for line in f: #fill columns
        line = fixline(line)
        parts = line.split()

        if len(parts) == num_columns: #skip things like empty lines
            for i, part in enumerate(parts):
                columns[i] += [float(part)]

    #convert to numarrays
    for i,column in enumerate(columns):
        columns[i] = n.array(column)

    return columns

def get_diff(index, data):
    """From an array of numarrays of sm.data data, return an
    array with absolute relative difference in a variable wrt the previous
    model."""
    d = (data[index] - data[index+5])/data[index]
    d[d<0] *= -1

    return d

def isotope_names():
    """Array of isotope names"""
    return ['neutron', 'h1', 'h2', 'he3', 'he4', 'li6', 'li7', 'be7', 'be9', 'b8', 'b10',
            'b11', 'c11', 'c12', 'c13', 'c14', 'n14', 'n15', 'o16', 'o17', 'o18', 'ne20', 
            'ne21', 'ne22', 'na23', 'mg24', 'mg25', 'mg26', 'al27', 'si28', 'si29', 'si30', 
            'fe56', 'f19', 'al26']

class FortranFile:
    """Class for reading "unformatted" Fortran binary files"""

    class ReadError(Exception):
        """Exception raised when a record was not read correctly.
        This means that the record header does not match the
        record trailer (both should contain the length of the
        data area"""

        def __init__(self, filename, message=""):
            self.filename = filename # name of file which caused an error
            self.message = message # optional message

        def __str__(self):
            """Return error message"""
            return "Error reading record from file %s. %s" % (self.filename,
                                                              self.message)

    def __init__(self, filename, byteorder='='):
        """
        Open filename. Optionally the byte order can be specified (default is
        native).
        """
        self.filename = filename
        self.byteorder = byteorder
        self.open(filename)
        self.build_index()

    def open(self, filename):
        """Open the file"""
        self.filename = filename
        if filename.endswith('.gz'):
            self.file = gzip.open(filename)
        else:
            self.file = open(filename)
    
    def close(self):
        """Close the file"""
        self.file.close()
    
    def skip_record(self):
        """Read past a record without unpacking the data.
        Returns length of the record including header and trailer."""
        f = self.file

        #read data
        length = struct.unpack(self.byteorder+'i', f.read(4))[0]#read length of data section
        data = f.read(length)                    #read data
        check = struct.unpack(self.byteorder+'i', f.read(4))[0] #read length of data section again
        if check != length:                      #see if header matches trailer
            raise self.ReadError(self.filename, "Header of record does not match trailer.")

        return length+8 #total record length, including header and trailer.
    
    def build_index(self):
        """Generate an index of the positions of the records in the file for
        easy retrieval of a certain record."""
        index = []
        pos = 0
        try:
            while 1:
                index += [pos]
                pos += self.skip_record()
        except self.ReadError, e:
            print "Error reading record", len(index)-1, e
            print 'Ignored fraction of the file:', 1 - pos*1.0/os.stat(self.filename).st_size
            #raise
        except Exception, e:
            pass # Probably end of file

        self.index = index[:-1]
        self.file.seek(0) # Reset file pointer
    
    def update_index(self):
        """
        Check for new records added to the file, and append
        them to the index. This does not check for changes to
        the old records.
        """
        index = self.index
        # Read the last record to position the file pointer
        pos = index[-1]
        pos += len(self.read_a_record(len(index) - 1)) + 8
        
        # Check for new records
        try:
            while 1:
                index += [pos]
                pos += self.skip_record()
        except self.ReadError:
            print "Error reading record", len(index)-1
            raise
        except Exception, e:
            pass # Probably end of file

        self.index = index[:-1]
        self.file.seek(0) # Reset file pointer
    
    def read_record(self):
        """
        Read one record:
        - 4 byte integer containing length in bytes of data
        - data
        - same 4 byte integer containing length of data
        """
        f = self.file
        
        # Read data
        length = struct.unpack(self.byteorder+'i', f.read(4))[0]#read length of data section
        data = f.read(length)                    #read data
        check = struct.unpack(self.byteorder+'i', f.read(4))[0] #read length of data section again
        if check != length:                      #see if header matches trailer
            raise self.ReadError(self.filename)
        
        return data
    
    def read_a_record(self, index):
        """read indicated record"""
        self.file.seek(self.index[index])
        return self.read_record()

    def __len__(self):
        """
        Return the number of records in this file.
        """
        return len(self.index)

    def __getitem__(self, number):
        """
        Shortcut to read_a_record, such that a record can be
        retrieve as:
        file = FortranFile(filename)
        record = file[number]

        Array slicing returns a new FortranFile with an
        index containing only the entries specified by the
        slice.
        """
        if isinstance(number, slice):
            fortranfile = copy.deepcopy(self)
            fortranfile.index = fortranfile.index[number]
            return fortranfile
        else:
            return self.read_a_record(number)

    def __str__(self):
        """
        Return a string representation
        """
        return self.filename

class BinFile(FortranFile):
    """Class for reading models from a .bin? file."""
    def __init__(self, filename, byteorder='=', burnfile=''):
        """
        Create an index of the bin file. Optionally information from
        a .burn file can be read.
        """
        FortranFile.__init__(self, filename, byteorder)
        if burnfile:
            self.burn = BurnFile(burnfile)
        else:
            self.burn = None

    def get_version(self, data):
        """Return version information from data."""
        return struct.unpack(self.byteorder+"i", data[:4])[0]

    def read_a_model(self, index):
        """Read requested model"""
        model = None
        data = self.read_a_record(index)
        version = self.get_version(data)
        if version == 10002:
            model = RecModel(data)
        elif version == 10001:
            model = Model10001(data)
        
        if self.burn:
            burn = self.burn.read_a_model(model.model)
            if burn:
                model.enucl = burn.enucl
                model.enucl3a = burn.enucl3a
                model.dydt = burn.dydt
                model.dydtdiff = burn.dydtdiff
                model.dcdtdiff = burn.dcdtdiff
            else:
                model.enucl = n.zeros(model.n)
                model.enucl3a = n.zeros(model.n)
                model.dydt = n.zeros(model.n)
                model.dydtdiff = n.zeros(model.n)
                model.dcdtdiff = n.zeros(model.n)
           
        if model!=None:
            return model
        
        raise Model.VersionError(version)

    def __getitem__(self, number):
        """
        Shortcut to read_a_model, such that a model can be
        retrieve as:
        bin = BinFile(filename)
        model = bin[number]
        """
        if isinstance(number, slice):
            fortranfile = copy.deepcopy(self)
            fortranfile.index = fortranfile.index[number]
            return fortranfile
        else:
            return self.read_a_model(number)

    def get_model_at_time(self, time, mode='either'):
        """
        Return the model closests to time. Optionally the mode can be set to 'before', 'after', or 'either'
        """
        # Do an iterative search of the models, minimizing the number of files that need to be read
        low = 0 # Initial guess: it is between the first and last model
        high = len(self) - 1
        mlow = self[low]
        mhigh = self[high]
        
        if time <= mlow.time: # Check initial guess
            return mlow
        elif time>= mhigh.time:
            return mhigh

        while high-low > 1: # Step through models
            next = (high + low)/2
            mnext = self[next]
            if mnext.time > time:
                high = next
                mhigh = mnext
            else:
                low = next
                mlow = mnext
        
        if mode=='before': # Return model based on specified mode
            return mlow
        elif mode=='after':
            return mhigh
        elif mhigh.time - time < time - mlow.time:
            return mhigh
        else:
            return mlow

class BurnFile(FortranFile):
    """
    Read information from a .burn? file.
    File contains information on energy generation
    rate, change in He4 and C12 mass fractions due
    to burning and diffusion.
    """

    def __init__(self, filename):
        """
        Create an index of the bin file.
        """
        FortranFile.__init__(self, filename)
        self.models = dict(zip(self.model_index(), range(len(self.index))))

    def model_index(self):
        """
        Generate an index of all model numbers
        for which information is available.
        """
        return [self.get_model(i) for i in range(len(self.index))]
    
    def get_model(self, number):
        """
        Return the model number of record number.
        """
        # Get data for record
        data = self.read_a_record(number)

        # Unpack record
        fmt_1 = "=i" #Note the = sign to disable byte padding!
        len1  = struct.calcsize(fmt_1)
        result1 = struct.unpack(fmt_1, data[:len1])
        
        return result1[0]

    def read_a_model(self, index):
        """Read requested model"""
        if index in self.models:
            return Burn(self.read_a_record(self.models[index]))

        return None

class Model(dict):
    """
    One BEC model. It inherits from dict, so that we can use it as
    a name space for eval().
    """

    class VersionError(Exception):
        """Exception raised when a record was not read correctly.
        This means that the record header does not match the
        record trailer (both should contain the length of the
        data area"""

        def __init__(self, version, expected=10002):
            self.version = version
            self.expected = expected

        def __str__(self):
            """Return error message"""
            return "Error reading record: record has wrong version (%i instead of %i)" % (self.version, self.expected)

    def __init__(self, data):
        self.process_result(self.unpack_data(data))

    def unpack_data(self, data):
        """
        Unpack a string of data representing a record.
        
        Reads version 10002 records
        """
        
        result = []
        
        #parameter definition
        nmox = 5
        version = 10002 #only version that is supported
        pos = 0 #pointer in data
        
        #read part 1
        fmt_1 = "=ididd3i3d" #Note the = sign to disable byte padding!
        len1  = struct.calcsize(fmt_1)
        result1 = struct.unpack(fmt_1, data[pos:pos+len1])
        pos += len1
        
        if result1[0] != version: #check record version
            raise self.VersionError(result1[0])
        
        N = result1[5]
        nsp1 = result1[7]
        
        #read part 2
        fmt_2 = "=B17di%id" % (7+nmox)
        len2  = struct.calcsize(fmt_2)
        result2 = []
        for i in range(N):
            result2 += [struct.unpack(fmt_2, data[pos:pos+len2])]
            pos += len2
            #append the result to (num)arrays
        
        #read part 3
        fmt_3 = "=%id" % (N*(nsp1-1))
        len3  = struct.calcsize(fmt_3)
        result3 = struct.unpack(fmt_3, data[pos:pos+len3])
        pos += len3
        
        #read part 4
        fmt_4 = "=%id" % (nsp1)
        len4  = struct.calcsize(fmt_4)
        result4 = [struct.unpack(fmt_4, data[pos:pos+len4])]
        pos += len4
        
        #read part 5
        fmt_5 = "=d3i"
        len5  = struct.calcsize(fmt_5)
        result5 = []
        for i in range(N):
            #print i, pos, len(data)
            result5 += [struct.unpack(fmt_5, data[pos:pos+len5])]
            pos += len5
            #append the result to (num)arrays
        
        return result1, result2, result3, result4, result5
    
    def process_result(self, result):
        """Process the contents of a record"""
        #part 1
        (self.nvers, self.gms, self.model, self.dtn, self.time,
         self.n, self.n1, self.nsp1, self.windmd, self.vvcmax,
         self.dtalt) = result[0]
        
        #part 2
        part2 = n.array(result[1])
        part2.transpose()
        (self.convection, self.u, self.r, self.ro, self.t, self.sl,
         self.e, self.al, self.vu, self.vr, self.vro, self.vt, self.vsl,
         self.dm, self.bfbr, self.bfbt, self.bfvisc, self.bfdiff,
         self.ibflag, self.bfq, self.bfq0, self.bfq1, self.ediss,
         self.cap, self.diff, self.dg, self.d1, self.d2, self.d3,
         self.d4, self.d5) = part2
        
        #part 3
        i = n.array(result[2])
        i.setshape((self.n, self.nsp1-1))
        i.transpose()
        self.h1, self.h2, self.he3, self.he4, = i[:4]

    def pack_data(self):
        """Pack this model and return a string of data which
        can be inserted into a .bin? file."""
        
        #parameter definition
        nmox = 5
        version = 10002 #only version that is supported
        
        #part 1
        fmt_1 = "=ididd3i3d" #Note the = sign to disable byte padding!
        result1 = struct.pack(fmt_1, self.nvers, self.gms, self.model, self.dtn,
                              self.time, self.n, self.n1, self.nsp1, self.windmd,
                              self.vvcmax, self.dtalt)
        
        N = self.N
        nsp1 = self.nsp1
        
        #part 2
        fmt_2 = "=B17di%id" % (7+nmox)
        result2 = []
        for i in range(N):
            result2 += [struct.pack(fmt_2, )]
        
        #part 3
        fmt_3 = "=%id" % (N*(nsp1-1))
        result3 = struct.pack(fmt_3, )
        
        #read part 4
        fmt_4 = "=%id" % (nsp1)
        result4 = [struct.pack(fmt_4, )]
        
        #read part 5
        fmt_5 = "=d3i"
        result5 = []
        for i in range(N):
            result5 += [struct.pack(fmt_5, )]

        data = result1 + result2 + result3 + result4 + result5
        return data
        
    def get_list(self, names):
        """Generic get routine to return a dictionary of
        variables with key=name from names and value=value of
        variable. A name may refer to a variable or to a function
        that returns a value.
        
        names: a list of variable names to be retrieved."""
        
        out = {}
        for name in names:
            out[name] = self.get(name)
        
        return out
    
    def get(self, name):
        """Generic get routine to return a variable with a certain
        name. A name may refer to a variable or to a function
        that returns a value."""
        var = getattr(self, name) #get the variable
        t = type(var)          #get its type
        if t == types.MethodType: #is it a function?
            return var()
        else:
            return var
    
    def get_where(self, name1, name2, value2):
        """
        Get the linearly interpolated value of field 'name1',
        where field 'name2' has value2.
        Assumes the fields are monotonically increasing or decreasing (no local extrema)
        """
        field1 = self.get(name1)
        field2 = self.get(name2)
        
        if field2[3] - field2[2]<0: # Get the gradient sign, skipping the first gridpoint which can have 0 value
            field1 = field1[::-1]
            field2 = field2[::-1]
        value1 = n.interp(value2, field2, field1)
        return value1
    
    def __getitem__(self, name):
        """
        Overwrite this dict method to use our get().
        """
        return self.get(name)
    
    def __repr__(self):
        """
        Return a string representation of this Model
        """
        return "%s()"%(self.__class__.__name__,)

    def __str__(self):
        """
        Return an informative string.
        """
        return "%s %i"%(self.__class__.__name__, self.model)
    
    def get_change(self, a, b):
        """Return the relative difference between a and b,
        defined as (a-b)/b"""
        d = a-b
        ind = n.logical_not(b==0)
        d[ind] /= b[ind]
        
        return d

    def get_alfrac(self, alfrac):
        """
        Given alfrac, return the gridpoint that is the first of the
        gridpoints that receive accreted angular momentum.
        """
        cumdm = n.cumsum(self.dm[::-1])[::-1] # Cumulative mass from outside in.
        malfrac = alfrac*self.dm.sum() # Mass in alfrac part of model.
        return n.where(cumdm<malfrac)[0][0]
    
    # Methods that return an array of data. Can be accessed using get(...)
    
    def gridpoints(self):
        """array of gridpoints. Note that we start at 1 to be consistent with
        output of the fortran code."""
        return n.arange(1,self.n+1)
    
    def dm1(self):
        """dm is the mass of shells. The outer shell has mass 0. The mass of
	the core is not included. dm1 starts with the core mass and ends at shell n-1"""
	mcore = self.gms*xmsun - self.dm.sum()
	return n.concatenate([[mcore], self.dm[:-1]])
    
    def alspec(self):
        """Specific angular momentum"""
        return self.al/self.dm1()
    
    def ai(self):
        """Moment of inertia. Note that the moment of inertia i is related to the mass
	shell before gridpoint i"""
	# ai for gridpoint 1 and above
	ri = self.r[:-1]
        ra = self.r[1:]
        dm2 = self.dm[:-1]
        rai = ra*ri
        ra2 = ra**2
        ri2 = ri**2
        rm2 = ri2 + rai + ra2
	ai = 0.4*dm2*(ri2**2 + rai*rm2 + ra2**2)/rm2
	# ai for gridpoint 0
	w1 = self.al[1]/ai[0]
	if w1 == 0:
		ai0 = 1e-20 #prevent division by 0
	else:
		ai0 = self.al[0]/(w1) #using w0 = w1. The real w0 is in m.dat.
        return n.concatenate([[ai0],ai])
    
    def w(self):
        """Angular velocity"""
	return self.al/self.ai()

    def minu(self):
        """-Radial velocity"""
        return -self.u
    
    def pr(self):
        """Radiation Pressure"""
	return 1/3.0*a*self.t**4
    
    def pg(self):
        """Ideal gas Pressure"""
	return 2/3.0*self.ro*(self.e)#-3.0*self.pr()/self.ro)
    
    def edegen(self):
        """Electron degeneracy criterium, electrons are degenerate when returned
	value < 1. See Carroll & Ostlie (15.5)."""
	return self.t/self.ro**(2./3)/1.3e5
    
    def degen(self):
        """Degeneracy for mu=0.5. Matter is degenerate when returned value < 1"""
	return self.t/self.ro**(2./3)/4.78e-5
    
    def xm(self):
        """Lagrangian mass coordinate"""
	mcore = self.gms*xmsun - self.dm.sum()
	return n.cumsum( n.concatenate([[mcore], self.dm[:-1]]) )

    def xm_exclusive(self):
        """
        Lagrangian mass coordinate excluding the core mass
        """
        return self.dm.cumsum()
    
    def y(self):
        """
        Column depth in g/cm2. Defined *on* the grid.
        """
        ym = n.cumsum(self.dm[::-1])[::-1]
        y = ym/4/pi/self.r**2
        return y
    
    def eg(self):
        """
        Gravitational potential
        """
	return -Grav*self.xm()*self.dm/self.r

    def g(self):
        """
        Return the gravitational acceleration at each gridpoint.
        """
        return -Grav*self.xm()/self.r**2

    def tau(self):
        """
        Optical depth. For the outer gridpoint we define the
        optical depth as 0.
        """
        dtau = self.cap*self.ro
        dtau = dtau[:-1]*(self.r[1:] - self.r[:-1])
        return n.concatenate([n.cumsum(dtau[::-1])[::-1], [0]])

    def xdif(self):
        """
        Coefficients for diffusion of angular momentum. Efficiency of
        mixing due to convection is assumed to be 1.0. As in BEC,
        xdif(0) = xdif(1) = xdif(2).
        """
        result = self.diff + (self.dg + self.bfvisc)*(4*pi*self.r**2*self.ro/self.xm()[-1])**2
        result1 = n.array(result)
        result1[1:] = result[:-1]
        result1[1] = result1[2]
        result1[0] = result1[1]
        return result1
    
    def ledd(self):
        """Eddington limit for luminosity at outer gridpoint"""
        return 4*pi*lightspeed*Grav*self.xm()[-1]/self.cap[-1]
    
    def ledd1(self):
        """Eddington limit for luminosity at every gridpoint"""
        return 4*pi*lightspeed*Grav*self.xm()/self.cap
    
    def du(self):
        """Relative change since previous model"""
        return self.get_change(self.u, self.vu)
    
    def dr(self):
        """Relative change since previous model"""
        return self.get_change(self.r, self.vr)
    
    def dro(self):
        """Relative change since previous model"""
        return self.get_change(self.ro, self.vro)
    
    def dt(self):
        """Relative change since previous model"""
        return self.get_change(self.t, self.vt)
    
    def dsl(self):
        """Relative change since previous model"""
        return self.get_change(self.sl, self.vsl)
    
    def t_bfvisc(self):
        """
        Timescale for angular momentum diffusion due to magnetic field.
        """
        dr = self.r[1:] - self.r[:-1]
        t = n.cumsum((dr**2/self.bfvisc[:-1])[::-1])[::-1]
        return n.concatenate([t, [0]])
    
    def t_conv(self):
        """
        Timescale for angular momentum diffusion due to magnetic field.
        """
        dr = self.r[1:] - self.r[:-1]
        t = n.cumsum((dr**2/self.diff[:-1])[::-1])[::-1]
        return n.concatenate([t, [0]])
    
    def t_bfdiff(self):
        """
        Timescale for chemical diffusion due to magnetic field.
        """
        dr = self.r[1:] - self.r[:-1]
        t = n.cumsum((dr**2/self.bfdiff[:-1])[::-1])[::-1]
        return n.concatenate([t, [0]])
    
    def t_dg(self):
        """
        Timescale for diffusion due to rotational instabilities.
        """
        dr = self.r[2:] - self.r[1:-1]
        t = n.cumsum((dr**2/self.dg[1:-1])[::-1])[::-1]
        return n.concatenate([[t[0]], t, [0]])
    
    def t_he_burn(self):
        """
        Timescale for helium burning. Note: works only in
        combination with .burn file. Note: helium creation
        results in negative timescales.
        """
        t = self.he4/(self.dydt/self.dtalt)
        t[t==1e1000] = 0
        return -t
    
    def t_delta(self, delta, quantity):
        """
        Given an array of "delta" (changes of a quantity
        during the last timestep), calculate the timescale.
        """
        t = quantity/(delta/self.dtalt)
        t[t==1e1000] = 0
        return t
    
    def evisc(self):
        """
        Viscous heating rate per gram due to magnetic viscosity.
        """
        e = 0.5*self.bfvisc*(self.bfq*self.w())**2
        return e
    
    def ecool(self):
        """
        Radiative cooling rate per gram.
        """
        # Using analytical expression
        e = 4.0*a*lightspeed*self.t**4/3.0/self.cap/self.y()**2
        e[e==1e1000] = 0
        
        # Using luminosity in model
        #e = (self.sl[1:] - self.sl[:-1])/self.dm[:-1]
        #e = n.concatenate([e, [0]])
        
        return e

    def cs(self):
        """
        Speed of sound
        Note: use ideal gas.
        """
        return n.sqrt(5./3*self.pg()/self.ro)

    def al26(self):
        """Al26 abundance is 1 - abundances of all other isotopes."""
	return 1 - (self.h1 + self.h2 + self.he3 + self.he4 + self.li6 + self.li7 + self.be7 + self.be9 + self.b8 + self.b10 + self.b11 + self.c11 + self.c12 + self.c13 + self.c14 + self.n14 + self.n15 + self.o16 + self.o17 + self.o18 + self.ne20 + self.ne21 + self.ne22 + self.na23 + self.mg24 + self.mg25 + self.mg26 + self.al27 + self.si28 + self.si29 + self.si30 + self.fe56 + self.f19)
    
    def mu(self):
        """
        Mean molecular weight in units of proton mass. Kippenhahn & Weigert (13.6)
        Assumes complete ionization.
        """
	#mu = n.zeros(self.n)*1.0
        #for name,Z in zip(self.isotope_names()[1:], self.A()[1:]): # Iterate over isotopes, excluding neutrons.
        #    abund = self.get(name)
        #    mu = mu + abund*Z
        
        A = self.A()
        Z = self.Z()
        names = self.isotope_names()
        mu = n.zeros(len(self.t))*1.0
        
        for j in xrange(len(A)): # For each isotope
            X = self[names[j]]
            mu = mu + X*(1 + Z[j])/A[j]
        
        return 1.0/mu

    def mue(self):
        """
        Mean molecular weight per electron in units of proton mass. Kippenhahn & Weigert (13.8)
        Assumes complete ionization.
        """
        A = self.A()
        Z = self.Z()
        names = self.isotope_names()
        mu = n.zeros(len(self.t))*1.0
        
        for j in xrange(len(A)): # For each isotope
            X = self[names[j]]
            mu = mu + X*Z[j]/A[j]
        
        return 1.0/mu

    def delta(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (6.6)
        """
        ro = self['ro']
        T = self['t']
        dlnro = 2.0*(ro[1:] - ro[:-1])/(ro[1:] + ro[:-1])
        dlnT = 2.0*(T[1:] - T[:-1])/(T[1:] + T[:-1])
        delta = -1.0*dlnro/dlnT
        return n.concatenate([[0.0], delta])

    def phi(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (6.6)
        """
        ro = self['ro']
        mu = self['mu']
        dlnro = 2.0*(ro[1:] - ro[:-1])/(ro[1:] + ro[:-1])
        dlnmu = 2.0*(mu[1:] - mu[:-1])/(mu[1:] + mu[:-1])
        phi = dlnro/dlnmu
        return n.concatenate([[0.0], phi])
    
    def alpha(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (6.6)
        From EOS
        """
        ro = self['ro']
        P = self['pn']
        dlnro = 2.0*(ro[1:] - ro[:-1])/(ro[1:] + ro[:-1])
        dlnP = 2.0*(P[1:] - P[:-1])/(P[1:] + P[:-1])
        alpha = dlnro/dlnP
        return n.concatenate([[0.0], alpha])
    
    def nablarad(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (5.27)
        """
        T = self['t']
        P = self['pn']
        dlnT = 2.0*(T[1:] - T[:-1])/(T[1:] + T[:-1])
        dlnP = 2.0*(P[1:] - P[:-1])/(P[1:] + P[:-1])
        nablarad = dlnT/dlnP
        return n.concatenate([[0.0], nablarad])
    
    def nablarad1(self):
        """
        Alternative calculation of nablarad. Kippenhahn & Weigert (5.28).
        Assumes hydrostatic equilibrium.
        """
        nablarad = 3.0/(16.0*n.pi*a*lightspeed*Grav)*self['cap']*self['sl']*self['pn']/(self['xm']*self['t']**4)
        return nablarad
    
    def nablaad(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (4.19)
        Uses helmholtz EOS.
        """
        from helmholtz import Helmholtz
        hh = Helmholtz()
        
        T = self['t']
        ro = self['ro']
        abar = self['abar']
        zbar = self['zbar']
        
        nablaad = n.zeros_like(T)
        for i in xrange(len(T)):
            hh.den = ro[i]
            hh.abar = abar[i]
            hh.zbar = zbar[i]
            
            T1 = T[i]
            hh.temp = T1
            hh.eos()
            P1 = hh.pres
            
            T2 = T1*(1 + 1e-4)
            hh.temp = T2
            hh.eos()
            P2 = hh.pres
            
            dlnT = 2.0*(T2 - T1)/(T2 + T1)
            dlnP = 2.0*(P2 - P1)/(P2 + P1)
            nablaad[i] = dlnT/dlnP
        
        return nablaad
    
    def nablamu(self):
        """
        Factor in Ledoux criterion. Kippenhahn & Weigert (6.10)
        """
        mu = self['mu']
        P = self['pn']
        dlnmu = 2.0*(mu[1:] - mu[:-1])/(mu[1:] + mu[:-1])
        dlnP = 2.0*(P[1:] - P[:-1])/(P[1:] + P[:-1])
        nablamu = -1.0*dlnmu/dlnP
        return n.concatenate([[0.0], nablamu])

    def hp(self):
        """
        Pressure scale height
        """
        P = self['pn']
        r = self['r']
        dr = r[1:] - r[:-1]
        dlnP = 2.0*(P[1:] - P[:-1])/(P[1:] + P[:-1])
        drdlnP = -dr/dlnP
        return n.concatenate([[0.0], drdlnP])
    
    def cap_sc(self):
        """
        Electron scattering opacity. Kippenhahn & Weigert (17.2)
        In cm2/g
        """
        return 0.2*(1.0 + self['h1'])
    
    def cap_ff(self):
        """
        Free-free opacity. Kippenhahn & Weigert (17.5)
        """
        A = self.A()
        Z = self.Z()
        names = self.isotope_names()
        B = n.zeros(len(self.t))*1.0
        for j in xrange(len(A)): # For each isotope
            if Z[j]>2: # Only heavier than helium
                X = self[names[j]]
                B = B + X*Z[j]*Z[j]/A[j]
        
        return 3.8e22*(1 + self['h1'])*(self['h1'] + self['he4'] + B)*self['ro']*self['t']**-3.5
    
    def Z(self):
        """number of protons per nucleus for each isotope"""
	return n.array([ 0,  1,  2,  3,  4,  6,  7,  7,  9,  8, 10, 11, 11, 12, 13, 14,
                        14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                        56, 19, 26])
    
    def A(self):
        """number of nuclei per nucleus for each isotope"""
	return n.array([  1.,   1.,   2.,   3.,   4.,   6.,   7.,   7.,   9.,   8.,
                         10.,  11.,  11.,  12.,  13.,  14.,  14.,  15.,  16.,  17.,
                         18.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,
                         29.,  30.,  56.,  19.,  26.])
    
    def cp(self):
        """
        Specific heat for ideal gas.
        """
        return 5./2*kb/(self.mu()*mproton)

    def t_thermal(self):
        """
        Thermal timescale.
        """
        return self.cp()*self.t*self.y()*4*pi*self.r**2/n.abs(self.sl)
    
    def isotope_names(self):
        """Array of isotope names"""
	return ['neutron', 'h1', 'h2', 'he3', 'he4', 'li6', 'li7', 'be7', 'be9', 'b8', 'b10',
	        'b11', 'c11', 'c12', 'c13', 'c14', 'n14', 'n15', 'o16', 'o17', 'o18', 'ne20', 
		'ne21', 'ne22', 'na23', 'mg24', 'mg25', 'mg26', 'al27', 'si28', 'si29', 'si30', 
		'fe56', 'f19', 'al26']
    
    def abundances(self):
        """Return array of isotope abundances for each gridpoint"""
	return 1
        
    def extract(self, recordarray, min=0, max=None):
        """
        Similar to the extract function in php, extract
        takes the fields in a recordarray and puts them as
        attributes into the Model instance.
        Optionally, for arrays, min and max can be specified, such
        that array[min:max] is returned.
        """
        
        if recordarray.shape==(1,):
            # Scalars
            for name in recordarray.dtype.names:
                setattr(self, name, recordarray.field(name)[0])
        else:
            # Arrays
            for name in recordarray.dtype.names:
                setattr(self, name, recordarray.field(name)[min:max])

class Model10001(Model):
    """Model with version 10001"""

    def __init__(self, data):
        Model.__init__(self, data)
    
    def unpack_data(self, data):
        """
        unpack a string of data representing a record.
        
        Reads version 10001 records.
        """
        
        result = []
        
        #parameter definition
        nmox = 5
        version = 10001 #only version that is supported in this model
        pos = 0 #pointer in data
        
        #read part 1
        fmt_1 = "=ididd3i3d" #Note the = sign to disable byte padding!
        len1  = struct.calcsize(fmt_1)
        result1 = struct.unpack(fmt_1, data[pos:pos+len1])
        pos += len1
        
        if result1[0] != version: #check record version
            raise Model.VersionError(result1[0], version)
        
        N = result1[5]
        nsp1 = result1[7]
        
        #read part 2
        fmt_2 = "=B%id" % (15+nmox)
        len2  = struct.calcsize(fmt_2)
        result2 = []
        for i in range(N):
            result2 += [struct.unpack(fmt_2, data[pos:pos+len2])]
            pos += len2
            #append the result to (num)arrays
        
        #read part 3
        fmt_3 = "=%id" % (N*(nsp1-1))
        len3  = struct.calcsize(fmt_3)
        result3 = struct.unpack(fmt_3, data[pos:pos+len3])
        pos += len3
        
        #read part 4
        fmt_4 = "=%id" % (nsp1)
        len4  = struct.calcsize(fmt_4)
        result4 = [struct.unpack(fmt_4, data[pos:pos+len4])]
        pos += len4
        
        #read part 5
        fmt_5 = "=d3i"
        len5  = struct.calcsize(fmt_5)
        result5 = []
        for i in range(N):
            #print i, pos, len(data)
            result5 += [struct.unpack(fmt_5, data[pos:pos+len5])]
            pos += len5
            #append the result to (num)arrays
        
        return result1, result2, result3, result4, result5
    
    def process_result(self, result):
        """Process the contents of a record"""
        #part 1
        (self.nvers, self.gms, self.model, self.dtn, self.time,
         self.n, self.n1, self.nsp1, self.windmd, self.vvcmax,
         self.dtalt) = result[0]
        
        #part 2
        part2 = n.array(result[1])
        part2.transpose()
        (self.convection, self.u, self.r, self.ro, self.t, self.sl,
         self.e, self.al, self.vu, self.vr, self.vro, self.vt, self.vsl,
         self.dm, self.diff, self.dg, self.d1, self.d2, self.d3,
         self.d4, self.d5) = part2
        
        #part 3
        i = n.array(result[2])
        i.setshape((self.n, self.nsp1-1))
        i.transpose()
        self.h1, self.h2, self.he3, self.he4, = i[:4]

class RecModel(Model):
    """One BEC model. A test for using numarray's Record arrays
    as a faster way to load data."""

    def __init__(self, data):
        Model.__init__(self, data)

    def unpack_data(self, data):
        """
        Unpack a string of data representing a record.
        Reads version 10002 records.
        """
        
        result = []
        
        # parameter definition
        nmox = 5
        version = 10002 # only version that is supported in this class
        pos = 0 # pointer in data
        
        # read part 1
        fmt_1 = "=ididd3i3d" # Note the = sign to disable byte padding!
        len1  = struct.calcsize(fmt_1)
        result1 = struct.unpack(fmt_1, data[pos:pos+len1])
        pos += len1
        
        if result1[0] != version: # check record version
            raise self.VersionError(result1[0])
        
        N = result1[5]
        nsp1 = result1[7]
        
        # read part 2
        fmt_2 = "=B17di%id" % (7+nmox)
        len2  = struct.calcsize(fmt_2)
        
        part2 = data[pos:pos+N*len2]
        fmt_2 = 'u1' + 17*',f8' + ',i4' + (7+nmox)*',f8'
        names_2 = "convection,u,r,ro,t,sl,e,al,vu,vr,vro,vt,vsl,dm,bfbr," \
                  + "bfbt,bfvisc,bfdiff,ibflag,bfq,bfq0,bfq1,ediss,cap," \
                  + "diff,dg,d1,d2,d3,d4,d5"
        r=rec.fromstring(part2, formats=fmt_2, shape=N, names=names_2)
        self.extract(r)
        pos += N*len2
        
        # read part 3
        fmt_3 = "=%id" % (N*(nsp1-1))
        len3  = struct.calcsize(fmt_3)
        
        part3 = data[pos:pos+len3]
        fmt_3 = 'f8'+(nsp1-2)*',f8'
        names_3 = "neutron,h1,h2,he3,he4,li6,li7,be7,be9,b8,b10,b11,c11,c12,c13," \
                  + "c14,n14,n15,o16,o17,o18,ne20,ne21,ne22,na23,mg24,mg25," \
                  + "mg26,al27,si28,si29,si30,fe56,f19"
        r=rec.fromstring(part3, formats=fmt_3, shape=N, names=names_3)
        self.extract(r)
        # from bin2am.f
        self.neutron = self.neutron*self.ro*6.0255E23
        pos += len3
        
        # read part 4. Describes how much of each isotope is lost to interstellar medium.
        fmt_4 = "=%id" % (nsp1)
        len4  = struct.calcsize(fmt_4)
        # result4 = [struct.unpack(fmt_4, data[pos:pos+len4])]
        pos += len4
        
        # read part 5
        fmt_5 = "=d3i"
        len5  = struct.calcsize(fmt_5)
        result5 = []
        for i in range(N):
            #print i, pos, len(data)
            #result5 += [struct.unpack(fmt_5, data[pos:pos+len5])]
            pos += len5
            #append the result to (num)arrays
        
        return (result1,)

    def process_result(self, result):
        """Process the contents of a record"""
        #part 1
        (self.nvers, self.gms, self.model, self.dtn, self.time,
         self.n, self.n1, self.nsp1, self.windmd, self.vvcmax,
         self.dtalt) = result[0]

class ReducedModel(Model):
    """A ReducedModel instance contains only a subset of the
    properties (data) available in a Model instance. Used to
    reduce memory usage."""

    def __init__(self, model, properties):
        """Load given properties from Model instance into
        a ReducedModel."""
        self.properties = properties
        for property in properties:
            setattr(self, property, model.get(property))

class Burn:
    """
    Nuclear burning information for a model.
    """

    def __init__(self, data):
        """
        Initialize a Burn instance from given data.
        """
        self.unpack_data(data)

    def unpack_data(self, data):
        """
        Unpack binary data.
        """
        fmt = '=2i'
        pos = 0
        len  = struct.calcsize(fmt)
        self.model, self.n = struct.unpack(fmt, data[pos:pos+len])
        pos = pos + len

        fmt = '=%id'%(self.n*5)
        len  = struct.calcsize(fmt)
        result = struct.unpack(fmt, data[pos:pos+len])
        pos = pos + len
        result = n.array(result)
        self.enucl = result[:self.n]
        self.enucl3a = result[self.n:self.n*2]
        self.dydt = result[self.n*2:self.n*3]
        self.dydtdiff = result[self.n*3:self.n*4]
        self.dcdtdiff = result[self.n*4:self.n*5]

class Plot:
    """Class for reading *.plot[12] files and making
    different plots"""

    def __init__(self, filename):
        """initialize by reading from filename"""
        f = open(filename)

        #initialize arrays
        self.time    = []
        self.Tc      = []
        self.Yc      = []
        self.LH      = []
        self.LHe     = []
        self.M       = []
        self.Teff    = []
        self.L       = []
        self.rhoc    = []
        self.LC      = []
        self.Lnu     = []
        self.MdotWind= []
        self.Tmax    = []
        self.rhoTmax = []
        self.MTmax   = []

        #fill arrays
        for line in f:
            #can't read e.g. 1D2, but I can read 1E2
            line = fixline(line)
            
            parts = line.split()
            self.time    += [float(parts[0])]
            self.Tc      += [float(parts[1])]
            self.Yc      += [float(parts[2])]
            self.LH      += [float(parts[3])]
            self.LHe     += [float(parts[4])]
            self.Teff    += [float(parts[7])]
            self.L       += [float(parts[8])]
            self.rhoc    += [float(parts[9])]
            self.LC      += [float(parts[10])]
            self.Lnu     += [float(parts[11])]
            self.MdotWind+= [float(parts[12])]
            self.Tmax    += [float(parts[13])]
            self.rhoTmax += [float(parts[14])]
            self.MTmax   += [float(parts[15])]


        f.close()

    def hrd(self):
        """Plot a HR-diagram"""
        figure(123)
        plot(n.log10(n.array(self.Teff)), self.L)
        xlabel('logT')
        ylabel('logL')

class Plot1:
    """Class for reading *.plot[12] files and making
    different plots"""

    def __init__(self, filename):
        """initialize by reading from filename"""
        f = ad.open(filename, exponentD=True)

        # Find places where we went back in time.
        t = n.array(f[0].tonumarray())
        dt = t[1:] - t[:-1]
        good = n.ones(len(t), n.Bool)
        ind = n.where(dt<0)[0]
        for i in ind:
            T = t[i+1]
            j = i
            while t[j]>=T and j>-1:
                good[j] = False
                j -= 1
        
        # Names of the fields.
        names = ['t', 'Tc', 'Yc', 'LH', 'LHe', 'm', 'edisp', 'Teff', 'L', 'rhoc', 'LC', 'Lnu',
                     'mdotwind', 'Tmax', 'rhoTmax', 'mrTmax', 'rquot', 'kip90', 'Lcore']
        for i,name in enumerate(names):
            setattr(self, name, n.array(f[i].tonumarray())[good])

        # Some conversions to sensible units.
        self.t *= year # Time in s.
        self.Tc *= 1e8 # Temperature in K.
        self.LH = 10**self.LH*lsun # Luminosity in erg/s
        self.LHe = 10**self.LHe*lsun # Luminosity in erg/s
        self.L = 10**self.L*lsun # Luminosity in erg/s
        self.rhoc = 10**self.rhoc # Density in g/cm3
        self.LC = 10**self.LC*lsun # Luminosity in erg/s
        self.Lnu = 10**self.Lnu*lsun # Luminosity in erg/s

class Abundances:
    """Class for reading *.cen[12] and *.surf[12] files
    containing isotope abundances.
    
    Arrays are named after isotope and are of type numarray."""

    def __init__(self, filename):
        """initialization by reading from filename"""
        f = open(filename)

        #remember the names
        self.names = ['H1', 'H2', 'He3', 'He4', 'Li6', 'Li7', 'Be7', 'Be9',
                      'B8', 'B10', 'B11', 'C11', 'C12', 'C13', 'N12', 'N14',
                      'N15', 'O16', 'O17', 'O18', 'Ne20', 'Ne21', 'Ne22',
                      'Na23', 'Mg24', 'Mg25', 'Mg26', 'Al27', 'Si28', 'Si29',
                      'Si30', 'Fe56', 'F19', 'Al26']

        #initialize arrays
        self.time = []
        self.H1   = []
        self.H2   = []
        self.He3  = []
        self.He4  = []
        self.Li6  = []
        self.Li7  = []
        self.Be7  = []
        self.Be9  = []
        self.B8   = []
        self.B10  = []
        self.B11  = []
        self.C11  = []
        self.C12  = []
        self.C13  = []
        self.N12  = []
        self.N14  = []
        self.N15  = []
        self.O16  = []
        self.O17  = []
        self.O18  = []
        self.Ne20 = []
        self.Ne21 = []
        self.Ne22 = []
        self.Na23 = []
        self.Mg24 = []
        self.Mg25 = []
        self.Mg26 = []
        self.Al27 = []
        self.Si28 = []
        self.Si29 = []
        self.Si30 = []
        self.Fe56 = []
        self.F19  = []
        self.Al26 = []

        #fill arrays
        for line in f:
            #can't read e.g. 1D2, but I can read 1E2
            line = fixline(line)
            
            parts = line.split()

            self.time += [float(parts[1])]
            self.H1   += [float(parts[2])]
            self.H2   += [float(parts[3])]
            self.He3  += [float(parts[4])]
            self.He4  += [float(parts[5])]
            self.Li6  += [float(parts[6])]
            self.Li7  += [float(parts[7])]
            self.Be7  += [float(parts[8])]
            self.Be9  += [float(parts[9])]
            self.B8   += [float(parts[10])]
            self.B10  += [float(parts[11])]
            self.B11  += [float(parts[12])]
            self.C11  += [float(parts[13])]
            self.C12  += [float(parts[14])]
            self.C13  += [float(parts[15])]
            self.N12  += [float(parts[16])]
            self.N14  += [float(parts[17])]
            self.N15  += [float(parts[18])]
            self.O16  += [float(parts[19])]
            self.O17  += [float(parts[20])]
            self.O18  += [float(parts[21])]
            self.Ne20 += [float(parts[22])]
            self.Ne21 += [float(parts[23])]
            self.Ne22 += [float(parts[24])]
            self.Na23 += [float(parts[25])]
            self.Mg24 += [float(parts[26])]
            self.Mg25 += [float(parts[27])]
            self.Mg26 += [float(parts[28])]
            self.Al27 += [float(parts[29])]
            self.Si28 += [float(parts[30])]
            self.Si29 += [float(parts[31])]
            self.Si30 += [float(parts[32])]
            self.Fe56 += [float(parts[33])]
            self.F19  += [float(parts[34])]
            self.Al26 += [float(parts[35])]

        #converting from -log(abundances) to abundances
        self.H1   = 10**-n.array(self.H1)
        self.H2   = 10**-n.array(self.H2)
        self.He3  = 10**-n.array(self.He3)
        self.He4  = 10**-n.array(self.He4)
        self.Li6  = 10**-n.array(self.Li6)
        self.Li7  = 10**-n.array(self.Li7)
        self.Be7  = 10**-n.array(self.Be7)
        self.Be9  = 10**-n.array(self.Be9)
        self.B8   = 10**-n.array(self.B8)
        self.B10  = 10**-n.array(self.B10)
        self.B11  = 10**-n.array(self.B11)
        self.C11  = 10**-n.array(self.C11)
        self.C12  = 10**-n.array(self.C12)
        self.C13  = 10**-n.array(self.C13)
        self.N12  = 10**-n.array(self.N12)
        self.N14  = 10**-n.array(self.N14)
        self.N15  = 10**-n.array(self.N15)
        self.O16  = 10**-n.array(self.O16)
        self.O17  = 10**-n.array(self.O17)
        self.O18  = 10**-n.array(self.O18)
        self.Ne20 = 10**-n.array(self.Ne20)
        self.Ne21 = 10**-n.array(self.Ne21)
        self.Ne22 = 10**-n.array(self.Ne22)
        self.Na23 = 10**-n.array(self.Na23)
        self.Mg24 = 10**-n.array(self.Mg24)
        self.Mg25 = 10**-n.array(self.Mg25)
        self.Mg26 = 10**-n.array(self.Mg26)
        self.Al27 = 10**-n.array(self.Al27)
        self.Si28 = 10**-n.array(self.Si28)
        self.Si29 = 10**-n.array(self.Si29)
        self.Si30 = 10**-n.array(self.Si30)
        self.Fe56 = 10**-n.array(self.Fe56)
        self.F19  = 10**-n.array(self.F19)
        self.Al26 = 10**-n.array(self.Al26)    

        f.close()

    def plot(self, minimum=1E-3):
        """plot isotope abundances with a given minimum"""

        isotopes = [self.H1, self.H2, self.He3, self.He4, self.Li6,
                    self.Li7, self.Be7, self.Be9, self.B8, self.B10,
                    self.B11, self.C11, self.C12, self.C13, self.N12,
                    self.N14, self.N15, self.O16, self.O17, self.O18,
                    self.Ne20, self.Ne21, self.Ne22, self.Na23, self.Mg24,
                    self.Mg25, self.Mg26, self.Al27, self.Si28, self.Si29,
                    self.Si30, self.Fe56, self.F19, self.Al26]

        for i in range(len(isotopes)):
            if isotopes[i].max()>minimum:
                semilogy(self.time, isotopes[i], label=self.names[i])

        axis([min(self.time),max(self.time),minimum,1])
        xlabel('Time [yr]')
        ylabel('Abundance fraction')
        legend(loc=(1.01,0), handlelen=0.025)

class Rotation:
    """Class for reading *.rot[12] files
    containing rotation related information."""

    def __init__(self, filename):
        """initialization by reading from filename"""
        f = open(filename)
        
        #initialize arrays
        self.time = []
        self.Porb = []
        self.vsurf = []
        self.Pspin = []
        self.Lstar = []
        self.Lstar_Lsync = []
        self.aw2 = []
        
        #fill arrays
        for line in f:
            line = fixline(line)
            
            parts = line.split()
            
            self.time += [float(parts[0])]
            self.Porb += [float(parts[1])]
            self.vsurf += [float(parts[2])]
            self.Pspin += [float(parts[3])]
            self.Lstar += [float(parts[4])]
            self.Lstar_Lsync += [float(parts[5])]
            self.aw2 += [float(parts[8])]

class Settings:
    """Read and write m.dat settings files"""

    def get_raw(self, lines, row, column):
        """return the value of an option from
        row, column in an array of lines"""
        return line[row][7+column*19]

    def get_int(self, lines, row, column):
        """return the integer value of an option from
        row, column in an array of lines"""
        return int(self.get_raw(lines, row, column))

    def get_float(self, lines, row, column):
        """return the float value of an option from
        row, column in an array of lines"""
        s = self.get_raw(lines, row, column)
        s.replace('D', 'E')#fix
        s.replace('d', 'e')
        return float(s)

    def get_string(self, lines, row, column):
        """return the string value of an option from
        row, column in an array of lines"""
        return self.get_raw(lines, row, column).strip()

    def __init__(self, filename):
        """Initialize using filename"""
        lines = open(filename).readlines()

        #Numerics
        self.FNAME      = self.get_string(lines, 1, 0)
        self.NR         = self.get_int(lines, 1, 1)
        self.IOUT       = self.get_int(lines, 1, 2)
        self.IPRN       = self.get_int(lines, 1, 3)
        self.MAXZAL     = self.get_int(lines, 2, 0)
        self.FNET       = self.get_int(lines, 2, 1)
        self.FTSH       = self.get_float(lines, 2, 2)
        self.FTSHE      = self.get_float(lines, 2, 3)
        self.FTS        = self.get_float(lines, 3, 0)
        self.FTSDM      = self.get_float(lines, 3, 1)
        self.ECHB       = self.get_float(lines, 3, 2)
        self.ECHEB      = self.get_float(lines, 3, 3)
        self.MAXIMP     = self.get_int(lines, 4, 0)
        self.NFMDOT     = self.get_int(lines, 4, 1)
        self.ALPRED     = self.get_float(lines, 4, 2)
        self.IAGNOS     = self.get_int(lines, 4, 3)
        self.A2OVER     = self.get_float(lines, 5, 0)
        self.IEG        = self.get_int(lines, 5, 1)
        self.EFMTU      = self.get_int(lines, 5, 2)
        self.FTLOB      = self.get_int(lines, 5, 3)
        self.HT         = self.get_float(lines, 6, 0)
        self.PLOTF      = self.get_int(lines, 6, 1)
        self.ALPVAR     = self.get_int(lines, 6, 2)
        self.hilfpa     = self.get_int(lines, 6, 3)
        self.FWIND      = self.get_float(lines, 7, 0)
        self.FDIFF      = self.get_float(lines, 7, 1)
        self.NCONV      = self.get_int(lines, 7, 2)

        #Grid parameters
        self.DLNTMA     = self.get_float(lines, 10, 0)
        self.DLNTMI     = self.get_float(lines, 10, 1)
        self.XRESMI     = self.get_float(lines, 10, 2)
        self.LCORE      = self.get_string(lines, 10, 3)
        self.CRESMI     = self.get_float(lines, 11, 0)
        self.CRESRA     = self.get_float(lines, 11, 1)
        self.NGHIS      = self.get_int(lines, 11, 2)
        self.NGHISI     = self.get_int(lines, 11, 3)
        self.NGINS      = self.get_int(lines, 12, 0)
        self.NGINSI     = self.get_int(lines, 12, 1)
        self.NGDEL      = self.get_int(lines, 12, 2)
        self.NGDELI     = self.get_int(lines, 12, 3)
        self.IRADFX     = self.get_int(lines, 13, 0)
        self.IINMAX     = self.get_int(lines, 13, 1)
        self.IOUMAX     = self.get_int(lines, 13, 2)

        #Physics
        self.FOVER      = self.get_float(lines, 15, 0)
        self.L_Hp       = self.get_float(lines, 15, 1)
        self.CAPMIN     = self.get_float(lines, 15, 2)
        self.IHAY       = self.get_int(lines, 15, 3)
        self.AHAY       = self.get_float(lines, 16, 0)
        self.BHAY       = self.get_float(lines, 16, 1)
        self.FJACC      = self.get_float(lines, 16, 2)
        self.IP         = self.get_float(lines, 16, 3)
        self.ICON       = self.get_int(lines, 17, 0)
        self.MTU        = self.get_int(lines, 17, 1)
        self.FC12A      = self.get_float(lines, 17, 2)
        self.SEM        = self.get_float(lines, 17, 3)
        self.THC        = self.get_float(lines, 18, 0)
        self.PBETA      = self.get_float(lines, 18, 1)
        self.PALPHA     = self.get_float(lines, 18, 2)
        self.IAL        = self.get_int(lines, 18, 3)
        self.DEPS       = self.get_float(lines, 19, 0)
        self.PPRED      = self.get_int(lines, 19, 1)
        self.HACMIN     = self.get_int(lines, 19, 2)
        self.WWD        = self.get_int(lines, 19, 3)
        self.FTMIN      = self.get_float(lines, 20, 0)
        self.FPMIN      = self.get_float(lines, 20, 1)
        self.FTWARN     = self.get_float(lines, 20, 2)
        self.FPWARN     = self.get_float(lines, 20, 3)
        self.IDMROT     = self.get_int(lines, 21, 0)
        self.DMSFAC     = self.get_float(lines, 21, 1)
        self.DMSREM     = self.get_float(lines, 21, 2)
        self.DMSKHF     = self.get_float(lines, 21, 3)
        self.DMSDYF     = self.get_float(lines, 22, 0)
        self.IKAP       = self.get_int(lines, 22, 1)
        self.DMDT       = self.get_float(lines, 22, 2)
        self.IDMDT      = self.get_int(lines, 22, 3)
        self.GAMTAU     = self.get_float(lines, 23, 0)
        self.IGAMME     = self.get_int(lines, 23, 1)
        self.LBV        = self.get_float(lines, 23, 2)
        self.FSYNC      = self.get_float(lines, 23, 3)
        self.ALFRAC     = self.get_float(lines, 24, 0)
        self.IVISC      = self.get_int(lines, 24, 1)
        self.IGRW       = self.get_int(lines, 24, 2)
        self.FJFAC      = self.get_float(lines, 24, 3)

        #Rotational mixing
        self.NMIX       = self.get_float(lines, 26, 0)
        self.FMY        = self.get_float(lines, 26, 1)
        self.FC         = self.get_float(lines, 26, 2)
        self.FJC        = self.get_float(lines, 26, 3)
        self.FJDSI      = self.get_float(lines, 27, 0)
        self.FJSHI      = self.get_float(lines, 27, 1)
        self.FJSSI      = self.get_float(lines, 27, 2)
        self.FJEZ       = self.get_float(lines, 27, 3)
        self.FJGSF      = self.get_float(lines, 28, 0)
        self.RCRIT      = self.get_float(lines, 28, 1)
        self.RICRIT     = self.get_float(lines, 28, 2)
        self.ANGSMT     = self.get_float(lines, 28, 3)
        self.ANGSMM     = self.get_float(lines, 29, 0)
        self.ANGSML     = self.get_float(lines, 29, 1)
        self.NAMSMG     = self.get_float(lines, 29, 2)
        self.MAGNET     = self.get_float(lines, 31, 0)
        self.MAGFMU     = self.get_float(lines, 31, 1)
        self.MAGFT      = self.get_float(lines, 31, 2)
        self.MAGFNU     = self.get_float(lines, 31, 3)
        self.MAGFDF     = self.get_float(lines, 32, 0)
        self.NR1        = self.get_float(lines, 34, 0)
        self.NR2        = self.get_float(lines, 34, 1)
        self.PORS       = self.get_float(lines, 34, 2)
        self.CMASS      = self.get_float(lines, 34, 3)
        self.INIT       = self.get_float(lines, 35, 0)
        self.ORBSEP     = self.get_float(lines, 35, 1)
        self.ZMETAL     = self.get_float(lines, 35, 2)
        self.VSURF      = self.get_float(lines, 35, 3)
        self.DTMIN      = self.get_float(lines, 37, 0)
        self.DTMAX      = self.get_float(lines, 37, 1)
        self.DTIN       = self.get_float(lines, 37, 2)
        self.ALPHA      = self.get_float(lines, 37, 3)
        self.F          = self.get_float(lines, 38, 0)
        self.TKGEL      = self.get_float(lines, 38, 1)
        self.LEN        = self.get_float(lines, 38, 2)
        self.GMD        = self.get_float(lines, 38, 3)
        self.ITMIN      = self.get_float(lines, 39, 0)
        self.ITMAX      = self.get_float(lines, 39, 1)
        self.ITERDT     = self.get_float(lines, 39, 2)
        self.XMREG0     = self.get_float(lines, 39, 3)
        self.NOQ        = self.get_float(lines, 40, 0)
        self.DYNFAK     = self.get_float(lines, 40, 1)
        self.ISB        = self.get_float(lines, 40, 2)
        self.DTMIN      = self.get_float(lines, 42, 0)
        self.DTMAX      = self.get_float(lines, 42, 1)
        self.DTIN       = self.get_float(lines, 42, 2)
        self.ALPHA      = self.get_float(lines, 42, 3)
        self.F          = self.get_float(lines, 43, 0)
        self.TKGEL      = self.get_float(lines, 43, 1)
        self.LEN        = self.get_float(lines, 43, 2)
        self.GMD        = self.get_float(lines, 43, 3)
        self.ITMIN      = self.get_float(lines, 44, 0)
        self.ITMAX      = self.get_float(lines, 44, 1)
        self.ITERDT     = self.get_float(lines, 44, 2)
        self.XMREG0     = self.get_float(lines, 44, 3)
        self.NOQ        = self.get_float(lines, 45, 0)
        self.DYNFAK     = self.get_float(lines, 45, 1)
        self.ISB        = self.get_float(lines, 45, 2)

class Isotope:
    """
    Contains information about 1 isotope.
    """

    def __init__(self, name, A, Z):
        """
        Set the name, number of nuclei A and number of protons Z.
        """
        self.name = name
        self.A = A
        self.Z = Z

    def __repr__(self):
        """
        Return a string representation of this Isotope.
        """
        return 'Isotope(%s, %i, %i)' % (self.name, self.A, self.Z)

    def __str__(self):
        """
        Return a string representing the contents of this Isotope.
        """
        return self.name

def load_isotopes(filename=None):
    """
    Return an array of Isotopes as used by BEC.
    """
    isotopes = {}
    
    if not filename:
        # Load from default file using $BEC_ROOT
        filename = os.environ['BEC_ROOT'] + '/data/chemie.dat'

    f = open(filename)
    for i in range(36):
        line = f.readline()
        nr = int(line[:4])
        name = line[5:10].strip()
        A = int(float(line[12:15]))
        Z = int(float(line[16:20]))
        isotopes[nr] = Isotope(name, A, Z)

    return isotopes
        
