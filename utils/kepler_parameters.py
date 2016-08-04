# An easy way to access kepler parameters
from keppar import p, q

class ParameterMap:
    """
    Make Kepler parameters accessible by name and number.
    """

    def __init__(self, ints, floats, parameters):
        """
        Initialize a new ParameterMap using given arrays
        of floats and integers, as well as an OrderedDict
        containing parameter names and types.
        """
        self.data = {0: ints, 1: floats}
        self.parameters = parameters
    
    def get(self, key, force_type=None):
        """
        Return the value of the parameter with given key,
        which can be a number or a name.
        force_type: 0 or 1 (int or float) to force using a different type
        """
        p = self.parameters
        try:
            ind = int(key)
        except:
            ind = self.get_index(key)
        if force_type==None:
            force_type = p.values()[ind]
        return self.data[force_type][ind]
    
    def get_name(self, index):
        """
        Return the name of the parameter with given index.
        """
        p = self.parameters
        return p.keys()[index]
    
    def get_index(self, name):
        """
        Return the index of the parameter with given name.
        """
        p = self.parameters
        return p.keys().index(name)
    
    def __getitem__(self, key):
        """
        Shorthand for ParameterMap.get()
        """
        return self.get(key)
    
    def __setitem__(self, key, value):
        """
        Set the value of the parameter with name or
        number <key> to <value>.
        """
        p = self.parameters
        try:
            ind = int(key)
        except:
            ind = self.get_index(key)
        self.data[p.values()[ind]][ind] = value
