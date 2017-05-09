# Modified from https://github.com/feiyuhug/lenet-5

from numpy import *

# Build a Layer with [Size x Size]x Numbers neurons
class Layer(object) :
    def __init__(self, lay_size = []) :
        self.lay_size = lay_size
        self.maps = []
        for map_size in lay_size :
            self.maps.append(zeros(map_size))
        self.maps = array(self.maps)