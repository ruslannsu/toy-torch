import numpy as np


class Parameter:
    layers = []
    calling = {}
    
    def __init__(self, data: list):
        Parameter.calling[data[0]] = data[1]
        Parameter.layers.append(data[0])
    
    

    