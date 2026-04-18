import numpy as np


class Parameter:
    param_list = []
    calling = {}
    def __init__(self, data: list):
        Parameter.calling[data[0]] = data[1]
        Parameter.param_list.append(data[1])
    
    

    