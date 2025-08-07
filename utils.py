import pandas as pd

# Skip utils installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

def get_chronicle(name, description):
    # name = 'minimum' | 'maximum' | 'cost' | 'value' ...
    type = description[name + '_type']
    if type == 'constant':
        return description[name]
    elif type == 'hourly':
        file, column = description[name].split('//')
        data = pd.read_csv(file, sep=';')
        return data[column]
    else:
        print('Invalid ' + name + 'type for this technology' )

# Déso pour ça
def get_chronicle2(path, type):
    if type == 'constant':
        return path
    elif type == 'hourly':
        file, column = path.split('//')
        data = pd.read_csv(file, sep=';')
        return data[column]
    else:
        print('Invalid ' + path + ' chronicle' )