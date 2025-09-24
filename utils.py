import pandas as pd

# Skip utils installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

def get_chronicle(name, description, i=None):
    # name = 'minimum' | 'maximum' | 'cost' | 'value' ...
    # If description describes only one link
    if i is None:
        type = description[name + '_type']
        if type == 'constant':
            return description[name]
        elif type == 'hourly':
            file, column = description[name].split('//')
            data = pd.read_csv(file, sep=';')
            return data[column]
        else:
            print('Invalid ' + name + 'type for this technology' )
    # If description describes a list of links
    else:
        type = description[name + '_type'][i]
        if type == 'constant':
            return description[name][i]
        elif type == 'hourly':
            file, column = description[name][i].split('//')
            data = pd.read_csv(file, sep=';')
            return data[column]
        else:
            print('Invalid ' + name + 'type for this technology' )

def get_chronicle_from_path(path, type):
    if type == 'constant':
        return path
    elif type == 'hourly':
        file, column = path.split('//')
        data = pd.read_csv(file, sep=';')
        return data[column]
    else:
        print('Invalid ' + path + ' chronicle' )

def get_chronicle2(value, log=False, i=None):
    # if value is a value
    if i is None:
        if type(value) == str:
            try:
                file, column = value.split('//')
            except ValueError:
                if log:
                    print(value + ' not recognize as a dataframe column')
                return value
            data = pd.read_csv(file, sep=';')
            return data[column]
        else:
            return value
    # else, then value is a list
    else:
        if type(value[i]) == str:
            try:
                file, column = value[i].split('//')
            except ValueError:
                if log:
                    print(value[i] + ' not recognize as a dataframe column')
                return value[i]
            data = pd.read_csv(file, sep=';')
            return data[column]
        else:
            return value[i]