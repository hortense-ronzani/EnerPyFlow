import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# Skip modules installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# colors that will be used for the plots
COLORS = ['#BFA85B',
 '#6C3E80',
 '#219EBC',
 '#484C7E',
 '#C7754C',
 '#EC674E',
 '#82A9A1',
 '#B41F58',
 '#4A62AB',
 '#ef476f',
 '#012640',
 '#FFBC42',
 '#0496FF',
 '#D81159',
 '#8F2D56',
 '#006BA6',
 '#FFFFFF',
]

def get_chronicle(value, log=False, i=None):
    """
    From a value in a configuration file, returns the value if it's raw data, and gets raw data indicated by
    the value if value is a path to a column of a csv file.

    Args:
        value (str | int | float): Data or path to location to get data from.
        log (bool): If True, additionnal information will be printed throughout the process.
        i (int): Index used to iterate within a list of specifications.

    Returns:
        list[float] | str | int | float: Value(s).
    """
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
        
def bound(value, t):
    """
    If value described the value of a function f depending on time t, returns f(t).
    If value is a constant, returns value. If value is a list, returns element number t of the list.

    Args:
        value (any, list[any]): Value or list of values.
        t(int): Timestep or index.
    
    Returns:
        any: f(t)
    """
    try:
        return value[t]
    except TypeError:
        return value
    
def get_components(file_of_components_list, components_class):
    """
    Gets information regarding components belonging to a type of base components.

    Args:
        file_of_components_list (str): Path to a yaml components list configuration file.
        components_class (str): Name of the class of base components to be found (Source, Storage, Demand, Converter).

    Returns:
        dict[str: any]: Dictionnary with component names related to type components_class as keys, with their configuration
            information.
    """
    with open(file_of_components_list, 'r') as file:
        components_list = yaml.safe_load(file)
    return components_list[components_class]

def value(flow_vars_t):
    """
    Gets the numerical value of the element.

    Args:
        flow_vars_t(float | int | pulp.LpVariable): Element to get the value from.

    Returns:
        float | int: Value of the element.
    """
    # flow_t is not a var but an input data
    if type(flow_vars_t) == float or type(flow_vars_t) == np.float64 or type(flow_vars_t) == int:
        return flow_vars_t
    # flow_t is a pl.LpVariable
    else:
        return flow_vars_t.value()
    
def get_from_elements_list(keyword, elements_list):
    """
    Looks for lines in a file that contains a keyword, a returns the value of the elements describing it.
    Ex: keyword = 'energy' ; line = 'energy: electricity' ; then this function returns ['electricity'].

    Args:
        keyword (str): The keyword to look for.
        elements_list (str): Path to the file.

    Returns:
        list[str]: Lis of elements found associated to the keyword.
    """
    energies = []
    with open(elements_list, 'r') as file:
        data = file.read()
        parameters = data.split('\n')
        for parameter in parameters:
            try:
                if parameter[0] == '#':
                    continue
                if keyword in parameter:
                    energies.append(parameter.split(':')[1][2:-1])
            except IndexError:
                pass
        energies = np.unique(energies)
        l = []
        for energy in energies:
            l.append(str(energy))
    return l

def get_label(string):
    """
    Transforms a variable name into a label to be displayed by replacing '_' character by spaces.

    Args:
        string(str): A string to be transformed.

    Returns:
        str: String transformed.
    """
    words = string.split('_')
    translation = ''
    for word in words:
        translation += word + ' '
    return translation[:-1]

def capitalise(string):
    """
    Capitalises only the first letter of a string, leaving the other characters unchanged.

    Args:
        string(str): A string to be transformed.

    Returns:
        str: String transformed.
    """
    return string[0].upper() + string[1:]

def plot_one_hub(dispatch, top_var, variables, ax, title,
                                    price=None, co=False, env1_env2=None, env1='', env2='',
                                    unit=None, start=0, end=0):
    """
    Plot the flow variables linked to a hub depending on the time as a cumulated step plot.

    Args:
        dispatch (pandas.Dataframe): Dataframe containing data to be displayed.
        top_var (str: None): Name of a variable stored in dispatch to be displayed as a line plot with the
            same unit as the main variables. Must match a column name of dispatch.
        variables (list[str]): List of names of variables to be displayed, must match some column names of
            dispatch.
        ax (matplotlib.axes._axes.Axes): ax object to add the plots to.
        title (str): Title of the subplot.
        price (tuple(str | float, str)| None): If not None, additional line to be plotted on a second axis on all subplots.
            Must be under the form (path_to_column | value). Ex: ('data_sample.csv//Electricity_price (euros/MWh)',
            'Electricity price (€/MWh)').
        co (bool): If True, periods of connection between two environments will be shown on all subplots.
        env1_env2 (list[float]): Timeseries of the periods of connection between environment1 and environment2.
        env1 (str): If co==True, name of the 1st environment of the connection to be displayed.
        env2 (str): If co==True, name of the 2nd environment of the connection to be displayed.
        unit (str): Unit of the data.
        start (int): Index of the data where to start to plot. Ex: start=2 -> start at the second value.
        end (int): len(dispatch)-end is the index of the data where to stop to plot Ex: stop=1 -> stop at
            the penultimate value.

    Returns:
        ax (matplotlib.axes._axes.Axes): ax object with the plots added.
    """
    dispatch = dispatch[start:len(dispatch)-end]

    variables = [var for var in variables if var in dispatch.columns]

    # Get max and sum values
    max_value = dispatch[variables].max().max()
    positive_total = dispatch[variables].clip(lower=0).sum().sum()
    negative_total = dispatch[variables].clip(upper=0).sum().sum()
    
    # Indicate when car is connected
    if co:
        env1_env2 = env1_env2[start:len(env1_env2)-end]
        ax.fill_between(x=dispatch['time'], y1=env1_env2*max_value/max(env1_env2),
                        step="mid", alpha=0.2, color='#219EBC', hatch='/', edgecolor='w',
                        label='Connection between ' + env1 + ' and ' + env2)
    
    # Plot the aggregated variable as a bold line
    if type(top_var)==str:
        ax.step(x=dispatch['time'], y=dispatch[top_var], label=capitalise(get_label(top_var)), lw=2., where='mid', zorder=2)
    
    # Stack plot of the components variables
    positive_stack = np.zeros(len(dispatch))
    negative_stack = np.zeros(len(dispatch))

    zorders = [len(variables)-i+3 for i in range(len(variables))]
    for i, var in enumerate(variables):
        positive_values = dispatch[var].clip(lower=0)
        negative_values = dispatch[var].clip(upper=0)
        if top_var=='PV' or top_var=='conso':
            rate = dispatch[var].sum()/top_var
            label = capitalise(get_label(var)) + ' (' + str(round(rate*100)) + '%)'
        else:
            try:
                positive_rate = positive_values.sum()/positive_total
            except ZeroDivisionError:
                positive_rate = 0.
            try:
                negative_rate = negative_values.sum()/negative_total
            except ZeroDivisionError:
                negative_rate = 0.
            label = capitalise(get_label(var)) + ' (' + str(round(positive_rate*100)) + '%$\\uparrow$ ' + str(round(negative_rate*100)) + '%$\\downarrow$)'

        # Positive values
        ax.bar(x=dispatch['time'], height=positive_values,
            color=COLORS[i], label=label,
            width=dispatch['time'].values[1] - dispatch['time'].values[0], zorder=zorders[i], bottom=positive_stack)
        # Negative values
        ax.bar(x=dispatch['time'], height=negative_values,
            color=COLORS[i], # label=get_label(var) + ' (' + str(round(rate*100)) + '%)',
            width=dispatch['time'].values[1] - dispatch['time'].values[0], zorder=zorders[i], bottom=negative_stack)
        
        positive_stack = positive_stack + positive_values
        negative_stack = negative_stack + negative_values

    # Plot electricity prices with its own axis
    if price is not None:

        value, label = price
        price = get_chronicle(value=value)
        price = price[start:len(price)-end]
        ax2 = ax.twinx()
        ax2.step(x=dispatch['time'], y=price, color='#333333', alpha=0.7, where='mid', label=label)
        ax2.set_ylabel('Electricity price (€/MWh)', fontsize=18)
        ax2.spines[['left', 'top']].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ymin2, ymax2 = price.min(), price.max()
        YMIN = ymin*(ymax2-ymin2)/(ymax-ymin)*3
        YMAX = ymax*(ymax2-ymin2)/(ymax-ymin)*3
        ax2.set_ylim(YMIN, YMAX)

    ax.set_xlim(dispatch['time'].values[0], dispatch['time'].values[-1])
    ax.set_ylabel(unit, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.spines[['right', 'top']].set_visible(False)

    return ax

def get_timeline(time):
    """
    Builds a timeline in the list of datetime format from a list of strings.

    Args:
        time (list[str]): List of strings of the form '%Y%m%d:%H' (ex: 20252909:12 = September 29th, 2025 12h).

    Returns:
        list[datetime]: List of datetime objects.
    """
    new_time = np.empty(len(time), dtype=datetime)
    for i, t in enumerate(time):
        new_time[i] = datetime.strptime(t, '%Y%m%d:%H')
    return new_time