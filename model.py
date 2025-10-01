import pulp as pl

# skip modules installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import yaml
from tqdm import tqdm
import shutil

from datetime import datetime

import importlib

import utils
from utils import *
importlib.reload(utils)

import components
from components import *
importlib.reload(components)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Model():
    """
    Class for the modelization of an energy system from configuration files A Model objects allows to
    build linear variables and equations, solve them as a linear problem, access all information relative
    to the simulation, and display the results once solved (optimized values of all variables, value of
    the objective function).

    Attributes:
        config_file (dict): General configuration file.
        elements_list (dict): Description of all base components of the system.
        run_num (int): Number of the run, for identification and saving purpose.
        run_name (str): Name of the run, for identification and saving purpose.
        energies (list[str]): List of all possible energy types.
        environments (list[str]): List of all possible environments.
        data (pandas.Dataframe | NoneType): [Optional] Dataframe with input data.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        time (List[str]): List of length nb_of_timesteps with timestamps.
        model (pulp.LpProblem): Pulp linear formulation of the optimization problem.
        components (dict[str: dict[str: Components]]): Dictionnary of all base components added to the system,
            with base components type (Source, Demand, Storage, Converter) and components names as keys.
        directory (str): Directory path to save configuration files, results and plots.
        hubs (pandas.Dataframe): Data table with index=environments and columns=energies representing of all
            possible hubs, filled with Hub(environment, energy) objects, and None when it doesn't exist.
        environmentsConnections (dict[str: EnvironmentsConnection]) : Dictionnary of all environments connections,
            represented by EnvironmentsConnection objects.
        simu_time (datetime): Timestamp of the instant where the optimization is completed.
        objective_value (float): Value of the objective function once the optimization is completed.
        dispatch (pandas.Dataframe): Dataframe of length nb_of_timesteps with flow variables and SOC values (optimal
            solution of the problem).

    Args:
        config_file (str): Path to general yaml configuration file.
        elements_list (str): Path to elements list yaml configuration file.
        data (str): [Optional] Path to input data csv file. Could be usefull to display information relative to input data in
            results plot.
    """
    def __init__(self, config_file='general_config_file.yaml', elements_list='elements_list.yaml', data=''):
        with open(config_file, 'r') as file:
            self.config_file = yaml.safe_load(file)
            self.config_file = self.config_file
        self.run_num = self.config_file['run_num']
        self.run_name = self.config_file['run_name']
        self.energies = get_from_elements_list('energy', elements_list)
        self.environments = get_from_elements_list('environment', elements_list)
        try:
            self.data = pd.read_csv(data, sep=';')
        except FileNotFoundError:
            self.data = None
        self.time = utils.get_chronicle(self.config_file['time'])
        self.elements_list = elements_list
        if self.config_file['optimization_sense'] == 'minimize':
            sense = pl.LpMinimize
        elif self.config_file['optimization_sense'] == 'maximize':
            sense = pl.LpMaximize
        self.components = {}
        self.nb_of_timesteps = len(self.time)

        # Create a directory for this run
        self.directory = str(self.run_num) + '_' + self.run_name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else :
            print('Warning : overwriting run ' + self.directory)

        self.model = pl.LpProblem(self.directory, sense)
        self.model.objective = pl.LpAffineExpression()

        # Save config files
        shutil.copy(config_file, self.directory + '/' + config_file)
        shutil.copy(elements_list, self.directory + '/' + elements_list)
    
    def initialize_hubs(self):
        """
        Creates hubs attribute.
        """
        self.hubs = pd.DataFrame(index=self.environments,
                                 columns=self.energies)
        
    def build_environment_level_variables_and_constraints(self, log=False):
        """
        Builds all base components (Source, Demand, Storage, Converter), including their variables and internal constraints
        (example: dispatching equations for dispatchable Demand objects, storage equations for Storage objects, conversion
        equations for Conversion objects). Updates hubs table, components dictionnary, and model.
        TO BE USED AFTER initialize_hubs().

        Args:
            log (bool): If True, additional information will be printed throughout the process.
        """
        # Different types of components available
        classes = {'Source': Source,
                   'Demand': Demand,
                   'Storage': Storage,
                   'Converter': Converter,}
        # For each type of component:
        for class_ in classes.keys():
            # Get specs from the config file
            try : components_specs = get_components(self.elements_list, class_)
            except KeyError :
                if log:
                    print('No ' + class_ + ' found.')
                continue # If no Storage or Converter or ... element, do nothing and continue
            self.components.update({class_: {}})
            # For each component of this type:
            for component_name in components_specs.keys():
                if components_specs[component_name]['activate']: # If component is activated
                    if log:
                        print('Found ' + component_name)
                    # Create object (Source, Demand, Storage or Converter) from the specs
                    component = classes[class_](component_name,
                                                components_specs[component_name],
                                                nb_of_timesteps=self.nb_of_timesteps,
                                                log=log,)
                    # Link the component object to its hub(s)
                    try:
                        self.hubs, self.model = component.build_equations(hubs=self.hubs, model=self.model, log=log)
                    except AttributeError:
                        raise Exception("ERROR: Model.hubs doesn't exist. Run Model.initialize_hubs() first.")
                    # Save the component
                    self.components[class_].update({component_name: component})

    def connect_environments(self, log=False):
        """
        Builds flow variables between different environments according to the connections descriptions given in the general
        configuration file. Store EnvironmentsConnection object in environmentsConnections dictionnary. Updates hubs table
        and model.

        Args:
            log (bool): If True, additional information will be printed throughout the process.
        """
         # Get connections description
        environments_connections = self.config_file['environments_connections']
        self.environmentsConnections = {}

        if environments_connections is None:
            print('No environments connections')

        else:
            for environment1 in environments_connections.keys():
                descriptions = environments_connections[environment1]
                envs = descriptions['envs']
                
                for i, environment2 in enumerate(envs):
                    co = EnvironmentsConnection(environment1, environment2, descriptions, i, self.nb_of_timesteps, log=log)
                    self.hubs, self.model = co.connect_as_input(self.hubs, self.model, log=log)
                    self.environmentsConnections.update({environment1 + '_' + environment2: co})

    def add_hubs_equations_to_model(self, log=False):
        """
        Adds hubs equations to the model, once all base components are added to the hubs equations. Updates model.

        TO BE USED AFTER build_environment_level_variables_and_constraints() AND connect_environments().

        Args:
            log (bool): If True, additional information will be printed throughout the process.
        """
        # Check that some components are added:
        if self.components == {}:
            raise Exception("ERROR: Model with no components.")
        # Check that environments connections are added:
        try:
            self.environmentsConnections
        except AttributeError:
            print('WARNING: No environments connection. To add environments connections, run Model.connect_environments() first.')
        # For each possible hub:
        for environment in self.environments:
            for energy in self.energies:
                hub = self.hubs.loc[environment, energy]
                # If the hub exists
                if type(hub) == Hub or type(hub) == components.Hub:
                    # Choose to display logs or not
                    if log:
                        print(hub.name)
                    # Add its equation to the model
                    for hub_equation in hub.equation:
                        self.model += hub_equation
                    
    def solve(self, log=False):
        """
        Solves the pulp problem, and save flow variables and SOC values in dispatch.

        Args:
            log (bool): If True, additional information will be printed throughout the process.
        """
        # Call solver and solve the problem
        self.model.solve(pl.PULP_CBC_CMD(timeLimit=20, msg=True))
        print('Problem solved with status: ' + pl.LpStatus[self.model.status])
        # Record timestamp of simulation
        self.simu_time = datetime.now()
        # Record the optimal value of the objective function
        self.objective_value = pl.value(self.model.objective)

        # Record the dispatching solution (= all optimal values of the problem variables)
        self.dispatch = pd.DataFrame(index=[t for t in range(self.nb_of_timesteps)])
        
        # Get primary energy flows values
        for class_ in ['Source', 'Demand', 'Storage', 'Converter']:
            if log:
                print(class_)
            columns = self.components[class_].keys()
            for column in columns:
                if log:
                    print(column)
                self.dispatch.insert(loc=0, column=column, value=np.zeros(self.nb_of_timesteps))
                if not class_=='Converter':
                    for t in tqdm(self.dispatch.index):
                        self.dispatch.loc[t, column] = (value(self.components[class_][column].flow_vars_out[t])
                                                        - value(self.components[class_][column].flow_vars_in[t]))
                elif class_=='Converter':
                    for t in tqdm(self.dispatch.index):
                        self.dispatch.loc[t, column] = - value(self.components[class_][column].flow_vars_in[t])

        # Get SOC values
        try:
            # storage_components = get_components(self.elements_list, 'Storage').keys()
            storage_components = self.components['Storage'].keys()
            for component in storage_components:
                self.dispatch.insert(loc=0,
                                    column=component + '_SOC',
                                    value=[value(self.components['Storage'][component].SOC[t]) for t in range(self.nb_of_timesteps)])
        except KeyError:
            pass
            
        # Get output flows from conversion devices
        # conversion_components = get_components(self.elements_list, 'Conversion').keys()
        conversion_components = self.components['Converter'].keys()
        for component in conversion_components:
            for output_energy in self.components['Converter'][component].output_energies.keys():
                self.dispatch.insert(loc=0,
                                    column=component + '_' + output_energy,
                                    value=[value(self.components['Converter'][component].flow_vars_out[output_energy][t]) for t in range(self.nb_of_timesteps)])
        
        # Get exchanges between environments
        for name in self.environmentsConnections.keys():
            environments_connection = self.environmentsConnections[name]
            environment1 = environments_connection.environment1
            environment2 = environments_connection.environment2

            for energy in environments_connection.vars_out.keys():
                self.dispatch.insert(loc=0,
                                     column=energy + '_'
                                     + environment1 + '_hub_to_'
                                     + environment2 + '_hub',
                                     value=[(value(environments_connection.vars_out[energy][t])
                                             - value(environments_connection.vars_in[energy][t])) for t in range(self.nb_of_timesteps)])
                self.dispatch.insert(loc=0,
                                     column=energy + '_'
                                     + environment2 + '_hub_to_'
                                     + environment1 + '_hub',
                                     value=[(value(environments_connection.vars_in[energy][t])
                                             - value(environments_connection.vars_out[energy][t])) for t in range(self.nb_of_timesteps)])

        # Get hubs losses values
        # For each possible hub:
        for environment in self.environments:
            for energy in self.energies:
                hub = self.hubs.loc[environment, energy]
                # If the hub exists
                if type(hub) == Hub or type(hub) == components.Hub:
                    self.dispatch.insert(loc=0,
                                         column='unused_' + environment + '_' + energy,
                                         value=[-value(hub.unused_energy[t]) for t in range(self.nb_of_timesteps)])

        # Add time
        self.dispatch.insert(loc=0,
                             column='time',
                             value=self.time)

        # Save in a .csv file
        solution_file = str(self.run_num) + '_' + self.run_name + '_dispatch.csv'
        self.dispatch.to_csv(self.directory + '/' + solution_file)

        # Prepare self.dispatch columns for plot
        self.dispatch.time = get_timeline(self.time)
        self.dispatch = self.dispatch.astype({'time': 'datetime64[ns]'})

    def hub_vars(self, environment, energy):
        """
        Gets all flow variable names linked to the hub(environment, energy.)

        Args:
            environment (str): Environment name.
            energy (str): Energy type name.

        Returns:
            list[str]
        """
        # List of flow variables linked to Hub(environment, energy)
        list_of_vars = []
        # Get connections between environments 
        for name in self.environmentsConnections.keys():
            co = self.environmentsConnections[name]
            if co.environment1 == environment:
                for energy_type in co.vars_in.keys():
                    if energy_type == energy:
                        list_of_vars.append('_'.join([energy, co.environment2, 'hub', 'to', environment, 'hub']))
            elif co.environment2 == environment:
                for energy_type in co.vars_out.keys():
                    if energy_type == energy:
                        list_of_vars.append('_'.join([energy, co.environment1, 'hub', 'to', environment, 'hub']))
            else: pass
        # Get components flow variables linked to Hub(environment, energy)
        list_of_vars = self.hubs.loc[environment, energy].component_names + list_of_vars
        # Get energy losses of Hub(environment, energy)
        list_of_vars.append('unused_' + environment + '_' + energy)

        return  list_of_vars
    
    def plot_hubs(self, save=False, log=False, co=False, env1='', env2='', price=None, unit='', start=0, end=0):
        """
        Plots energy flows of all hubs that exist, depending on the time.

        Args:
            save (bool): If True, plot will be saved as a png file in the run directory.
            log (bool): If True, additional information will be printed throughout the process.
            co (bool): If True, periods of connection between two environments will be shown on all subplots.
            env1 (str): If co==True, name of the 1st environment of the connection to be displayed.
            env2 (str): If co==True, name of the 2nd environment of the connection to be displayed.
            price (tuple(str | float, str)| None): If not None, additional line to be plotted on a second axis on all subplots.
                Must be under the form (path_to_column | value). Ex: ('data_sample.csv//Electricity_price (euros/MWh)', 'Electricity price (€/MWh)').
            unit (str): Unit of energy flows, for legend.
            start (int): Index of the data where to start to plot. Ex: start=2 -> start at the second value.
            end (int): len(dispatch)-end is the index of the data where to stop to plot Ex: stop=1 -> stop at
                the penultimate value.
        """
        # One plot per hub that exists
        nb_of_plots = self.hubs.notna().sum().sum()
        if log:
            print(str(nb_of_plots) + ' subplot(s) to be plotted.')
        fig = plt.figure(figsize=(21, 6*nb_of_plots), layout='constrained')
        # Position of the 1st plot within the figure
        ax_position = nb_of_plots*100 + 10 + 1
        for environment in self.environments:
            for energy in self.energies:
                hub = self.hubs.loc[environment, energy]
                # If hub exists :
                if type(hub) == Hub or type(hub)==components.Hub:
                    ax = fig.add_subplot(ax_position) # Create the frame hub's plot
                    ax_position += 1 # Position of the next plot within the figure
                    title = 'Hub ' + environment + ' ' + energy
                    if log:
                        print('Plot at position number ' + str(ax_position) + ': ' + title)
                    # Get hub flows names
                    variables = list(dict.fromkeys(self.hub_vars(environment, energy)))
                    if log:
                        print(variables)
                    if co:
                        env1_env2 = self.environmentsConnections[env1 + '_' + env2].maximum_out
                    else:
                        env1_env2 = None
                    # Plot hub flows
                    ax = plot_one_hub(self.dispatch,
                                        None,
                                        variables,
                                        ax,
                                        title=title,
                                        price=price,
                                        co=co,
                                        env1_env2=env1_env2,
                                        env1=env1,
                                        env2=env2,
                                        unit=unit,
                                        start=start,
                                        end=end,)
                    # Place a legend with the subplot
                    ax.legend(loc='best', ncols=2, frameon=False, fontsize=18).set_zorder(50)
        
        if save:
            file = str(self.run_num) + '_' + self.run_name + '/' + str(self.run_num) + '_' + self.run_name + '.png'
            fig.savefig(file)
            print('Saved at: ' + file)
    
    def plot_SOC(self, variables='all', unit='', save=False, log=False):
        """
        Plots SOC values depending on the time.

        Args:
            variables (str | list[str]): If 'all', all SOC variables will be displayed. Else, list of names of variables
                to be displayed.
            unit (str): Unit of variables, for legend.
            save (bool): If True, plot will be saved as a png file in the run directory.
            log (bool): If True, additional information will be printed throughout the process.
        """
        # Time management
        self.dispatch.index = self.dispatch['time']

        # Plot SOC of all storage means on the same graph - beware of the units
        if variables == 'all':
            variables = []
            for storage_name in self.components['Storage'].keys():
                variables.append(self.components['Storage'][storage_name].name)
        
        # Create figure
        fig = plt.figure(figsize=(21, 6))
        ax = fig.add_subplot(111)
        for i, var in enumerate(variables): # For each storage
            try:
                total_capa = self.components['Storage'][var].capacity * value(self.components['Storage'][var].volume_factor)
                ax.hlines(total_capa,
                            xmin=self.dispatch['time'].values[0],
                            xmax=self.dispatch['time'].values[-1],
                            color=COLORS[i],
                            ls='--',
                            alpha=0.5,
                            lw=2.5)
                ax.plot(self.dispatch[var + '_SOC'],
                        color=COLORS[i],
                        label=get_label(var),
                        lw=2.5)
            except KeyError:
                pass
        ax.legend(fontsize=18)
        ax.set_ylabel(unit, fontsize=18)
        ax.set_xlim((self.dispatch['time'].values[0], self.dispatch['time'].values[-1]))
        ax.set_title('SOC')
        ax.spines[['right', 'top']].set_visible(False)

        if save:
            file = str(self.run_num) + '_' + self.run_name + '/' + str(self.run_num) + '_' + self.run_name + '_SOC_' + '_'.join(variables) + '.png'
            fig.savefig(file)
            print('Saved at: ' + file)

    def get_design(self, components, factors, units=None):
        """
        Shortcut to get the values of components attributes.

        Args:
            components (list[str]): List of components names.
            factors (list[str]): List of associated factors to be displayed, must be the same length as components (with typically
                "factor" or "volume_factor" as values).
            units (list[str] | None): List of associated units to be displayed, must be the same length as components if
                not None.
            
        Returns:
            pandas.Dataframe: Table filled with factors values
        """
        # Builds a table to store the results
        table = pd.DataFrame(columns=components)
        # For every component name
        for i, component in enumerate(components):
            # Find matching Component object
            for class_ in self.components.keys():
                components_of_type = self.components[class_]
                if component in components_of_type.keys():
                    table.loc[factors[i], component] = value(components_of_type[component].__getattribute__(factors[i]))
        if units is not None:
            # Add units to columns names
            table.columns = [components[i] + ' (' + units[i] + ')' for i in range(len(components))]

        return table