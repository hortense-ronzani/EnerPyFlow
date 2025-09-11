import pulp as pl

# Skip utils installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import yaml
from tqdm import tqdm
import shutil

from datetime import datetime
#from utils import *
#import utils.get_chronicle

import utils

import importlib
importlib.reload(utils)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

M = 50000
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

class Component:
    def __init__(self, name, energy, environment, description, model=None, nb_of_timesteps=None, isHub=False, log=False):
        self.energy = energy
        self.environment = environment
        self.model = model
        self.nb_of_timesteps = nb_of_timesteps
        self.name = name

        if not isHub:
            # Get bounds of Component -> Hub variables
            try:
                self.minimum_out_type = description['minimum_out_type']
                self.minimum_out = utils.get_chronicle('minimum_out', description)
            except KeyError:
                self.minimum_out_type = 'constant'
                self.minimum_out = 0.

            try:
                self.maximum_out_type = description['maximum_out_type']
                self.maximum_out = utils.get_chronicle('maximum_out', description)
            except KeyError:
                self.maximum_out_type = 'constant'
                self.maximum_out = float('inf')

            # Get bounds of Hub -> Component variables
            try:
                self.minimum_in_type = description['minimum_in_type']
                self.minimum_in = utils.get_chronicle('minimum_in', description)
            except KeyError:
                self.minimum_in_type = 'constant'
                self.minimum_in = 0.

            try:
                self.maximum_in_type = description['maximum_in_type']
                self.maximum_in = utils.get_chronicle('maximum_in', description)
            except KeyError:
                self.maximum_in_type = 'constant'
                self.maximum_in = float('inf')

            # Get factor specifications
            try:
                self.factor = description['factor']
            except KeyError:
                self.factor = 1.
            try:
                self.factor_type = description['factor_type']
            except KeyError:
                self.factor_type = 'Continuous'
            try:
                self.factor_low_bound = description['factor_low_bound']
            except KeyError:
                self.factor_low_bound = 1.
            try:
                self.factor_up_bound = description['factor_up_bound']
            except KeyError:
                self.factor_up_bound = 1.
            if self.factor == 'auto':
                self.factor = pl.LpVariable(self.name + '_factor',
                                            lowBound=self.factor_low_bound,
                                            upBound=self.factor_up_bound,
                                            cat=self.factor_type)
                
            # Cost specifications
            # Cost of Hub -> Component
            try:
                self.cost_in_type = description['cost_in_type']
                self.cost_in = utils.get_chronicle('cost_in', description)
            except KeyError:
                self.cost_in_type = 'constant'
                self.cost_in = 0.
            # Cost of Component -> Hub
            try:
                self.cost_out_type = description['cost_out_type']
                self.cost_out = utils.get_chronicle('cost_out', description)
            except KeyError:
                self.cost_out_type = 'constant'
                self.cost_out = 0.
            # Installation cost per unit of factor
            try:
                self.installation_cost = description['installation_cost']
            except KeyError:
                self.installation_cost = 0.
        
    def get_hub(self, hubs, environment=None, energy=None, log=False):
        if environment is None:
            environment = self.environment
        if energy is None:
            energy = self.energy
        hub = hubs.loc[environment, energy]

        if type(hub) == Hub:
            if log:
                print('hub_' + environment + '_' + energy + ' already exists')
        else:
            hub = Hub(environment, energy, model=self.model, nb_of_timesteps=self.nb_of_timesteps)
            hubs.loc[environment, energy] = hub
            if log:
                print('hub_' + environment + '_' + energy + ' created')

        return hub
    
    def link(self, hubs, model, log=False):    
        self.flow_vars_in = [pl.LpVariable('hub_to_' + self.name + '_' + str(t)) for t in range(self.nb_of_timesteps)]
        self.flow_vars_out = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]

        # Add bounds to flow variables
        for t in range(self.nb_of_timesteps):
            model += self.flow_vars_out[t] <= self.factor*bound(self.maximum_out, self.maximum_out_type, t)
            model += self.flow_vars_out[t] >= self.factor*bound(self.minimum_out, self.minimum_out_type, t)
            model += self.flow_vars_in[t] <= self.factor*bound(self.maximum_in, self.maximum_in_type, t)
            model += self.flow_vars_in[t] >= self.factor*bound(self.minimum_in, self.minimum_in_type, t)

        # Add installation cost to the model objective function
        model.objective += self.factor*self.installation_cost

        # Add flow cost to the model objective function
        if model.name == 'cost':
            model.objective += pl.lpSum([self.flow_vars_out[t]*bound(self.cost_out, self.cost_out_type, t)
                                         + self.flow_vars_in[t]*bound(self.cost_in, self.cost_in_type, t) for t in range(self.nb_of_timesteps)])

        if log:
            print('Flow vars added to model')

        # Add variables to the Hub equations
        hub = self.get_hub(hubs=hubs, log=log)
        hub.add_link(variables=self.flow_vars_in, component_name=self.name, sign='-')
        hub.add_link(variables=self.flow_vars_out, component_name=self.name, sign='+')

        if log:
            print('Flow vars added to Hub equation')

        return hubs, model
    
class Hub(Component):
    def __init__(self, environment, energy, model, nb_of_timesteps, log=False):
        name = 'hub_' + energy + '_' + environment
        super().__init__(name, energy, environment, description=None, model=model, nb_of_timesteps=nb_of_timesteps, isHub=True, log=log,)

        self.unused_energy = [pl.LpVariable('unused_energy_' + self.name + '_' + str(t), lowBound=0) for t in range(self.nb_of_timesteps)]
        self.equation = [pl.LpConstraint(e=-self.unused_energy[t],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_' + str(t),
                                        rhs=0) for t in range(self.nb_of_timesteps)]
        self.component_names = []
        
    def add_link(self, variables, component_name, sign='+', log=False):
        if sign == '+':
            self.equation = [self.equation[t] + variables[t] for t in range(self.nb_of_timesteps)]
        elif sign == '-':
            self.equation = [self.equation[t] - variables[t] for t in range(self.nb_of_timesteps)]
        self.component_names.append(component_name)
   
class Convertor(Component):
    def __init__(self, name, description, model, nb_of_timesteps, log=False):
        super().__init__(name, description['input_energy'], description['environment'], description, model, nb_of_timesteps, log=log,)

        self.name = name

        self.input_energy = self.energy
        self.output_energies = description['output_energies']

        # No energy flow from Convertor to input Hub
        self.maximum_out = 0.
        
        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model=None, log=False):
        # self.model = model

        # input_hub = self.get_hub(hubs, log=log)
        
        # # Energy can only flow from the input hub to the output hubs
        # self.flow_vars_in = [pl.LpVariable(self.name
        #                                 + '_to_hub_' + self.input_energy
        #                                 + '_' + str(t),
        #                                 upBound=0) for t in range(self.nb_of_timesteps)]
        
        # # Add the flow linking convertor to its input hub to the hub equation
        # input_hub.add_link(self.flow_vars, self.name)

        # Create link from input energy hub
        hubs, model = self.link(hubs, model, log=log)

        self.equations = {}

        # Add input energy flow to the conversion equation
        # self.equation = [pl.LpConstraint(e=self.flow_vars[t],
        #                                  sense=pl.LpConstraintEQ,
        #                                  name='equation_' + self.name + '_' + str(t),
        #                                  rhs=0) for t in range(self.nb_of_timesteps)]

        # Crush flow_vars_out previously linked to input hub (which is 0 anyway)
        # and link the new ones to output_hubs
        self.flow_vars_out = {}
        for output_energy in self.output_energies.keys():
            output_hub = self.get_hub(hubs, energy=output_energy, log=log)

            # Energy can only flow from the input hub to the output hubs
            flow_vars_out = [pl.LpVariable(self.name
                                            + '_to_hub_' + output_energy
                                            + '_' + str(t),
                                            lowBound=0) for t in range(self.nb_of_timesteps)]
            
            # Add the flow linking convertor to its output hubs to the hubs equations
            output_hub.add_link(flow_vars_out, self.name + '_' + output_energy, sign='+', log=log)
            # No flow_vars_in relatef to output_hub
            self.flow_vars_out.update({output_energy: flow_vars_out})

            # Conversion ratio between input energy flow and output energy flow
            # ratio = 1./self.output_energies[output_energy]
            ratio = self.output_energies[output_energy]
            # Add output energy flow to the conversion equation
            self.equations[output_energy] = [pl.LpConstraint(e=self.flow_vars_out[output_energy][t]-ratio*self.flow_vars_in[t],
                                                             sense=pl.LpConstraintEQ,
                                                             name='equation_' + self.name + '_' + output_energy + '_' + str(t),
                                                             rhs=0) for t in range(self.nb_of_timesteps)]
            # self.equation = [self.equation[t] + ratio*flow_vars[t] for t in range(self.nb_of_timesteps)]

            if model is not None:
                for t in range(self.nb_of_timesteps):
                    model += self.equations[output_energy][t]

        if log:
            print(self.name + ' conversion equation created')

        return hubs, model

class Storage(Component):
    def __init__(self, name, description, model, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, model, nb_of_timesteps, log=log)

        # Storage-specific parameters
        self.capacity = description['capacity']
        self.initial_SOC = description['initial_SOC']

        # infinite maximum flow values would cause the solver to crash
        self.maximum_in = min(self.maximum_in, self.capacity*10000)
        self.maximum_out = min(self.maximum_out, self.capacity*10000)

        try:
            self.efficiency = description['efficiency']
        except:
            self.efficiency = 1.
        
        try:
            self.calendar_aging = description['calendar_aging']
        except:
            self.calendar_aging = 1.
        
        # Volume factor
        try:
            self.volume_factor = description['volume_factor']
        except:
            self.volume_factor = 1.
        try:
            self.volume_factor_type = description['volume_factor_type']
        except KeyError:
            self.volume_factor_type = 'Continuous'
        try:
            self.volume_factor_low_bound = description['volume_factor_low_bound']
        except KeyError:
            self.volume_factor_low_bound = 1.
        try:
            self.volume_factor_up_bound = description['volume_factor_up_bound']
        except KeyError:
            self.volume_factor_up_bound = 1.
        if self.volume_factor == 'auto':
            self.volume_factor = pl.LpVariable(self.name + '_volume_factor',
                                        lowBound=self.volume_factor_low_bound,
                                        upBound=self.volume_factor_up_bound,
                                        cat=self.volume_factor_type)
        try:
            self.volume_installation_cost = description['volume_installation_cost']
        except:
            self.volume_installation_cost = 0.
        
        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model, log=False):
        # self.model = model

        hubs, model = self.link(hubs, model)

        # Create variables to contain the state of charge of the storage
        self.SOC = [pl.LpVariable(self.name + '_' + str(t),
                                  lowBound=0) for t in range(self.nb_of_timesteps)]
        
        for t in range(self.nb_of_timesteps):
            model += self.SOC[t] <= self.capacity*self.volume_factor

        model.objective += self.volume_factor * self.volume_installation_cost

        # # Get hub associated to storage
        # hub = self.get_hub(hubs, log=log)
        
        # # Energy flow from storage to hub
        # self.flow_vars = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]

        # # TODO: faire ça mieux
        # model.objective += pl.lpSum(self.flow_vars)*0.00000001
        
        # # Add storage to its hub equation
        # hub.add_link(self.flow_vars, self.name)

        # Initial condition
        self.equation = [pl.LpConstraint(e=self.SOC[0] - self.initial_SOC * self.capacity * self.volume_factor,
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_0',
                                        rhs=0),]
        # Storage equation    
        for t in range(1, self.nb_of_timesteps):
            self.equation.append(pl.LpConstraint(e=self.SOC[t]
                                                 - self.calendar_aging*self.SOC[t-1]
                                                 - np.sqrt(self.efficiency)*self.flow_vars_in[t-1]
                                                 + np.sqrt(1/self.efficiency)*self.flow_vars_out[t-1],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_' + str(t),
                                        rhs=0))
        # Add Storage equation to model    
        for t in range(self.nb_of_timesteps):
            model += self.equation[t]

        return hubs, model

class Demand(Component):
    def __init__(self, name, description, model, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, model, nb_of_timesteps, log=log)
                
        try:
            self.dispatchable = description['dispatchable']
            self.dispatch_window = description['dispatch_window']
            self.maximum_hourly_dispatch = description['maximum_hourly_dispatch']
            if log:
                print(name + ' is dispatchable.')
        except KeyError:
            self.dispatchable = False
            self.dispatch_window = None
            self.maximum_hourly_dispatch = None
            if log:
                print(name + ' is not dispatchable.')
            
        # Bounds of Demand -> Hub
        self.maximum_out_type = 'constant'
        self.minimum_out_type = 'constant'
        self.maximum_out = 0.
        self.minimum_out = 0.

        self.value_type = description['value_type']
        self.value = utils.get_chronicle('value', description)
        if not self.dispatchable:
            # Bounds of Hub -> Demand
            self.maximum_in_type = self.value_type
            self.minimum_in_type = self.value_type
            self.maximum_in = self.value
            self.minimum_in = self.value
        elif self.dispatchable:
            # Bounds of Hub -> Demand
            self.maximum_in_type = 'constant'
            self.minimum_in_type = 'constant'
            self.maximum_in = self.maximum_hourly_dispatch/self.factor
            self.minimum_in = 0.

        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model, log=False):
        # Build flow variables and link them to Hub
        hubs, model = self.link(hubs, model, log=log)

        if self.dispatchable:
            # Check
            if np.mean(self.value*self.factor) >= self.maximum_hourly_dispatch:
                print('ERROR: MAXIMUM HOURLY DISPATCH ALLOWED FOR CONSUMPTION IS NOT SUFFICIENT, ' \
                'MUST BE AT LEAST ' + str(round(np.mean(self.value*self.factor))+1))

            y_vars = {}
            # Dispatch every hourly consumption in the allowed time window
            for t_initial in range(self.nb_of_timesteps):
                # At max, where can consumption initially at t_initial can be moved in the past
                tmin = max(0, t_initial - int(self.dispatch_window/2))
                # At max, where can consumption initially at t_initial can be moved in the future
                tmax = min(t_initial + int(self.dispatch_window/2), self.nb_of_timesteps)

                # Create y_t_initial variables (= where does consumption initially at t_initial goes?)
                y_t_initial = np.zeros(self.nb_of_timesteps, dtype=pl.LpVariable)
                y_t_initial[tmin:tmax] = [pl.LpVariable('y_' + self.name
                                                        + '_from_' + str(t_initial)
                                                        + '_to_' + str(t),
                                                        lowBound=0.) for t in range(tmin, tmax)]


                # Add constraint Somme_t(y_vars[t0, t]) = input_consumption[t0]
                model += pl.lpSum(y_t_initial) == value(self.value[t_initial])*self.factor
                
                column = 'y_' + self.name+ '_from_' + str(t_initial)
                y_vars[column] = y_t_initial
            
            self.y_vars = pd.DataFrame(y_vars)

            # Flow_vars contains dispatched consumption
            # self.flow_vars_in = [pl.LpVariable(self.name + '_' + str(t),
            #                                 lowBound=-self.maximum_hourly_dispatch) for t in range(self.nb_of_timesteps)]
            # Dispatched consumption is the sum of the dispatch of every hourly consumption
            for t in range(self.nb_of_timesteps):
                self.model += pl.lpSum(self.y_vars.loc[t]) == self.flow_vars_in[t]


        return hubs, self.model

class Source(Component):
    def __init__(self, name, description, model, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, model, nb_of_timesteps, log=log,)
        
        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model, log=False):
        
        hubs, model = self.link(hubs=hubs, model=model, log=log)

        return hubs, model
    
class EnvironmentsConnection():
    def __init__(self, environment1, environment2, descriptions, i, nb_of_timesteps):
        self.environment1 = environment1
        self.environment2 = environment2
        self.status = 'Not connected'
        self.nb_of_timesteps = nb_of_timesteps

        try:
            self.maximum_out_type = descriptions['maximum_out_type'][i]
            self.maximum_out = utils.get_chronicle('maximum_out', descriptions, i)
        except KeyError:
            self.maximum_out_type = 'constant'
            self.maximum_out = 0.
        
        try:
            self.maximum_in_type = descriptions['maximum_in_type'][i]
            self.maximum_in = utils.get_chronicle('maximum_in', descriptions, i)
        except KeyError:
            self.maximum_in_type = 'constant'
            self.maximum_in = 0.

        try:
            self.cost_out_type = descriptions['cost_out_type'][i]
            self.cost_out =utils.get_chronicle('cost_out', descriptions, i)
        except KeyError:
            self.cost_out_type = 'constant'
            self.cost_out = 0.

        try:
            self.cost_in_type = descriptions['cost_in_type'][i]
            self.cost_in = utils.get_chronicle('cost_in', descriptions, i)
        except KeyError:
            self.cost_in_type = 'constant'
            self.cost_in = 0.
        
        self.reverse = descriptions['reverse'][i]

    def connect_as_input(self, hubs, model):
            self.vars_out = {}
            self.vars_in = {}
            # If connexion time slots between environment 1 and environment 2 is an input data (then never a variable)
            for hub1, hub2, energy in zip(hubs.loc[self.environment1], hubs.loc[self.environment2], hubs.columns):
                if type(hub1) == Hub and type(hub2) == Hub:
                    # Create flow variables between hub1 and hub2
                    hub1_to_hub2 = [pl.LpVariable(energy + '_' + self.environment1 + '_hub_to_'
                                                + self.environment2 + '_hub_'
                                                + str(t),
                                                lowBound=0.) for t in range(hub1.nb_of_timesteps)]
                    hub2_to_hub1 = [pl.LpVariable(energy + '_' + self.environment2 + '_hub_to_'
                                                + self.environment1 + '_hub_'
                                                + str(t),
                                                lowBound=0.) for t in range(hub1.nb_of_timesteps)]
                    
                    # Set an upBound to flow between environment1 and environment2 (= 0. when disconnected)
                    for t in range(hub1.nb_of_timesteps):
                        model += hub1_to_hub2[t] <= bound(self.maximum_out, self.maximum_out_type, t)
                        model += hub2_to_hub1[t] <= bound(self.maximum_in, self.maximum_in_type, t)

                    # # Cut flow when environment1 and environment2 are disconnected
                    # for t in range(hub1.nb_of_timesteps):
                    #     if not conditions[t]:
                    #         hub1_to_hub2[t] = 0. # Not a pl.LpVariable anymore but a scalar value

                    # Add these new variables in the two hubs equations
                    hub1.add_link(hub1_to_hub2, component_name='to_' + hub2.name, sign='-')
                    hub1.add_link(hub2_to_hub1, component_name='from_' + hub2.name, sign='+')
                    hub2.add_link(hub1_to_hub2, component_name='to_' + hub1.name, sign='+')
                    hub2.add_link(hub2_to_hub1, component_name='from_' + hub1.name, sign='-')

                    # Update the hubs table
                    hubs.loc[self.environment1, energy] = hub1
                    hubs.loc[self.environment2, energy] = hub2
                    
                    # Save flow variables in self.vars_out and self.vars_in
                    self.vars_out.update({energy: hub1_to_hub2})
                    self.vars_in.update({energy: hub2_to_hub1})

                    model.objective += pl.lpSum([hub1_to_hub2[t]*bound(self.cost_out, self.cost_out_type, t)])
                    model.objective += pl.lpSum([hub2_to_hub1[t]*bound(self.cost_in, self.cost_in_type, t)])

                    print(hub1.name + ' and ' + hub2.name + ' connected')
            self.status = 'Connected'

            return hubs, model
    
    def trailer_config_connection(self, hubs, conditions):
        # TRAILER IS SUPPOSED TO BE ENVIRONMENT 1
        timeslots = get_connection_timeslots(conditions)
        self.switchs = [pl.LpVariable('switch_'
                                      + self.environment1 + '_'
                                      + self.environment2 + '_'
                                      + self.environment3 + '_'
                                      + str(i)) for i in range(len(timeslots))]
        
        def do_one_connection(env1, env2, sense):
            for hub1, hub2, energy in zip(hubs.loc[env1], hubs.loc[env2], hubs.columns):
                print(energy)
                if type(hub1) == Hub and type(hub2) == Hub:
                    print('hubs both exist')
                    # Create flow variables between hub1 and hub2
                    hub1_to_hub2 = [pl.LpVariable(energy + '_'
                                                  + env1 + '_hub_to_'
                                                  + env2 + '_hub_'
                                                  + str(t),
                                                  lowBound=get_bounds(t, self.switchs, timeslots, conditions, sense)[0],
                                                  upBound=get_bounds(t, self.switchs, timeslots, conditions, sense)[1])
                                                  for t in range(self.nb_of_timesteps)]
                    # Add these new variables in the two hubs equations
                    hub1.add_link(hub1_to_hub2, component_name=self.energy + '_' + env2 + '_to_' + env1, sign='-')
                    hub2.add_link(hub1_to_hub2, component_name=self.energy + '_' + env1 + '_to_' + env2)

                    # Update the hubs table
                    hubs.loc[env1, energy] = hub1
                    hubs.loc[env2, energy] = hub2

                    # Save flow variables in self.vars
                    self.vars.update({energy: hub1_to_hub2})

                    print(hub1.name + ' and ' + hub2.name + ' connected')
                else:
                    print("hubs don't both exist")

        # Environment 1 - environment 2 connection
        do_one_connection(self.environment1, self.environment2, sense=1)
        # Environment 1 - environment 3 connection
        do_one_connection(self.environment1, self.environment3, sense=-1)

        self.status = 'Connected'

        return hubs

class Model():
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
        # self.conditions = utils.get_chronicle2(config_file['conditions'], 'hourly')
        self.time = utils.get_chronicle_from_path(self.config_file['time'], 'hourly')
        self.elements_list = elements_list
        self.optimization_variable = self.config_file['optimization_variable']
        if self.config_file['optimization_sense'] == 'minimize':
            sense = pl.LpMinimize
        elif self.config_file['optimization_sense'] == 'maximize':
            sense = pl.LpMaximize
        self.model = pl.LpProblem(self.optimization_variable, sense)
        self.components = {}
        self.nb_of_timesteps = len(self.time)
        self.model.objective = pl.LpAffineExpression()

        # Create a directory for this run
        self.directory = str(self.run_num) + '_' + self.run_name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else :
            print('Warning : overwriting run ' + self.directory)

        # Save config files
        shutil.copy(config_file, self.directory + '/' + config_file)
        shutil.copy(elements_list, self.directory + '/' + elements_list)
    
    def initialize_hubs(self):
        self.hubs = pd.DataFrame(index=self.environments,
                                 columns=self.energies)
        
    def build_environment_level_variables_and_constraints(self, log=False):
        # Different types of components available
        classes = {'Source': Source,
                   'Demand': Demand,
                   'Storage': Storage,
                   'Convertor': Convertor,}
        # For each type of component:
        for class_ in classes.keys():
            # Get specs from the config file
            try : components_specs = get_components(self.elements_list, class_)
            except KeyError :
                if log:
                    print('No ' + class_ + ' found.')
                continue # If no Storage or Convertor or ... element, do nothing and continue
            self.components.update({class_: {}})
            # For each component of this type:
            for component_name in components_specs.keys():
                if components_specs[component_name]['activate']: # If component is activated
                    if log:
                        print('Found ' + component_name)
                    # Create object (Source, Demand, Storage or Convertor) from the specs
                    component = classes[class_](component_name,
                                                components_specs[component_name],
                                                self.model,
                                                self.nb_of_timesteps,
                                                log=log,)
                    # Link the component object to its hub(s)
                    self.hubs, self.model = component.build_equations(hubs=self.hubs, model=self.model, log=log)
                    # self.hubs = component.link(self.hubs, log=False)
                    # Save the component
                    self.components[class_].update({component_name: component})

    def add_hubs_equations_to_model(self, log=False):
        # To be used once all elements are added to the model
        # ie when hubs equations are completed
        # For each possible hub:
        for environment in self.environments:
            for energy in self.energies:
                hub = self.hubs.loc[environment, energy]
                # If the hub exists
                if type(hub) == Hub:
                    # Choose to display logs or not
                    if log:
                        print(hub.name)
                    # Add its equation to the model
                    for hub_equation in hub.equation:
                        self.model += hub_equation

    def connect_environments(self):
         # Get connections description
        environments_connections = self.config_file['environments_connections']
        self.environmentsConnections = {}

        if environments_connections is None:
            print('No environments connections')

        else:
            for environment1 in environments_connections.keys():
                descriptions = environments_connections[environment1]
                envs = descriptions['envs']
                
                reverse = environments_connections[environment1]['reverse']
                for i, environment2 in enumerate(envs):
                    co = EnvironmentsConnection(environment1, environment2, descriptions, i, self.nb_of_timesteps)
                    self.hubs, self.model = co.connect_as_input(self.hubs, self.model)
                    self.environmentsConnections.update({environment1 + '_' + environment2: co})
                    
    def solve(self, log=False):
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
        for class_ in ['Source', 'Demand', 'Storage', 'Convertor']:
            if log:
                print(class_)
            columns = self.components[class_].keys()
            for column in columns:
                if log:
                    print(column)
                self.dispatch.insert(loc=0, column=column, value=np.zeros(self.nb_of_timesteps))
                if not class_=='Convertor':
                    for t in tqdm(self.dispatch.index):
                        self.dispatch.loc[t, column] = (value(self.components[class_][column].flow_vars_out[t])
                                                        - value(self.components[class_][column].flow_vars_in[t]))
                elif class_=='Convertor':
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
        conversion_components = self.components['Convertor'].keys()
        for component in conversion_components:
            for output_energy in self.components['Convertor'][component].output_energies.keys():
                self.dispatch.insert(loc=0,
                                    column=component + '_' + output_energy,
                                    value=[value(self.components['Convertor'][component].flow_vars_out[output_energy][t]) for t in range(self.nb_of_timesteps)])
        
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
                if type(hub) == Hub:
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
        # self.dispatch = duplicate(self.dispatch)
        self.dispatch.time = get_timeline(self.time)
        self.dispatch = self.dispatch.astype({'time': 'datetime64[ns]'})

        # Prepare self.data for plot
        # self.data.time = get_timeline(self.time)
        # self.data = self.data.astype({'time': 'datetime64[ns]'})

    def hub_vars(self, environment, energy):
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
    
    def plot_hubs(self, save=False, log=False, co=False, env1='', env2='', price=None, unit=''):
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
                if type(hub) == Hub:
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
                    ax = plot_vars_car_connected_version(self.dispatch,
                                                         self.data,
                                                         None,
                                                         variables,
                                                         ax,
                                                         title=title,
                                                         price=price,
                                                         co=co,
                                                         env1_env2=env1_env2,
                                                         env1=env1,
                                                         env2=env2,
                                                         unit=unit,)
                    # Place a legend with the subplot
                    ax.legend(loc='best', ncols=2, frameon=False, fontsize=18).set_zorder(50)
        
        if save:
            file = str(self.run_num) + '_' + self.run_name + '/' + str(self.run_num) + '_' + self.run_name + '.png'
            fig.savefig(file)
            print('Saved at: ' + file)
    
    def plot_SOC(self, variables='all', unit='', save=False, log=False):

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
        table = pd.DataFrame(columns=components)
        for i, component in enumerate(components):
            for class_ in self.components.keys():
                components_of_type = self.components[class_]
                if component in components_of_type.keys():
                    table.loc[factors[i], component] = value(components_of_type[component].__getattribute__(factors[i]))
        if units is not None:
            # Add units to columns names
            table.columns = [components[i] + ' (' + units[i] + ')' for i in range(len(components))]

        return table



def bound(value, value_type, t):
    if value_type == 'constant' or value_type is None:
        return value
    else:
        return value[t]

def get_connection_timeslots(conditions):
    timeslots = pd.DataFrame(columns=['deconnection_time', 'connection_time'])
    old = conditions.values.astype(int)[:-1] # conditions[t]
    new = conditions.values.astype(int)[1:] # consitions[t-1]
    # Quand la dérivée est non nulle, la connexion change d'état (connectée / déconnectée)
    deco = np.where(new-old==-1)[0] # Dérivée négative -> deconnexion
    co = np.where(new-old==1)[0] # Dérivée positive -> connexion
    timeslots['deconnection_time'] = deco
    timeslots['connection_time'] = co
    return timeslots

def get_bounds(t, switchs, timeslots, conditions, sense):
    if conditions[t]: # Environments are connected
        return None, None # Then no connection limitations
    else:
        i = 0
        while timeslots.loc[i, 'deconnection_time'] > t:
            i += 1
        # sense == 1 or sense == -1
        lowBound = sense * M * switchs[i]
        upBound = sense * -M * (1-switchs[i])

        return lowBound, upBound
    
def get_components(file_of_components_list, components_class):
    with open(file_of_components_list, 'r') as file:
        components_list = yaml.safe_load(file)
    return components_list[components_class]

def value(flow_vars_t):
    # flow_t is not a var but an input data
    if type(flow_vars_t) == float or type(flow_vars_t) == np.float64 or type(flow_vars_t) == int:
        return flow_vars_t
    # flow_t is a pl.LpVariable
    else:
        return flow_vars_t.value()
    
def get_from_elements_list(keyword, elements_list):
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

def duplicate(dispatch):
    for column in dispatch.columns:
        # Duplicate and reverse every energy flow between hubs
        if 'hub' in column:
            words = column.split('_')
            words_reordered = [words[i] for i in [0, 4, 2, 3, 1, 5]]
            new_column = '_'.join(words_reordered)
            dispatch.insert(loc=1, column=new_column, value=-dispatch[column])
    return dispatch

def get_label(string):
    words = string.split('_')
    translation = ''
    for word in words:
        # translation += dictionnary[word] + ' '
        translation += word + ' '
    return translation[:-1]

def capitalise(string):
    return string[0].upper() + string[1:]

def plot_vars_car_connected_version(dispatch, data, top_var, variables, ax, title,
                                    price=None, co=False, env1_env2=None, env1='', env2='',
                                    unit=None, start=0, end=0):

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
        path, var_type, label = price
        price = utils.get_chronicle_from_path(path=path, type=var_type)
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
    new_time = np.empty(len(time), dtype=datetime)
    for i, t in enumerate(time):
        new_time[i] = datetime.strptime(t, '%Y%m%d:%H')
    return new_time