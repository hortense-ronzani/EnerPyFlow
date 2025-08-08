import pulp as pl

# Skip utils installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import yaml
from tqdm import tqdm

from datetime import datetime
#from utils import *
#import utils.get_chronicle

import utils

import importlib
importlib.reload(utils)
import pandas as pd
import numpy as np

NB_OF_TIMESTEPS = 168
M = 50000

class Component:
    def __init__(self, environment, energy, model=None):
        self.energy = energy
        self.environment = environment
        self.nb_of_timesteps = NB_OF_TIMESTEPS
        self.model = model

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
            hub = Hub(environment, energy, model=self.model)
            hubs.loc[environment, energy] = hub
            if log:
                print('hub_' + environment + '_' + energy + ' created')

        return hub
    
    # def link(self, hubs, log=False):
    #     hub = self.get_hub(hubs, log=log)
        
    #     self.flow_vars = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]
        
    #     hub.add_link(self.flow_vars)

    #     return hubs
    
class Hub(Component):
    def __init__(self, environment, energy, model):
        super().__init__(environment, energy, model)
        
        self.name = 'hub_' + energy + '_' + environment

        self.unused_energy = [pl.LpVariable('unused_energy_' + self.name + '_' + str(t), lowBound=0) for t in range(self.nb_of_timesteps)]
        self.equation = [pl.LpConstraint(e=-self.unused_energy[t],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_' + str(t),
                                        rhs=0) for t in range(self.nb_of_timesteps)]
        
    def add_link(self, variables, sign='+'):
        if sign == '+':
            self.equation = [self.equation[t] + variables[t] for t in range(self.nb_of_timesteps)]
        elif sign == '-':
            self.equation = [self.equation[t] - variables[t] for t in range(self.nb_of_timesteps)]
   
class Conversion(Component):
    def __init__(self, name, description, model, log=False):
        super().__init__(description['environment'], description['input_energy'], model)

        self.name = name

        self.input_energy = self.energy
        self.output_energies = description['output_energies']
        
        if log:
            print(name + ' created.')

    def link(self, hubs, model, log=False):
        self.model = model

        input_hub = self.get_hub(hubs, log=log)
        
        # Energy can only flow from the input hub to the output hubs
        self.flow_vars = [pl.LpVariable(self.name
                                        + '_to_hub_' + self.input_energy
                                        + '_' + str(t),
                                        upBound=0) for t in range(self.nb_of_timesteps)]
        
        # Add the flow linking convertor to its input hub to the hub equation
        input_hub.add_link(self.flow_vars)

        # Add input energy flow to the conversion equation
        self.equation = [pl.LpConstraint(e=self.flow_vars[t],
                                         sense=pl.LpConstraintEQ,
                                         name='equation_' + self.name + '_' + str(t),
                                         rhs=0) for t in range(self.nb_of_timesteps)]

        self.flow_vars_out = {}
        for output_energy in self.output_energies.keys():
            output_hub = self.get_hub(hubs, energy=output_energy, log=log)

            # Energy can only flow from the input hub to the output hubs
            flow_vars = [pl.LpVariable(self.name
                                            + '_to_hub_' + output_energy
                                            + '_' + str(t),
                                            lowBound=0) for t in range(self.nb_of_timesteps)]
            # Conversion ratio between input energy flow and output energy flow
            ratio = 1./self.output_energies[output_energy]
            # Add output energy flow to the conversion equation
            self.equation = [self.equation[t] + ratio*flow_vars[t] for t in range(self.nb_of_timesteps)]

            # Add the flow linking convertor to its output hubs to the hubs equations
            output_hub.add_link(flow_vars)
            self.flow_vars_out.update({output_energy: flow_vars})

        if log:
            print(self.name + ' conversion equation created')

        if self.model is not None:
            for t in range(self.nb_of_timesteps):
                self.model += self.equation[t]

        return hubs, self.model

class Storage(Component):
    def __init__(self, name, description, model, log=False):
        super().__init__(description['environment'], description['energy'], model)

        self.name = name

        self.capacity = description['capacity']
        self.efficiency = description['efficiency']
        self.initial_SOC = description['initial_SOC']
        self.calendar_aging = description['calendar_aging']
        
        try:
            self.factor = description['factor']
        except:
            self.factor = 1.

        try:
            self.unit_cost = description['unit_cost']
        except:
            self.unit_cost = 0.
        
        if log:
            print(name + ' created.')

    def link(self, hubs, model, log=False):
        self.model = model

        # Create variables to contain the state of charge of the storage
        self.SOC = [pl.LpVariable(self.name + '_' + str(t),
                                  lowBound=0) for t in range(self.nb_of_timesteps)]
        
        if self.factor == 'auto':
            self.factor = pl.LpVariable(self.name + '_factor',
                                        lowBound=0.,
                                        upBound=5.)

        for t in range(self.nb_of_timesteps):
            self.model += self.SOC[t] <= self.capacity*self.factor

        self.model.objective += self.factor * self.unit_cost

        # Get hub associated to storage
        hub = self.get_hub(hubs, log=log)
        
        # Energy flow from storage to hub
        self.flow_vars = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]
        
        # Add storage to its hub equation
        hub.add_link(self.flow_vars)

        # Initial condition
        self.equation = [pl.LpConstraint(e=self.SOC[0],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_0',
                                        rhs=self.initial_SOC * self.capacity),]
        # Storage equation    
        for t in range(1, self.nb_of_timesteps):
            self.equation.append(pl.LpConstraint(e=self.SOC[t]
                                                 - self.calendar_aging*self.SOC[t-1]
                                                 + self.flow_vars[t-1],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_' + str(t),
                                        rhs=0))
            
        if self.model is not None:
            for t in range(self.nb_of_timesteps):
                self.model += self.equation[t]

        return hubs, self.model

class Consumption(Component):
    def __init__(self, name, description, model, log=False):
        super().__init__(description['environment'], description['energy'], model)

        self.name = name

        try:
            self.factor = description['factor']
        except KeyError:
            self.factor = 1.
        
        self.value_type = description['value_type']
        self.value = utils.get_chronicle('value', description)*self.factor
        
        try:
            self.dispatchable = description['dispatchable']
            self.dispatch_window = description['dispatch_window']
            self.maximum_hourly_dispatch = description['maximum_hourly_dispatch']
        except KeyError:
            self.dispatchable = False
            self.dispatch_window = None
            self.maximum_hourly_dispatch = None

        if log:
            print(name + ' created.')

    def link(self, hubs, model, log=False):
        self.model = model

        hub = self.get_hub(hubs, log=log)
        
        if not self.dispatchable:
            self.flow_vars = [bound(-self.value, self.value_type, t) for t in range(self.nb_of_timesteps)]                           
            hub.add_link(self.flow_vars)

        if self.dispatchable:
            # Check
            if np.mean(self.value) >= self.maximum_hourly_dispatch:
                print('ERROR: MAXIMUM HOURLY DISPATCH ALLOWED FOR CONSUMPTION IS NOT SUFFICIENT, ' \
                'MUST BE AT LEAST ' + str(round(np.mean(self.value))+1))

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
                                                        upBound=0.) for t in range(tmin, tmax)]


                # Add constraint Somme_t(y_vars[t0, t]) = input_consumption[t0]
                self.model += pl.lpSum(y_t_initial) == -value(self.value[t_initial])
                
                column = 'y_' + self.name+ '_from_' + str(t_initial)
                y_vars[column] = y_t_initial
            
            self.y_vars = pd.DataFrame(y_vars)

            # Flow_vars contains dispatched consumption
            self.flow_vars = [pl.LpVariable(self.name + '_' + str(t),
                                            lowBound=-self.maximum_hourly_dispatch) for t in range(self.nb_of_timesteps)]
            # Dispatched consumption is the sum of the dispatch of every hourly consumption
            for t in range(self.nb_of_timesteps):
                self.model += pl.lpSum(self.y_vars.loc[t]) == self.flow_vars[t]

            hub.add_link(self.flow_vars)

        return hubs, self.model

class Production(Component):
    def __init__(self, name, description, model, log=False):
        super().__init__(description['environment'], description['energy'], model)

        self.name = name

        try:
            self.factor = description['factor']
        except KeyError:
            self.factor = 1.

        try:
            self.unit_cost = description['unit_cost']
        except KeyError:
            self.unit_cost = 0.

        try:
            self.minimum_type = description['minimum_type']
            self.minimum = utils.get_chronicle('minimum', description)
        except KeyError:
            self.minimum_type = None
            self.minimum = None

        try:
            self.maximum_type = description['maximum_type']
            self.maximum = utils.get_chronicle('maximum', description)
        except KeyError:
            self.maximum_type = None
            self.maximum = None

        try: 
            self.cost_type = description['cost_type']
            self.cost = utils.get_chronicle('cost', description)
        except KeyError:
            self.cost_type = 'constant'
            self.cost = 0.
        
        if log:
            print(name + ' created.')

    def link(self, hubs, model, log=False):
        self.model = model
        self.flow_vars = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]

        if self.factor == 'auto':
            # TODO: change bounds
            self.factor = pl.LpVariable(self.name + '_factor',
                                        lowBound=0,
                                        upBound=10)

        self.model.objective += self.factor*self.unit_cost

        for t in range(self.nb_of_timesteps):
            if self.maximum_type is not None:
                self.model += self.flow_vars[t] <= self.factor*bound(self.maximum, self.maximum_type, t)
            if self.minimum_type is not None:
                self.model += self.flow_vars[t] >= self.factor*bound(self.minimum, self.minimum_type, t)

        hub = self.get_hub(hubs, log=log)
        
        hub.add_link(self.flow_vars)

        if self.model.name == 'cost':
            self.model.objective += pl.lpSum([self.flow_vars[t]*bound(self.cost, self.cost_type, t) for t in range(self.nb_of_timesteps)])

        return hubs, self.model
    
class EnvironmentsConnection():
    def __init__(self, environment1, environment2, model=None):
        self.environment1 = environment1
        self.environment2 = environment2
        self.status = 'Not connected'
        self.model = model

    def connect_as_input(self, hubs, conditions):
            self.nb_of_timesteps = len(conditions)
            self.vars = {}
            # If connexion time slots between environment 1 and environment 2 is an input data (then never a variable)
            for hub1, hub2, energy in zip(hubs.loc[self.environment1], hubs.loc[self.environment2], hubs.columns):
                if type(hub1) == Hub and type(hub2) == Hub:
                    # Create flow variables between hub1 and hub2
                    hub1_to_hub2 = [pl.LpVariable(energy + '_' + self.environment1 + '_hub_to_'
                                                + self.environment2 + '_hub_'
                                                + str(t)) for t in range(hub1.nb_of_timesteps)]

                    # Cut flow when environment1 and environment2 are disconnected
                    for t in range(hub1.nb_of_timesteps):
                        if not conditions[t]:
                            hub1_to_hub2[t] = 0. # Not a pl.LpVariable anymore but a scalar value

                    # Add these new variables in the two hubs equations
                    hub1.add_link(hub1_to_hub2, sign='-')
                    hub2.add_link(hub1_to_hub2)

                    # Update the hubs table
                    hubs.loc[self.environment1, energy] = hub1
                    hubs.loc[self.environment2, energy] = hub2
                    
                    # Save flow variables in self.vars
                    self.vars.update({energy: hub1_to_hub2})

                    print(hub1.name + ' and ' + hub2.name + ' connected')
            self.status = 'Connected'

            return hubs
    
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
                    hub1.add_link(hub1_to_hub2, sign='-')
                    hub2.add_link(hub1_to_hub2)

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
    def __init__(self, config_file='general_config_file.yaml', elements_list='elements_list.yaml'):
        with open(config_file, 'r') as file:
            config_file = yaml.safe_load(file)
            self.config_file = config_file
        self.run_num = config_file['run_num']
        self.run_name = config_file['run_name']
        self.energies = get_from_elements_list('energy', elements_list)
        self.environments = get_from_elements_list('environment', elements_list)
        self.data = pd.read_csv('data_sample.csv', sep=';')
        # self.conditions = utils.get_chronicle2(config_file['conditions'], 'hourly')
        self.time = utils.get_chronicle2(config_file['time'], 'hourly')
        self.elements_list = elements_list
        self.optimization_variable = config_file['optimization_variable']
        if config_file['optimization_sense'] == 'minimize':
            sense = pl.LpMinimize
        elif config_file['optimization_sense'] == 'maximize':
            sense = pl.LpMaximize
        self.model = pl.LpProblem(self.optimization_variable, sense)
        self.components = {}
        self.nb_of_timesteps = len(self.time)
        self.model.objective = pl.LpAffineExpression()
    
    def initialize_hubs(self):
        self.hubs = pd.DataFrame(index=self.environments,
                                 columns=self.energies)
        
    def build_environment_level_variables_and_constraints(self, log=False):
        # Different types of components available
        classes = {'Production': Production,
                   'Consumption': Consumption,
                   'Storage': Storage,
                   'Conversion': Conversion,}
        # For each type of component:
        for class_ in classes.keys():
            # Get specs from the config file
            try : components_specs = get_components(self.elements_list, class_)
            except KeyError : continue # If no Storage or Conversion or ... element, do nothing and continue
            self.components.update({class_: {}})
            # For each component of this type:
            for component_name in components_specs.keys():
                if components_specs[component_name]['activate']: # If component is activated
                    # Create object (Production, Consumption, Storage or Conversion) from the specs
                    component = classes[class_](component_name,
                                                components_specs[component_name],
                                                self.model,)
                    # Link the component object to its hub(s)
                    self.hubs, self.model = component.link(self.hubs, self.model)
                    # self.hubs = component.link(self.hubs, log=False)
                    # Save the component
                    self.components[class_].update({component_name: component})

    def add_hubs_equations_to_model(self, logs=False):
        # To be used once all elements are added to the model
        # ie when hubs equations are completed
        # For each possible hub:
        for environment in self.environments:
            for energy in self.energies:
                hub = self.hubs.loc[environment, energy]
                # If the hub exists
                if type(hub) == Hub:
                    # Choose to display logs or not
                    if logs:
                        print(hub.name)
                    # Add its equation to the model
                    for hub_equation in hub.equation:
                        self.model += hub_equation

    def connect_environments(self):
         # Connections between different environments

         # Get connections description
        environments_connections = self.config_file['environments_connections']

        self.vars = []
        for environment1 in environments_connections.keys():
            envs = environments_connections[environment1]['envs']
            conditions_type = environments_connections[environment1]['conditions_type']
            conditions = environments_connections[environment1]['conditions']
            reverse = environments_connections[environment1]['reverse']
            for i, environment2 in enumerate(envs):
                if conditions_type[i]=='input':
                    co = EnvironmentsConnection(environment1, environment2)
                    conditions[i] = utils.get_chronicle2(conditions[i], 'hourly')
                    if reverse[i]:
                        conditions[i] = ~conditions[i]
                    self.hubs = co.connect_as_input(self.hubs, conditions[i])
                    self.vars.append(co)
                else:
                    print('ERROR: conditions_type=' + conditions_type[i] +
                          ' BUT NOTHING ELSE THAN conditions_type="input" IS VALID')
        
        # # Connect house and car with input conditions
        # co_house_car = EnvironmentsConnection('house', 'car')
        # self.hubs = co_house_car.connect(self.hubs, 'input', self.conditions)
        # self.vars.append(co_house_car)

        # # Connect trailer and house and car with custom conditions
        # co_trailer_house = EnvironmentsConnection('trailer', 'house', 'car')
        # self.hubs = co_trailer_house.connect(self.hubs, 'input', self.conditions)
        # self.vars.append(co_trailer_house)

        # # Connect trailer and house and car with custom conditions
        # co_trailer_car = EnvironmentsConnection('trailer', 'car', 'house')
        # self.hubs = co_trailer_car.connect(self.hubs, 'input', self.conditions==False)
        # self.vars.append(co_trailer_car)

    def solve(self):
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
        for class_ in ['Production', 'Consumption', 'Storage', 'Conversion']:
            columns = self.components[class_].keys()
            for column in columns:
                self.dispatch.insert(loc=0, column=column, value=np.zeros(self.nb_of_timesteps))
                for t in tqdm(self.dispatch.index):
                    self.dispatch.loc[t, column] = value(self.components[class_][column].flow_vars[t])

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
        conversion_components = self.components['Conversion'].keys()
        for component in conversion_components:
            for output_energy in self.components['Conversion'][component].output_energies.keys():
                self.dispatch.insert(loc=0,
                                    column=component + '_' + output_energy,
                                    value=[value(self.components['Conversion'][component].flow_vars_out[output_energy][t]) for t in range(self.nb_of_timesteps)])
        
        # Get exchanges between environments
        for environments_connection in self.vars:
            # Sorry pour le code smell
            environment1 = environments_connection.environment1
            environment2 = environments_connection.environment2

            for energy in environments_connection.vars.keys():
                self.dispatch.insert(loc=0,
                                     column=energy + '_'
                                     + environment1 + '_hub_to_'
                                     + environment2 + '_hub',
                                     value=[value(environments_connection.vars[energy][t]) for t in range(self.nb_of_timesteps)])

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
        self.dispatch.to_csv(solution_file)

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