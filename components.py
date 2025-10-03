import pulp as pl

# Skip modules installation
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import yaml

import utils
from utils import *

import importlib
importlib.reload(utils)
import pandas as pd
import numpy as np

# default values part 1 = that can be given directly
default_values_Component_phase1 = {'minimum': 0.,
                                    'maximum': 1000000.,
                                    'cost': 0.,
                                    'factor': 1.,
                                    'factor_type': 'Continuous',
                                    'factor_low_bound': 1.,
                                    'factor_up_bound': 1.,
                                    'installation_cost': 0.,
}

# default values that depend on the actual values given in phase 1
def get_default_values_Component_phase2(o):
    """
    Gets default values of the attributes of a Component that depends on other attributes values.

    Args:
        o (Component): Component object to read the already created attributes from.
    
    Returns:
        dict(): Dictionnary with new attributes names as keys containing their default values.
    """
    default_values_Component_phase2 = {'minimum_in': getattr(o, 'minimum'),
                                    'minimum_out': getattr(o, 'minimum'),
                                    'maximum_in': getattr(o, 'maximum'),
                                    'maximum_out': getattr(o, 'maximum'),
                                    'cost_in': -getattr(o, 'cost'),
                                    'cost_out': getattr(o, 'cost'),
                                    }
    return default_values_Component_phase2

class Component:
    """
    This class represents the base components and some fictional objects and is used to build the links (energy flows).

    Attributes:
        name (str): Name of the component.
        energy (str): Main energy type of the component.
        environment (str): Main environment of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints).
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object. Default is 0. ; it highly
            recommanded to NOT PUT NEGATIVE VALUES.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints).
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints).
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects ; if the best size of
            component has to be automatically optimized, set factor='auto'. WARNING: 'auto' option is not compatible
            with dispatchable Demand objects.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1. 
        flow_vars_out (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-out variables (from Component to Hub)
            of the Component at each timestep. Once the problem solved, flow_vars_out[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.

    Args:
        name (str): Name of the component.
        energy (str): Main energy type of the component.
        environment (str): Main environment of the component.
        description (dict): Sub-section of the configuration file describing the characteristics of the links (attributes of the Component).
        nb_of_timesteps (int): Number of timesteps of the simulation.
        isHub (bool): Must be True if the component described is a Hub (Component sub-class), else False.
        i (int | None): To be used only if the component described is an EnvironmentsConnection (Component sub-class) to iterate within a list of specifications.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, name, energy, environment, description, nb_of_timesteps=None, isHub=False, i=None, log=False):
        self.energy = energy
        self.environment = environment
        # self.model = model
        self.nb_of_timesteps = nb_of_timesteps
        self.name = name

        # For every Component except Hubs
        if not isHub:
            # Get attributes that are common to all components phase 1
            for attribute in default_values_Component_phase1.keys():
                try:
                    # Read value from the configuration file
                    self.__setattr__(attribute, get_chronicle(description[attribute], log=log, i=i))
                except KeyError:
                    # If no value given in the configuration, set default value
                    self.__setattr__(attribute, default_values_Component_phase1[attribute])
            

            # Get attributes that are common to all components phase 2
            default_values_Component_phase2 = get_default_values_Component_phase2(self)
            for attribute in default_values_Component_phase2.keys():
                try:
                    # Read value from the configuration file
                    self.__setattr__(attribute, get_chronicle(description[attribute], log=log, i=i))
                except KeyError: # if attribute not found in the configuration file
                    # If no value given in the configuration, set default value
                    self.__setattr__(attribute, default_values_Component_phase2[attribute])

            if self.factor == 'auto':
                self.factor = pl.LpVariable(self.name + '_factor',
                                            lowBound=self.factor_low_bound,
                                            upBound=self.factor_up_bound,
                                            cat=self.factor_type)
        
    def get_hub(self, hubs, environment=None, energy=None, log=False):
        """
        This method gets the Hub(environment, energy) to be linked to the Component object,
        and creates it if it doesn't exist.

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            environment (str): Environment of the Hub to be selected. Default is self.environment, the main
                environment attached to the Component object.
            energy (str): Energy of the Hub to be selected. Default is self.energy, the main energy type
                attached to the Component object.
            log (bool): If True, additional information will be printed throughout the process.
        
        Returns:
            Hub(Component): Hub object corresponding to (environment, energy).
        """
        if environment is None:
            environment = self.environment
        if energy is None:
            energy = self.energy
        hub = hubs.loc[environment, energy]

        if type(hub) == Hub:
            if log:
                print('hub_' + environment + '_' + energy + ' already exists')
        else:
            hub = Hub(environment, energy, nb_of_timesteps=self.nb_of_timesteps)
            hubs.loc[environment, energy] = hub
            if log:
                print('hub_' + environment + '_' + energy + ' created')

        return hub
    
    def link(self, hubs, model, log=False):
        """
        This methods creates flow_vars_in and flow_vars_out lists of flow variables that will be optimized, adds them
        to them to the Hub equation (energy conservation equation), and add the constraints factor*minimum_in[t] <=
        flow_vars_in[t] <= factor*maximum_in[t] and factor*minimum_out[t] <= flow_vars_out[t] <= factor*maximum_out[t]
        for every timestep t between 0 and nb_of_timesteps-1. Updates the model objective function with costs attributed
        to the flow variables: model.objective += sum_t flow_vars_in[t]*cost_in[t] + flow_vars_out[t]*cost_out[t].

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            model (pulp.LpProblem): Pulp linear problem to be optimized.
            log (bool): If True, additional information will be printed throughout the process.

        Returns:
            pandas.DataFrame: Updated table of all hubs.
            pulp.LpProblem: Updated pulp linear problem to be optimized.
        """
        self.flow_vars_in = [pl.LpVariable('hub_to_' + self.name + '_' + str(t)) for t in range(self.nb_of_timesteps)]
        self.flow_vars_out = [pl.LpVariable(self.name + '_to_hub_' + str(t)) for t in range(self.nb_of_timesteps)]

        # Add bounds to flow variables
        for t in range(self.nb_of_timesteps):
            model += self.flow_vars_out[t] <= self.factor*bound(self.maximum_out, t)
            model += self.flow_vars_out[t] >= self.factor*bound(self.minimum_out, t)
            model += self.flow_vars_in[t] <= self.factor*bound(self.maximum_in, t)
            model += self.flow_vars_in[t] >= self.factor*bound(self.minimum_in, t)

        # Add installation cost to the model objective function
        model.objective += self.factor*self.installation_cost

        # Add flow cost to the model objective function
        model.objective += pl.lpSum([self.flow_vars_out[t]*bound(self.cost_out, t)
                                    + self.flow_vars_in[t]*bound(self.cost_in, t) for t in range(self.nb_of_timesteps)])

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
    """
    Sub-class of Component class representing the link between other Components belonging to the same environment with the same
    energy type. Examples: Hub(house, electricity), Hub(house, heat) or Hub(car, electricity). Is used to implement the energy
    conservation equation within each environment and for each energy type.

    Inherited attributes:
        name (str): Name of the hub.
        environment (str): Main environment of the hub.
        energy (str): Main energy type of the hub.
        nb_of_timesteps (int): Number of timesteps of the simulation.

    Attributes:
        unused_energy (list[pulp.LpVariable]): List of variables representing the energy losses of the hub at each timestep.
        equation (list[pulpLpConstraint]): List of equations representing the conservation of energy (everything entering
            the hub = everything going out of the hub) at each timestep.
        component_names (list[str]): List of names of the components linked to the hub.

    Args:
        environment (str): Main environment of the hub.
        energy (str): Main energy type of the hub.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, environment, energy, nb_of_timesteps, log=False):
        name = 'hub_' + energy + '_' + environment
        # Instanciate the hub as a Component
        super().__init__(name, energy, environment, description=None, nb_of_timesteps=nb_of_timesteps, isHub=True, log=log,)

        self.unused_energy = [pl.LpVariable('unused_energy_' + self.name + '_' + str(t), lowBound=0) for t in range(self.nb_of_timesteps)]
        # Initialize the hub equation with right hand side = 0 and left hand side = - losses[t] for each timestep t
        self.equation = [pl.LpConstraint(e=-self.unused_energy[t],
                                        sense=pl.LpConstraintEQ,
                                        name='equation_' + self.name + '_' + str(t),
                                        rhs=0) for t in range(self.nb_of_timesteps)]
        self.component_names = []
        
    def add_link(self, variables, component_name, sign='+', log=False):
        """
        Link a new component to the hub by adding its flow variables to the hub equation.

        Args:
            variables (list[pulp.LpVariable]): Liste of length nb_of_timesteps with flow variables to be added to the hub
                equation.
            component_name (str): Name of the component to be linked.
            sign (str): Can be '+' or '-'. Describes whether it is flow_in or flow_out variables. If sign='+' then +variables[t]
                is added to the left hand side of the hub equation, else if sign='-' then -variables[t] is added to the left hand
                side of the hub equation.
            log (bool): If True, additional information will be printed throughout the initialization.

        """
        if sign == '+':
            self.equation = [self.equation[t] + variables[t] for t in range(self.nb_of_timesteps)]
        elif sign == '-':
            self.equation = [self.equation[t] - variables[t] for t in range(self.nb_of_timesteps)]
        # Update the list of names of components linked to this hub
        self.component_names.append(component_name)
        if log:
            print(self.name + ' updated with ' + component_name)
   
class Converter(Component):
    """
    Sub-class of Component class representing a converter object. Allows to link different hubs within a same environment
    with different energy types, with efficiency factors for the conversion. Only one sense is allowed for energy conversion,
    from only one input hub to one or several output hubs.

    Inherited Attributes:
        name (str): Name of the component.
        energy (str): Main energy type of the component.    
        environment (str): Main environment of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints). ALWAYS=0. FOR A CONVERTER OBJECT.
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object. Default is 0. ; it highly
            recommanded to NOT PUT NEGATIVE VALUES.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints).
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints).
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects ; if the best size of
            component has to be automatically optimized, set factor='auto'. WARNING: 'auto' option is not compatible
            with dispatchable Demand objects.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.

    Attributes:
        input_energy (str): Input energy type.
        output_energies (list[str]): List of output energies.
        flow_vars_out (dict[str:list[pulp.LpVariable]]): Dictionnary with output energy types as keys and lists of length nb_of_timesteps
            containing the flow-out variables (from Component to Hub) of the Converter at each timestep, for each type of output energy.
            Once the problem solved, flow_vars_out[t].value() returns the optimized value of the variable for each timestep t between 0
            and nb_of_timesteps-1.
        equations (dict[str:list[pulp.LpConstraint]]): Dictionnary with output energy types as keys and lists of length nb_of_timesteps
            containing the conversion equations from the input energy type to each output energy type, at each timestep:
            flow_vars_out[output_energy][t] = conversion_ratio*flow_vars_in[t] for each timestep t between 0 and nb_of_timesteps-1.

    Args:
        name (str): Name of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, name, description, nb_of_timesteps, log=False):
        super().__init__(name, description['input_energy'], description['environment'], description, nb_of_timesteps, log=log,)

        self.input_energy = self.energy
        self.output_energies = description['output_energies']

        # No energy flow from Converter to input Hub
        self.maximum_out = 0.
        
        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model=None, log=False):
        """
        Method to build the conversion equations between output energy types and input energy type. Calls method Component.link()
        to build variables and constraints related to the input energy hub, creates flow_vars_out variables for the links with the
        output energy hubs, adds them to the output energy hubs equations, builds conversions equations between input energy flows
        and output energy flows and adds them to the linear problem as constraints.

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            model (pulp.LpProblem): Pulp linear problem to be optimized.
            log (bool): If True, additional information will be printed throughout the process.
        
        Returns:
            pandas.DataFrame: Updated table of all hubs.
            pulp.LpProblem: Updated pulp linear problem to be optimized.
        """
        # Create link from input energy hub
        hubs, model = self.link(hubs, model, log=log)

        # Initialize a dictionnary to store Converter equations
        self.equations = {}

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
            
            # Add the flow linking converter to its output hubs to the hubs equations
            output_hub.add_link(flow_vars_out, self.name + '_' + output_energy, sign='+', log=log)
            # No flow_vars_in relatef to output_hub
            self.flow_vars_out.update({output_energy: flow_vars_out})

            # Conversion ratio between input energy flow and output energy flow
            ratio = self.output_energies[output_energy]
            # Add output energy flow to the conversion equation
            self.equations[output_energy] = [pl.LpConstraint(e=self.flow_vars_out[output_energy][t]-ratio*self.flow_vars_in[t],
                                                             sense=pl.LpConstraintEQ,
                                                             name='equation_' + self.name + '_' + output_energy + '_' + str(t),
                                                             rhs=0) for t in range(self.nb_of_timesteps)]

            if model is not None:
                for t in range(self.nb_of_timesteps):
                    model += self.equations[output_energy][t]

        if log:
            print(self.name + ' conversion equation created')

        return hubs, model

class Storage(Component):
    """
    Sub-class of Component class representing a storage device.

    Inherited Attributes:
        name (str): Name of the component.
        energy (str): Main energy type of the component.    
        environment (str): Main environment of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints).
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object. Default is 0. ; it highly
            recommanded to NOT PUT NEGATIVE VALUES.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints).
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints).
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects ; if the best size of
            component has to be automatically optimized, set factor='auto'. WARNING: 'auto' option is not compatible
            with dispatchable Demand objects.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
        flow_vars_out (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-out variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_out[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.

    Attributes:
        SOC (list[pulp.LpVariable]): List of length nb_of_timesteps of variables describing the energy contained in the storage at every
            timestep.
        capacity (float): Storage capacity, in the same unit as its energy type.
        initial_SOC (float): Initial storage state-of-charge rate (at t=0), must be comprised between 0. and 1.
        final_SOC (float): Final storage state-of-charge rate (at t=nb_of_timesteps - 1), must be comprised between 0. and 1.
        efficiency (float): Storage efficiency = energy out / energy in. Must be comprised between 0. and 1.
        calendar_loss (float): Loss of energy stored at each timestep = energy stored at t / energy stored at t-1 without external flows.
            Must be comprised between 0. and 1.
        volume_factor (float | int | str): Actual storage capacity is multiplied by volume_factor: 0 <= SOC[t] <= volume_factor * capacity for
            each timestep t. If volume_factor is set to 'auto', best volume_factor will be automatically determined.
        volume_factor_type (str): Used if volume_factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
        volume_factor_low_bound (float): Used if volume_factor is set to 'auto' and set the factor type. volume_factor_low_bound <= volume_factor.
        volume_factor_up_bound (float): Used if volume_factor is set to 'auto' and set the factor type. volume_factor_up_bound => volume_factor.
        volume_installation_cost (float): Cost for one unit installed (volume_factor=1.) Total installation due to storage volume will be
            volume_installation_cost * volume_factor.
        equation (list[pulp.LpConstraints]): List of length nb_of_timesteps containing the storage energy conservation equations for every
            timestep t>0 : SOC[t] = calendar_loss*SOC[t-1] + (efficiency)^(1/2)*flow_vars_in[t-1] - efficiency^(-1/2)*flow_vars_out[t-1].
    
    Args:
        name (str): Name of the component.
        description (dict): Portion of the configuration file describing the characteristics of the component links.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, name, description, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, nb_of_timesteps, log=log)

        # Storage-specific parameters
        self.capacity = description['capacity']
        self.initial_SOC = description['initial_SOC']
        self.final_SOC = description['final_SOC']

        # Get Storage-specific parameters
        try:
            self.efficiency = description['efficiency']
        except:
            self.efficiency = 1.
        
        try:
            self.calendar_loss = description['calendar_aging']
        except:
            self.calendar_loss = 1.
        
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
        """
        Method to build storage-specific variables and constraints. Calls method Component.link() to build variables and
        constraints related to the energy flows, creates SOC variables for the energy stored at each timestep, builds 
        equations to modelize the storage capacity and energy flows and adds them to the linear problem as constraints.

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            model (pulp.LpProblem): Pulp linear problem to be optimized.
            log (bool): If True, additional information will be printed throughout the process.
        
        Returns:
            pandas.DataFrame: Updated table of all hubs.
            pulp.LpProblem: Updated pulp linear problem to be optimized.
        """
        

        # Link to Hub
        hubs, model = self.link(hubs, model)

        # Create variables to contain the state of charge of the storage
        self.SOC = [pl.LpVariable(self.name + '_' + str(t),
                                  lowBound=0) for t in range(self.nb_of_timesteps)]
        
        for t in range(self.nb_of_timesteps):
            model += self.SOC[t] <= self.capacity*self.volume_factor

        model.objective += self.volume_factor * self.volume_installation_cost

        # Initial condition
        self.equation = [pl.LpConstraint(e=self.SOC[0] - self.initial_SOC * self.capacity * self.volume_factor,
                                        sense=pl.LpConstraintEQ,
                                        name='initial_condition_' + self.name + '_0',
                                        rhs=0),]
        
        # Final condition
        self.equation.append(pl.LpConstraint(e=self.SOC[self.nb_of_timesteps-1] - self.final_SOC * self.capacity * self.volume_factor,
                                        sense=pl.LpConstraintEQ,
                                        name='final_condition_' + self.name + '_' + str(self.nb_of_timesteps-1),
                                        rhs=0))

        # Storage equation    
        for t in range(1, self.nb_of_timesteps):
            self.equation.append(pl.LpConstraint(e=self.SOC[t]
                                                 - self.calendar_loss*self.SOC[t-1]
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
    """
    Sub-class of Component class representing an energy demand. Can be dispatchable within some limitations or not
    dispatchable (default).

    Inherited Attributes:
        name (str): Name of the component.
        energy (str): Main energy type of the component.    
        environment (str): Main environment of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints). ALWAYS=0. FOR A DEMAND OBJECT.
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object. ALWAYS=0. FOR A DEMAND OBJECT.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects. 'AUTO' OPTION IS NOT COMPATIBLE WITH
            DISPATCHABLE DEMAND OBJECTS.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
        flow_vars_out (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-out variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_out[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
    
    Attributes:
        dispatchable (bool): True if demand is dispatchable, else False.
        dispatch_window (int): Used only if demand is dispatchable. Indicates the extend to when each chunk of initial demand can be
            displaced. Example: if dispatch_window=24, each chunk of initial demand can be dispatched within the time window [-12, +12].
        value (List[float]): List of length nb_of_timesteps with initial demand values, in energy type unit.
        y_vars (pandas.DataFrame): Only if demand is dispatchable. Table of variables indicating when each chunk of initial demand is
            displaced.
    
    Args:
        name (str): Name of the component.
        description (dict): Portion of the configuration file describing the characteristics of the component links.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, name, description, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, nb_of_timesteps, log=log,)
                
        try:
            self.dispatchable = description['dispatchable']
            self.dispatch_window = description['dispatch_window']
            if log:
                print(name + ' is dispatchable.')
        except KeyError:
            self.dispatchable = False
            self.dispatch_window = None
            if log:
                print(name + ' is not dispatchable.')
            
        # Bounds of Demand -> Hub
        self.maximum_out = 0.
        self.minimum_out = 0.

        self.value = get_chronicle(description['value'], log=log)
        if not self.dispatchable:
            # Bounds of Hub -> Demand
            self.maximum_in = self.value
            self.minimum_in = self.value

        if type(self.factor) is pl.LpVariable:
            if self.dispatchable:
                print('ERROR: factor = "auto" should not be used for dispatchable Demand object ' + self.name + '.')
                raise Exception('ERROR: factor = "auto" should not be used for dispatchable Demand object ' + self.name + '.')
            

        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model, log=False):
        """
        Method to build demand-specific variables and constraints. Calls method Component.link() to build
        variables and constraints related to the energy flows, and if demand is dispatchable, creates dispatching
        variables and equations variables and adds them to the linear problem as constraints.

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            model (pulp.LpProblem): Pulp linear problem to be optimized.
            log (bool): If True, additional information will be printed throughout the process.
        
        Returns:
            pandas.DataFrame: Updated table of all hubs.
            pulp.LpProblem: Updated pulp linear problem to be optimized.
        """

        # Build flow variables and link them to Hub
        hubs, model = self.link(hubs, model, log=log)

        if self.dispatchable:
            # Check
            # if np.mean(self.value*self.factor) >= self.maximum_hourly_dispatch:
            if  np.mean(self.value) >= self.maximum_in:
                print('ERROR: maximum hourly dispatch allowed for consumption is not sufficient, ' \
                'must be at least ' + str(round(np.mean(self.value))+1))
                raise Exception('ERROR: maximum hourly dispatch allowed for consumption is not sufficient, ' \
                'must be at least ' + str(round(np.mean(self.value))+1))

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
                                                        lowBound=bound(self.minimum, t)) for t in range(tmin, tmax)]

                # Add constraint Somme_t(y_vars[t0, t]) = input_consumption[t0]
                model += pl.lpSum(y_t_initial) == value(bound(self.value, t_initial))*self.factor
                
                column = 'y_' + self.name+ '_from_' + str(t_initial)
                y_vars[column] = y_t_initial
            
            self.y_vars = pd.DataFrame(y_vars)

            # Flow_vars contains dispatched consumption
            # self.flow_vars_in = [pl.LpVariable(self.name + '_' + str(t),
            #                                 lowBound=-self.maximum_hourly_dispatch) for t in range(self.nb_of_timesteps)]
            # Dispatched consumption is the sum of the dispatch of every hourly consumption
            for t in range(self.nb_of_timesteps):
                model += pl.lpSum(self.y_vars.loc[t]) == self.flow_vars_in[t]


        return hubs, model

class Source(Component):
    """
    Sub-class of Component class representing an energy source. Note that if specified,it is possible to send energy
    toward the source (reinjecting electricity to the grid for instance).

    Inherited Attributes:
        name (str): Name of the component.
        energy (str): Main energy type of the component.    
        environment (str): Main environment of the component.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints).
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
        flow_vars_out (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-out variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_out[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
    
    Args:
        name (str): Name of the component.
        description (dict): Portion of the configuration file describing the characteristics of the component links.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, name, description, nb_of_timesteps, log=False):
        super().__init__(name, description['energy'], description['environment'], description, nb_of_timesteps, log=log,)
        
        if log:
            print(name + ' created.')

    def build_equations(self, hubs, model, log=False):
        """
        Calls method Component.link() to build variables and constraints related to the energy flows.

        Args:
            hubs (pandas.DataFrame): Table of all hubs.
            model (pulp.LpProblem): Pulp linear problem to be optimized.
            log (bool): If True, additional information will be printed throughout the process.
        
        Returns:
            pandas.DataFrame: Updated table of all hubs.
            pulp.LpProblem: Updated pulp linear problem to be optimized.
        """
        hubs, model = self.link(hubs=hubs, model=model, log=log)
        if log:
            print(self.name + ' linked to hubs, model updated.')
        return hubs, model
    
class EnvironmentsConnection(Component):
    """
    Sub-class of Component class used to represent the interface between two different environments.

    Inherited Attributes:
        name (str): Name of the component.
        energy (str): Not used, forced to energy=''.
        environment (str): Main environment of the component. For an EnvironmentsConnection object, corresponds
            to environment1.
        description (dict): Portion of the configuration file describing the characteristics of the links
        nb_of_timesteps (int): Number of timesteps of the simulation.
        maximum (float | str): maximum value(s) of energy flow in and out of the component. Can be one value
            (ex: maximum electricity exchanges through grid) or a sequence of the same duration of the simulation
            duration (ex: hourly PV production). In this case, a string describing where the data is stored must
            be given under this form : "path/to/data.csv//column_name_of_maximum". Value(s) must be given in the
            unit of the energy type attached to the object.
        maximum_in (float | str): maximum of the energy flow in the sense Hub -> Component. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_in and maximum are given (only maximum_in in
            used in the problem constraints).
        maximum_out (float | str): maximum of the energy flow in the sense Component -> Hub. Same format as maximum,
            default value is maximum, crushes maximum if both maximum_out and maximum are given (only maximum_out in
            used in the problem constraints).
        minimum (float | str): minimum value(s) of energy flow in and out of the component. Can be one value or a
            sequence of the same duration of the simulation duration. In this case, a string describing where the
            data is stored must be given under this form : "path/to/data.csv//column_name_of_minimum". Value(s)
            must be given in the unit of the energy type attached to the object.
        minimum_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_in and minimum are given (only minimum_in in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        minimum_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as minimum,
            default value is minimum, crushes minimum if both minimum_out and minimum are given (only minimum_out in
            used in the problem constraints). If demand is not dispatchable, will be force to minimum_in=value.
        cost (float | str): cost value(s) of energy flow in and out of the component per energy type unit. Can be one
            value or a sequence of the same duration of the simulation duration. In this case, a string describing
            where the data is stored must be given under this form : "path/to/data.csv//column_name_of_cost". Value(s)
            must be given in cost unit per unit of the energy type attached to the object. Default is 0. A positive
            value is a cost, a negative value is a gain.
        cost_in (float | str): minimum of the energy flow in the sense Hub -> Component. Same format as cost, default
            value is -cost, crushes cost if both cost_in and cost are given (only cost_in in used in the problem
            constraints).
        cost_out (float | str): minimum of the energy flow in the sense Component -> Hub. Same format as cost, default
            value is cost, crushes cost if both cost_out and cost are given (only cost_out in used in the problem
            constraints).
        factor (int | float | str): actual minimum and maximum flow values are multiplied by factor. Example of use: if the
            input timeseries given as maximum_out describes the production of a 1 kWc PV panel but we want to have 5 kWc
            of PV installed, set factor to 5 ; if the input timeseries describes a demand that we want to split as 50%
            dispatchable and 50% not dispatchable, create to Demand objects (one with dispatchable=True, the other with
            dispatchable=False) with this input timeseries and set factor=0.5 for both objects.
        factor_low_bound (int | float): Used if factor is set to 'auto'. factor_low_bound <= factor.
        factor_up_bound (int | float): Used if factor is set to 'auto'. factor_up_bound => factor.
        factor_type (str): Used if factor is set to 'auto' and set the factor type. Can be 'Continuous', 'Integer' or 'Binary'.
            Example of use for 'Integer' : how many PV panels of 200 Wc should be installed ? Example of use for 'Binary' :
            should a generator be installed ? (Yes : factor=1., No : factor=0.)
        installation_cost (float): Cost for one unit installed (factor=1.) Total installation cost will be
            installation_cost * factor.
        flow_vars_in (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-in variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_in[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.
        flow_vars_out (list[pulp.LpVariable]): List of length nb_of_timesteps containing the flow-out variables (from Hub to Component)
            of the Component at each timestep. Once the problem solved, flow_vars_out[t].value() returns the optimized value of the
            variable for each timestep t between 0 and nb_of_timesteps-1.

    Attributes:
        environment1 (str): Name of 1st environment to connect.
        environment2 (str): Name of 2nd environment to connect.
        vars_out (dict[str: list[pulp.LpVariable]]): Dictionnary with energy types common to both environment1 and environment2 as keys,
            containing lists of length nb_of_timesteps containing the flow variables from Hub(environment1) to Hub(environment2) for each
            common energy type. Once the problem solved, vars_out[energy][t].value() returns the optimized value of these flow variables
            from for each timestep t between 0 and nb_of_timesteps-1.
        vars_in (dict[str: list[pulp.LpVariable]]): Dictionnary with energy types common to both environment1 and environment2 as keys,
            containing lists of length nb_of_timesteps containing the flow variables from Hub(environment2) to Hub(environment1) for each
            common energy type. Once the problem solved, vars_in[energy][t].value() returns the optimized value of these flow variables
            from for each timestep t between 0 and nb_of_timesteps-1.

    Args:
        environment1 (str): Name of 1st environment to connect.
        environment2 (str): Name of 2nd environment to connect.
        descriptions (dict): Sub-section of the configuration file describing the characteristics of the links.
        i (int): Used to iterate within a list of specifications.
        nb_of_timesteps (int): Number of timesteps of the simulation.
        log (bool): If True, additional information will be printed throughout the initialization.
    """
    def __init__(self, environment1, environment2, descriptions, i, nb_of_timesteps, log=False):
        self.environment1 = environment1
        self.environment2 = environment2
        name = self.environment1 + '_' + self.environment2

        super().__init__(name, '', environment1, description=descriptions, i=i, nb_of_timesteps=nb_of_timesteps, isHub=False, log=log,)
        
    def connect_as_input(self, hubs, model, log=False):
            """
            Creates vars_out and vars_in to store the flow variables between hubs of environment1 and hubs of environment2, add them
            to hubs of environment1 and hubs of environment2 equations, adds minimum and maximum constraints (describing the connection
            conditions between the two environments) to the linear problem, and updates the model objective function with costs attributed
            to the flow variables.

            Args:
                hubs (pandas.DataFrame): Table of all hubs.
                model (pulp.LpProblem): Pulp linear problem to be optimized.
                log (bool): If True, additional information will be printed throughout the process.
            
            Returns:
                pandas.DataFrame: Updated table of all hubs.
                pulp.LpProblem: Updated pulp linear problem to be optimized.
            """
            self.vars_out = {}
            self.vars_in = {}
            # If connexion time slots between environment 1 and environment 2 is an input data (then never a variable)
            for hub1, hub2, energy in zip(hubs.loc[self.environment1], hubs.loc[self.environment2], hubs.columns):
                if type(hub1) == Hub and type(hub2) == Hub:
                    # Create flow variables between hub1 and hub2
                    hub1_to_hub2 = [pl.LpVariable(energy + '_' + self.environment1 + '_hub_to_'
                                                + self.environment2 + '_hub_'
                                                + str(t)) for t in range(hub1.nb_of_timesteps)]
                    hub2_to_hub1 = [pl.LpVariable(energy + '_' + self.environment2 + '_hub_to_'
                                                + self.environment1 + '_hub_'
                                                + str(t)) for t in range(hub1.nb_of_timesteps)]
                    
                    # Set bounds to flow between environment1 and environment2 (= 0. when disconnected)
                    for t in range(hub1.nb_of_timesteps):
                        model += hub1_to_hub2[t] >= bound(self.minimum_out, t)
                        model += hub2_to_hub1[t] >= bound(self.minimum_in, t)
                        model += hub1_to_hub2[t] <= bound(self.maximum_out, t)
                        model += hub2_to_hub1[t] <= bound(self.maximum_in, t)

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

                    model.objective += pl.lpSum([hub1_to_hub2[t]*bound(self.cost_out, t)])
                    model.objective += pl.lpSum([hub2_to_hub1[t]*bound(self.cost_in, t)])

                    if log:
                        print(hub1.name + ' and ' + hub2.name + ' connected')

            return hubs, model
