# -*- coding: utf-8 -*-

"""
pso.py
Functions related to implementation of PSO algorithm.


"""



from numpy.core.records import array
import Simulation.global_ as global_
from PSO import fitness_func as fit

import os

import numpy as np
import importlib

import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from sklearn.metrics import mean_squared_error

from numpy.random import seed
from numpy.random import randn
import copy


class SearchSpaceInitializer(object):

    def sample(self,dim,minf,maxf,n):
        pass


class UniformSSInitializer(SearchSpaceInitializer):

    def sample(self, dim, minf, maxf,n):
        x = np.zeros((n, dim))
        for i in range(n):
            x[i] = np.random.uniform(minf, maxf, dim)
        return x




class Particle:
    """
    Particle class enabling the creation of multiple and dynamic number of particles.
    """
class Fish(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.weight = np.nan
        self.cost = np.nan
        self.has_improved = False

class FSS(object):
    
    global step_individual_init
    global step_individual_final
    global step_volitive_init
    global step_volitive_final
    global w_scale
    global min_w 
    
    step_individual_init = 0.1
    step_individual_final = 0.01
    
    step_volitive_init = 0.1
    step_volitive_final = 0.1
    
    w_scale = 1
    min_w = 1
    
    

    def __init__(self, dim, minf, maxf, n_iter, school_size):
            

            
            self.dim = dim
            self.minf = np.array(minf)
            self.maxf = np.array(maxf)
            self.n_iter = n_iter

            self.school_size = school_size
            self.step_individual_init = step_individual_init
            self.step_individual_final = step_individual_final
            self.step_volitive_init = step_volitive_init
            self.step_volitive_final = step_volitive_final

            self.curr_step_individual = self.step_individual_init * (self.maxf - self.minf)
            self.curr_step_volitive = self.step_volitive_init * (self.maxf - self.minf)
            self.min_w = min_w
            self.w_scale = w_scale
            self.prev_weight_school = 0.0
            self.curr_weight_school = 0.0
            self.best_fish = None

            #self.optimum_cost_tracking_iter = []
            #self.optimum_cost_tracking_eval = []
            self.best_fish_pos = []
            self.BaryCenter = []


    """Create particles swarm"""

    def __gen_weight(self):
        return self.w_scale / 2.0

    #def __init_fss(self):
        #self.optimum_cost_tracking_iter = []
        #self.optimum_cost_tracking_eval = []

    def __init_fish(self, pos):
        fish = Fish(self.dim)
        fish.pos = pos
        fish.weight = self.__gen_weight()
        #fish.cost = self.objective_function.evaluate(fish.pos)
        #self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        return fish

    def init_school(self):
        self.best_fish = Fish(self.dim)
        self.best_fish.cost = np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        positions = UniformSSInitializer.sample(self,self.dim, self.minf , self.maxf , self.school_size)

        for idx in range(self.school_size):
            fish = self.__init_fish(positions[idx])
            self.school.append(fish)
            self.curr_weight_school += fish.weight
        self.prev_weight_school = self.curr_weight_school
        self.update_best_fish()
        #self.optimum_cost_tracking_iter.append(self.best_fish.cost)
        
    def max_delta_cost(self):
        max_ = 0
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0
        for fish in self.school:
            self.curr_weight_school += fish.weight

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=float)
        density = 0.0

        for fish in self.school:
            density += fish.weight
            for dim in range(self.dim):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density
        
        #print(barycenter)
        
        self.BaryCenter.append(barycenter)
        
        return barycenter

    def update_steps(self, curr_iter):
        self.curr_step_individual = self.step_individual_init - curr_iter * float(
            self.step_individual_init - self.step_individual_final) / self.n_iter

        self.curr_step_volitive = self.step_volitive_init - curr_iter * float(
            self.step_volitive_init - self.step_volitive_final) / self.n_iter

    def update_best_fish(self):
        for fish in self.school:
            if self.best_fish.cost > fish.cost:
                self.best_fish = copy.copy(fish)

    def feeding(self):
        for fish in self.school:
            if self.max_delta_cost():
                fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    def individual_movement(self,funcion,i, bypass):
        j = 0
        for fish in self.school:
            j += 1
            new_pos = np.zeros((self.dim,), dtype= float)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + (self.curr_step_individual[dim] * np.random.uniform(-1, 1))
                if new_pos[dim] < self.minf[dim]:
                    new_pos[dim] = self.minf[dim]
                elif new_pos[dim] > self.maxf[dim]:
                    new_pos[dim] = self.maxf[dim]
                    
            cost, data_to_store,derivative_array = funcion(new_pos, i ,j, bypass )
            #self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            if cost < fish.cost:
                fish.delta_cost = abs(cost - fish.cost)
                fish.cost = cost
                delta_pos = np.zeros((self.dim,), dtype= float)
                for idx in range(self.dim):
                    delta_pos[idx] = new_pos[idx] - fish.pos[idx]
                fish.delta_pos = delta_pos
                fish.pos = new_pos
            else:
                fish.delta_pos = np.zeros((self.dim,), dtype= float)
                fish.delta_cost = 0
                
            

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dim,), dtype= float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dim):
                cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
        for dim in range(self.dim):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype= float)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < self.minf[dim]:
                    new_pos[dim] = self.minf[dim]
                elif new_pos[dim] > self.maxf[dim]:
                    new_pos[dim] = self.maxf[dim]

            fish.pos = new_pos

    def collective_volitive_movement(self,funcion,i,bypass,db_manager):
        self.total_school_weight()
        barycenter = self.calculate_barycenter()
        j =0
        
        for fish in self.school:
            j += 1
            new_pos = np.zeros((self.dim,), dtype= float)
            for dim in range(self.dim):
                if self.curr_weight_school > self.prev_weight_school:
                    new_pos[dim] = fish.pos[dim] - ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive[dim] *
                                                    np.random.uniform(0, 1))
                else:
                    new_pos[dim] = fish.pos[dim] + ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive[dim] *
                                                    np.random.uniform(0, 1))
                if new_pos[dim] < self.minf[dim]:
                    new_pos[dim] = self.minf[dim]
                elif new_pos[dim] > self.maxf[dim]:
                    new_pos[dim] = self.maxf[dim]

            cost, data_to_store, derivative_array = funcion(new_pos,i,j,bypass)
            #self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            fish.cost = cost
            fish.pos = new_pos
            
            data_to_plot={
                "dy2_31":derivative_array[0],
                "x_31":derivative_array[1],
                "y2_21":derivative_array[2],
                "x_21":derivative_array[3]
            }
            
            db_manager.save_data_to_plot(data_to_plot,i+1,j)
    

    def optimize(self):
        #self.__init_fss()
        self.init_school()

        for i in range(self.n_iter):
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()  
            self.best_fish_pos.append(self.best_fish.pos) 
    

#Fitness func how close a given solution is to the optimum solution
def fitness(results,iter):
    
    importlib.reload(fit)

    s11 = results[0]
    s21 = results[1]
    s31 = results[2]
    s41 = results[3]
    amp_imbalance = results[4]
    #rashid 166-208
    #regions=[169,205] 

    #standar 2+3 band
    #regions = [70, 108]
    
    #band 3 84-116
    #regions = [68, 113]

    regions=[87,113] 
    
    num_of_constraints= 5 #we are evaluating two signals S31 y S41

    fitnessObj=fit.fitness_func()
    fitnessObj.regions = regions
    fitnessObj.set_priorization_regions(regions,s21[:,0],s21[:,1],s31[:,1],amp_imbalance[:,1]) #create array of subdivided domain values
    fitnessObj.set_bandwidth(s21[:,0]) #set bandwidtharray and normalize domain
    
    funcs = fitnessObj.set_sub_functions(deviation_s31,deviation_s21,oscillation_s31,oscillation_s21)
    fi_matrix = calculate_subfunction(funcs,fitnessObj,num_of_constraints)
    #you can create as many constraint as you want 


   

    _=fitnessObj.set_constraints('region_1_S31',true_value=-3.0, min_val=-3.5, max_val=-2.5,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_1_S21',true_value=-5.0, min_val=-3.5, max_val=-2.5,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_1_over_S31',true_value=-2, min_val=-3.5, max_val=-1,type_constraint='overshoot')
    _=fitnessObj.set_constraints('region_1_over_S21',true_value=-3, min_val=-5, max_val=-1,type_constraint='overshoot')


    _=fitnessObj.set_constraints('region_2_S31',true_value=-2.9, min_val=-3.3, max_val=-2.7,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_2_S21',true_value=-3.1, min_val=-3.3, max_val=-2.7,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_2_over_S31',true_value=-3.2, min_val=-3.5, max_val=-2.5,type_constraint='overshoot')
    _=fitnessObj.set_constraints('region_2_over_S21',true_value=-2.8, min_val=-3.5, max_val=-2.5,type_constraint='overshoot')


    _=fitnessObj.set_constraints('region_3_S31',true_value=-3.0, min_val=-3.5, max_val=-2.5,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_3_S21',true_value=-3.0, min_val=-3.5, max_val=-2.5,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_3_over_S31',true_value=-3, min_val=-3.5, max_val=-1,type_constraint='overshoot')
    _=fitnessObj.set_constraints('region_3_over_S21',true_value=-3, min_val=-5, max_val=-1,type_constraint='overshoot')


    _=fitnessObj.set_constraints('region_1_ampimb',true_value=1, min_val=0.0, max_val=0.9,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_2_ampimb',true_value=0.8, min_val=0.05, max_val=0.8,type_constraint='in_limit')
    _=fitnessObj.set_constraints('region_3_ampimb',true_value=1, min_val=0.0, max_val=0.8,type_constraint='in_limit')

    
    """Evaluate S31 Constraints"""
    evaluated_constraint_S31=[]

    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[0],
                                        constrain_name='region_1_S31'))
    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[1],
                                        constrain_name='region_2_S31'))
    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[2],
                                        constrain_name='region_3_S31'))


    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[0],
                                        constrain_name='region_1_over_S31'))
    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[1],
                                        constrain_name='region_2_over_S31'))
    evaluated_constraint_S31.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s31[2],
                                        constrain_name='region_3_over_S31'))
    """Evaluate S21 Constraints"""
    evaluated_constraint_S21=[]
    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[0],
                                        constrain_name='region_1_S21'))
    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[1],
                                        constrain_name='region_2_S21'))
    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[2],
                                        constrain_name='region_3_S21'))

    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[0],
                                        constrain_name='region_1_over_S21'))
    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[1],
                                        constrain_name='region_2_over_S21'))
    evaluated_constraint_S21.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_s21[2],
                                        constrain_name='region_3_over_S21'))

    evaluated_constraint_ampImbalance=[]
    
    evaluated_constraint_ampImbalance.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_amp_imbalance[0],
                                        constrain_name='region_1_ampimb'))
    evaluated_constraint_ampImbalance.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_amp_imbalance[1],
                                        constrain_name='region_2_ampimb'))
    evaluated_constraint_ampImbalance.append(
        fitnessObj.evaluate_constraints(data_to_asses=fitnessObj.Y_regions_amp_imbalance[2],
                                        constrain_name='region_3_ampimb'))

    """Penalties for Dynamic Weighting"""
    num_of_regions= len(fitnessObj.normalized_regions_bandwidth)
    #Constraint for S31 Level
    #Constraint for S41 Level
    #Constraint for S31 Overshoot
    #Constraint for S41 Overshoot
    #ampimbalance level

    penalties = np.empty([num_of_regions, num_of_constraints])

    for i in range(num_of_regions):
        penalties[i][0] = 1-np.mean(evaluated_constraint_S31[i])#S31 general constraints by region

    for i in range(num_of_regions):    
        penalties[i][1] = 1-np.mean(evaluated_constraint_S21[i])#S41 general constraints by region

    for i in range(num_of_regions, num_of_regions*2):
        penalties[i-num_of_regions][2] = np.mean(evaluated_constraint_S31[i])#S31 overshoot constraints by region

    for i in range(num_of_regions, num_of_regions*2):
        penalties[i-num_of_regions][3] = np.mean(evaluated_constraint_S21[i])#S31 overshoot constraints by region
    
    for i in range(num_of_regions):
        penalties[i-num_of_regions][4] = np.mean(evaluated_constraint_ampImbalance[i])#S31 general constraints by region

    """row = every region
    column every signal evaluated.
    We have constraint evaluated in every region for every signal"""

    """Penalty matrix creates dynamics weigths adding a penalty to bad constraint performance
    [[S31_region1 S31_region2 S31_region3]
    [S41_region1 S41_region2 S41_region3]]
    """

    
    weights = fitnessObj.set_weights()
    weights_penalized = (100*weights + penalties)
    xmax, xmin = weights_penalized.max(), weights_penalized.min()   
    weights_penalized = (weights_penalized - xmin)/(xmax - xmin)
    """Evaluate products and sums"""
    prod =np.dot(weights_penalized.T,fi_matrix )

    log_prod = np.log(100*prod)
    #log_prod = prod

    #Just take diagonal as they are de FiWi product.
    products = np.diagonal(log_prod)  
    lambda_= 0.008 # 10 creates a strong relevance on oscillations

    #Cost counts for signals being close to truth value
    #and being inside constraint boundaries in every region
    #cost = np.sum(products[0:2])/len(funcs)

    #considers all functions witouth amp constraint
    """Usar en caso de guía plana"""
    #cost = np.sum([products[0],products[1]])


    #considers just all function. work well when testing wings
    """Usar en caso de guía con extrusiones"""
    cost = np.sum([products[0],products[1]])
    #cost = np.sum([products[4]])

    #Oscillation term counts for the integral to measure big resonances 
    #in every region and counts for the overshoot values in every region
    oscillation = lambda_*np.sum([products[2],products[3]])

    fitness =  cost + oscillation
    fitness = fitness.round(decimals=5, out=None)

    print("fitness="+str(iter))
    print("fitness="+str(fitness))
    fitnessObj.reset_constraints()

    del fitnessObj.regions
    del fitnessObj.normalized_bandwith #whole domain normalized
    del fitnessObj.regions_bandwidth #array of subdivided freq axis
    del fitnessObj.normalized_regions_bandwidth
    del fitnessObj.functions_array
    del fitnessObj
    del fi_matrix
    del funcs
 

    data_to_store = {
        "weights":{
            "note":"changing weights-focus on S31 and S41 parameters to test their part of the function by removing s11 s21 closeness eval.",
           "w_similarity_s11": 0,
            "w_similarity_s31":0,
            "w_closeness_s31_s21":0,
            "w_grad":0
        },
        "function":'testing new function',
    }
    #si cambio la formula, cambio el string para almacenarla

    #return indice,data_to_store,[dydx2_31, x_values_31, dydx2_41, x_values_41]
    return fitness, data_to_store,[[], [], [], []]


def calculate_subfunction(sub_functions,fitnessObj,num_of_constraints):
    
    #subfunctions_matrix = np.empty([len(fitnessObj.normalized_regions_bandwidth), (num_of_constraints)])
    subfunctions_matrix = np.ones((len(fitnessObj.normalized_regions_bandwidth), (num_of_constraints)))
    for idx , region in  enumerate(fitnessObj.normalized_regions_bandwidth):
        data = []
        if idx==0:#deviation_s31
            pass

        if idx==1:#deviation_s41
            pass

        if idx==2:#oscillation_s31
            pass


        if idx==3: #oscillation_s41 
            pass

        """Calculating for S31 MSE"""
        result = sub_functions[0](fitnessObj.Y_regions_s31[idx],-3)
        #print("S31 region " + str(idx)+", MSE="+str(result))
        print(result)

        subfunctions_matrix[idx][0] = result

        """Calculating for S41 MSE"""
        result = sub_functions[1](fitnessObj.Y_regions_s21[idx],-3)
        #print("S41 region " + str(idx)+", MSE="+str(result))

        subfunctions_matrix[idx][1] = result


        """Calculatign for S31"""
        data_31 = np.stack((region,fitnessObj.Y_regions_s31[idx]), axis=0)
        result = sub_functions[2](data_31)

        #print("S31 region " + str(idx)+", abs-integral of the 2nd derivative="+str(np.abs(result)))

        subfunctions_matrix[idx][2] = result

        """Calculatign for S41 integral of 2nd der."""

        data_21 = np.stack((region,fitnessObj.Y_regions_s21[idx]), axis=0)
        result = sub_functions[3](data_21)

        #print("S41 region " + str(idx)+", abs-integral of the 2nd derivative="+str(np.abs(result)))

        subfunctions_matrix[idx][3] = result

    return subfunctions_matrix



def derivate(data ):
    dydx1 = np.gradient(data[1],data[0])
    dydx2 = np.gradient(dydx1,data[0])
    return dydx2,data[0]

def integrate_data (x,y):
    y_int = np.abs(integrate.trapz(y, x=x))
    return y_int

def deviation_s31(data_Y,level):
    
    Y_true = np.full( shape=len(data_Y),fill_value=level, dtype=np.int32)
    Y_pred = data_Y

    # Calculation of Mean Squared Error (MSE)
    MSE = mean_squared_error(Y_true,Y_pred)

    return MSE

def deviation_s21(data_Y,level):
    Y_true = np.full( shape=len(data_Y),fill_value=level, dtype=np.int32)
    Y_pred = data_Y

    # Calculation of Mean Squared Error (MSE)
    MSE = mean_squared_error(Y_true,Y_pred)

    return MSE

def oscillation_s31(data):
    
    derivative = derivate(data)
    result = integrate_data(derivative[0],data[0])
    return result

def oscillation_s21(data):
    
    derivative = derivate(data)
    result = integrate_data(derivative[0],data[0])
    return result
    
    
    

    
