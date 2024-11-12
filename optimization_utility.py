import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK45
import math 
import pandas as pd
import utility as ut
from scipy.signal import find_peaks
from scipy.optimize import minimize
import utility as u
from scipy.optimize import LinearConstraint,NonlinearConstraint,Bounds,differential_evolution,minimize
from RC_utility import RC
import pickle
import best_parameters_validation as best_prameters_validation
from LinearRegression import LinearRegressor
import seaborn as sns
from ZebiakCaneDataUtility import ZebiakCaneModel
import os
from pathlib import Path
import json
import sys
from matplotlib import rc, rcParams
from scipy.stats import pearsonr
from utility import compute_monthly_data

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"
rcParams['font.size']=10

class CNOPSolver():

    def __init__(self,model,zc,spin=None,
    cycle=None,annual_period=None):

        self.model=model
        self.zc=zc

        if(isinstance(self.model,RC)):
            self.evaluation_data=zc.X_validation
            self.spin=spin
            self.cycle=cycle
            self.validation_time=zc.time_line_validation
            self.annual_period=annual_period
                

    def __callable_function_optimal_perturbation(self,perturbation,
    lead_time,evaluation_data,
    time_line_validation,sign=-1):

        if(isinstance(self.model,RC)):

            w1=2
            w2=50

            perturbation=perturbation.copy()
            perturbation[0]=perturbation[0]/w1
            perturbation[1]=perturbation[1]/w2
            perturbation[2]=perturbation[2]/w2

            evaluation_data=evaluation_data.copy()
            spin=self.spin
            time_line=time_line_validation.copy()

            reconstruction_perturbed=self.model.return_reconstruction(evaluation_data,
            spin,cycle=True,time_line_validation=time_line,
            apply_perturbation=True,perturbation=perturbation,
            total_lenght=lead_time)

            reconstruction_not_perturbed=self.model.return_reconstruction(evaluation_data,
            spin,cycle=True,time_line_validation=time_line,
            apply_perturbation=False,perturbation=None,
            total_lenght=lead_time)

            time_line_validation=time_line_validation[spin:]

            T_perturbated=(reconstruction_perturbed[:,1]*2)
            T_not_perturbed=(reconstruction_not_perturbed[:,1]*2)

            T_perturbated=compute_monthly_data(T_perturbated,time_line_validation)
            T_not_perturbed=compute_monthly_data(T_not_perturbed,time_line_validation)

            distance=np.sqrt(np.sum((T_perturbated-T_not_perturbed)**2))

        return sign*distance

    def conditional_non_linear_perturbation(self,delta,lead_time,indexes_interval,
    trials=10):

        w1 = 2
        w2 = 50

        evaluation_data=self.zc.X_validation[:,indexes_interval[0]:indexes_interval[1]+1].copy()
        time_line_validation=self.zc.time_line_validation[indexes_interval[0]:indexes_interval[1]+1].copy()

        con = lambda x: math.sqrt((x[0]/w1)**2+(x[1]/w2)**2+(x[2]/w2)**2)
        nc = NonlinearConstraint(con,-np.inf,delta)

        initial_conditions = np.random.uniform(-1, 1, (trials, 3))
        
        scaling_factors = np.sqrt((initial_conditions[:, 0]/w1)**2 + 
                    (initial_conditions[:, 1]/w2)**2 + 
                    (initial_conditions[:, 2]/w2)**2)

        # Scale the initial conditions to satisfy the constraint
        initial_conditions /= scaling_factors[:, np.newaxis] / delta
        
        list_initial_condistions=[]
        list_final_results=[]

        for i,initial_condition in enumerate(initial_conditions):
            res=minimize(lambda x: self.__callable_function_optimal_perturbation(x,
            lead_time=lead_time,
            evaluation_data=evaluation_data,time_line_validation=time_line_validation),
            initial_condition,method="COBYLA",constraints=nc,
            options={'maxiter': 5000, 'tol': 1e-4})

            if(res.success==False):
                continue
        
            if(i==0):
                best_result=res
            else:
                if(res.fun<=best_result.fun):
                    best_result=res

            list_initial_condistions.append([res.x[0],res.x[1],res.x[2]])
            list_final_results.append(res.fun)
                
        return best_result
    
def compute_CNOPs_different_conditions(initial_month,lead_time,
zc,spin,cycle,delta,variables,weekly=True):
    
    w1=2
    w2=50

    dictionary_results={'TE':[],'HW':[],'HE':[],
    'perturbation_amplitude':[],'distance':[],'correlation':[],
    'rms':[]}

    if(weekly):
        lead_time=lead_time*3

    with open("ReservoirWeights_experiments/Drag{}BestWeightsLead{}Variables{}_tmp".format(drag,lead_time,variables),'rb') as file:
        rc=pickle.load(file)
    
    sys.stdout.flush()
    initial_month_tmp=initial_month
      
    while(True):

        if(initial_month_tmp+spin+lead_time>=zc.X_validation.shape[1]):
            break

        print("Analyzing month:{}".format(initial_month_tmp))
        print("Analyzing lead time{}".format(lead_time))

        cnop=CNOPSolver(model=rc,zc=zc,spin=spin,cycle=cycle,
        annual_period=12)
        indexes_validation_interval=[initial_month_tmp,initial_month_tmp+spin+lead_time]
        res=cnop.conditional_non_linear_perturbation(delta,lead_time,indexes_validation_interval)

        dictionary_results['TE'].append(res.x[0])
        dictionary_results['HE'].append(res.x[1])
        dictionary_results['HW'].append(res.x[2])
        dictionary_results['perturbation_amplitude'].append(math.sqrt((res.x[0]/w1)**2+(res.x[1]/w2)**2+(res.x[2]/w2)**2))

        dictionary_results['distance'].append(res.fun) 

        print("*************************")
        print("\n")
        print("Distance estimated Cobyla:{}".format(res.fun))
        print(res)
        print("\n")
        print("*************************")
        sys.stdout.flush()

        evaluation_data=zc.X_validation[:,indexes_validation_interval[0]:indexes_validation_interval[1]+1].copy()
        time_line_validation=zc.time_line_validation[indexes_validation_interval[0]:indexes_validation_interval[1]+1].copy()

        perturbation=[res.x[0],res.x[1],res.x[2]]
        perturbation[0]=perturbation[0]/w1
        perturbation[1]=perturbation[1]/w2
        perturbation[2]=perturbation[2]/w2

        reconstruction_perturbed=rc.return_reconstruction(evaluation_data,
        spin,cycle=True,time_line_validation=time_line_validation,
        apply_perturbation=True,perturbation=perturbation,
        total_lenght=lead_time)

        reconstruction_not_perturbed=rc.return_reconstruction(evaluation_data,
        spin,cycle=True,time_line_validation=time_line_validation,
        apply_perturbation=False,perturbation=None,
        total_lenght=lead_time)

       

        T_perturbated=(reconstruction_perturbed[:,1]*2)
        T_not_perturbed=(reconstruction_not_perturbed[:,1]*2)

        if(weekly):
            T_perturbated=compute_monthly_data(T_perturbated,time_line_validation)
            T_not_perturbed=compute_monthly_data(T_not_perturbed,time_line_validation)

        correlation=pearsonr(T_perturbated,T_not_perturbed)
        correlation=correlation.statistic
        rms=np.sqrt(np.mean((T_perturbated-T_not_perturbed)**2))
        distance=np.sqrt(np.sum((T_perturbated-T_not_perturbed)**2))


        dictionary_results['correlation'].append(correlation)
        dictionary_results['rms'].append(rms)

        print("*******************************")
        print("\n")
        print("Distance after Cobyla:{}".format(distance))
        print("Correlation:{}".format(correlation))
        print("RMS:{}".format(rms))
        print("\n")
        print("\n")
        print(res)
        print("\n")
        print("********************************")
        sys.stdout.flush()

        if(weekly):
            initial_month_tmp=initial_month_tmp+36
        else:
            initial_month_tmp=initial_month_tmp+12
    
    directory = Path("./resultsCNOPsRC/Drag{}".format(drag))
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
    json_file_name="CNOPsIm{}Delta{}Lead{}Interval[{},{}]Weekly{}Variables{}_tmp.json".format(initial_month,
    delta,lead_time,zc.validation_interval[0],zc.validation_interval[1],weekly,variables)
    file=directory.joinpath(json_file_name)
 
    with open(file, 'w') as json_file:
        json.dump(dictionary_results, json_file)

def read_results(drag,initial_month,lead,interval,delta):

    file_path = './resultsCNOPsRC/Drag{}/CNOPsIm{}Delta{}Lead{}Interval[{},{}].json'.format(drag,
    initial_month,delta,lead,interval[0],interval[1])

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

#compute CNOPs for RC and save results in the directory resultsCNOPsRC the bestweights are read from the directory ReservoirWeights_experiments 
#that contains the best weights according to our experiments
spin=180
warm_up_iterations=0

transitional_years=300
train_years=300
validation_years=[950,1000]

drag=0.9
variables=["Nino3","ThE","ThW","Wind"]
zc=ZebiakCaneModel("../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",drag,transitional_years,train_years,
validation_years,variables,True)
variables=4

for delta in [0.05]:
    for lead_time in [3,6,9]:
        for initial_month in [0,1,2,3,4,5,6,7,8,9,10,11]:
            compute_CNOPs_different_conditions(initial_month,lead_time,
            zc,spin,cycle=True,delta=delta,variables=variables,weekly=True)

