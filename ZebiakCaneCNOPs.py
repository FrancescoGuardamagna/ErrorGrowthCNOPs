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
from LinearRegression import LinearRegressor
import seaborn as sns
from scipy.optimize import minimize
import os
from matplotlib import rc, rcParams
from pathlib import Path
import json
from scipy.stats import pearsonr
import sys
import subprocess
import time
import matplotlib.pyplot as plt 

os.chdir("./ZebiakCaneFiles")

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 30
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 30
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"
rcParams['font.size']=30


class CNOPSolverZebiak():

    def __init__(self):

        return
    
    def objective_function_different_conditions(self,x,configuration_file,
    main_file_field,nino_file):
        
        try:

            w1=2
            w2=50

            os.chdir("../ZebiakCaneFiles")

            path_configuration="./configuration_files/"+configuration_file

            with open(path_configuration, 'r') as file:
                lines = file.readlines()
                lines[54]="PF     =  0.0\n"
                lines[5]="P_NINO =  0.0\n"
                lines[6]="P_THW  =  0.0\n"
                lines[7]="P_THE  =  0.0\n"
            
            file.close()
            
            with open(path_configuration, 'w') as file:
                file.writelines(lines)

            #os.system("./"+main_file_field+" > /dev/null 2>&1") 
            #os.system("./"+main_file_field)

            command_main=["./"+main_file_field]
            result_main=subprocess.run(command_main,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            err_main=result_main.stderr
            err_main=err_main.decode('utf-8')
            output_main=result_main.stdout
            output_main=output_main.decode('utf-8')

            print("Perturbation Nino:{}".format(x[0]/2))
            print("Perturbation HE:{}".format(x[1]/50))
            print("Perturbation HW:{}".format(x[2]/50))

            nino_not_perturbed=pd.read_csv(nino_file,header=None)
            nino_not_perturbed=np.array(nino_not_perturbed)
            nino_not_perturbed_weekly=pd.read_csv("nino_weekly.gdat",header=None)
            nino_not_perturbed_weekly=np.array(nino_not_perturbed_weekly)
            thermocline_east_not_perturbed=pd.read_csv('H_east_weekly.gdat',header=None)
            thermocline_west_not_perturbed=pd.read_csv('H_west_weekly.gdat',header=None)
            thermocline_east_not_perturbed=np.array(thermocline_east_not_perturbed)
            thermocline_west_not_perturbed=np.array(thermocline_west_not_perturbed)

            thermocline_east_not_perturbed_H1=pd.read_csv('H_east_regridded_weekly.gdat',header=None)
            thermocline_west_not_perturbed_H1=pd.read_csv('H_west_regridded_weekly.gdat',header=None)
            thermocline_east_not_perturbed_H1=np.array(thermocline_east_not_perturbed_H1)
            thermocline_west_not_perturbed_H1=np.array(thermocline_west_not_perturbed_H1)

            file.close()

            with open(path_configuration, 'r') as file:
                lines = file.readlines()
                lines[54]="PF     =  1.0\n"
                lines[5]="P_NINO =  {}\n".format(x[0])
                lines[6]="P_THW  =  {}\n".format(x[1])
                lines[7]="P_THE  =  {}\n".format(x[2])

            with open(path_configuration, 'w') as file:
                file.writelines(lines)
            
            file.close()

            #os.system("./"+main_file_field+" > /dev/null 2>&1") 
            #os.system("./"+main_file_field)
            command_main_pertubed=["./"+main_file_field]
            result_main_pertubed=subprocess.run(command_main_pertubed,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            output_main_pertubed=result_main_pertubed.stdout
            output_main_pertubed=output_main_pertubed.decode('utf-8')
            nino_perturbed=pd.read_csv(nino_file,header=None)
            nino_perturbed=np.array(nino_perturbed)
            thermocline_east_perturbed=pd.read_csv('H_east_weekly.gdat',header=None)
            thermocline_west_perturbed=pd.read_csv('H_west_weekly.gdat',header=None)
            thermocline_east_perturbed=np.array(thermocline_east_perturbed)
            thermocline_west_perturbed=np.array(thermocline_west_perturbed)
            thermocline_east_perturbed=np.squeeze(thermocline_east_perturbed)
            thermocline_west_perturbed=np.squeeze(thermocline_west_perturbed)
            thermocline_east_not_perturbed=np.squeeze(thermocline_east_not_perturbed)
            thermocline_west_not_perturbed=np.squeeze(thermocline_west_not_perturbed)
            nino_not_perturbed=np.squeeze(nino_not_perturbed)
            nino_perturbed=np.squeeze(nino_perturbed)
            distance=np.sqrt(np.sum((nino_not_perturbed-nino_perturbed)**2))
            nino_perturbed_weekly=pd.read_csv("nino_weekly.gdat",header=None)
            nino_perturbed_weekly=np.array(nino_perturbed_weekly)
            nino_not_perturbed_weekly=np.squeeze(nino_not_perturbed_weekly)
            nino_perturbed_weekly=np.squeeze(nino_perturbed_weekly)

            thermocline_east_perturbed_H1=pd.read_csv('H_east_regridded_weekly.gdat',header=None)
            thermocline_west_perturbed_H1=pd.read_csv('H_west_regridded_weekly.gdat',header=None)
            thermocline_east_perturbed_H1=np.array(thermocline_east_perturbed_H1)
            thermocline_west_perturbed_H1=np.array(thermocline_west_perturbed_H1)
            thermocline_east_perturbed_H1=np.squeeze(thermocline_east_perturbed_H1)
            thermocline_west_perturbed_H1=np.squeeze(thermocline_west_perturbed_H1)
            thermocline_east_not_perturbed_H1=np.squeeze(thermocline_east_not_perturbed_H1)
            thermocline_west_not_perturbed_H1=np.squeeze(thermocline_west_not_perturbed_H1)

            print("**************************************************")
            print("\n")
            print("Perturbation amplitude:{}".format(math.sqrt((x[0]/w1)**2+(x[1]/w2)**2+(x[2]/w2)**2)))
            print("Perturbation Niño3:{}".format(x[0]))
            print("Perturbation ThW:{}".format(x[1]))
            print("Perturbation ThE:{}".format(x[2]))
            print("Distance:{}".format(distance))
            print("\n")
            print("**************************************************")
            os.chdir("../PythonCode")
            return -distance
    
        except Exception as e:
            # This will catch any exception and handle it
            print(f"An error occurred: {e}")
    
    def conditional_non_linear_perturbation_different_conditions(self,delta,
    initial_month_cnop,lead_time,drag,initial_year,end_year,
    configuration_file,main_file_field,nino_file,makefile_file,trials=10,iterations=12000.5,
    initial_save=11400):
        
        dictionary_results={'TE':[],'HW':[],'HE':[],
        'perturbation_amplitude':[],'distance':[],'rms':[],
        'correlation':[]}
        
        initial_month=12*initial_year+0.167+initial_month_cnop
        end_month=12*(end_year-1)+0.167+initial_month_cnop

        w1 = 2
        w2 = 50

        path_configuration="../ZebiakCaneFiles/configuration_files/"+configuration_file
        tfind=initial_month

        with open(main_file, 'r') as file:
            lines = file.readlines()
        
        lines[61]="\tOPEN(5,FILE='configuration_files/"+configuration_file+"',status='old')\n"
        lines[45]="      open(91,file='"+nino_file+"'\n"

        file.close()

        with open(main_file, 'w') as file:
            file.writelines(lines)
        
        file.close()

        with open(makefile_file, 'r') as file:
            lines = file.readlines()
        
        lines[8]="TARGET = "+main_file_field+"\n"
        lines[10]="SOURCES = {} akcalc.f bndary.f fft2c.f cforce.f constc.f initdat.f mloop.f nrdhist.f openfl.f setup.f setup2.f ssta.f tridag.f uhcalc.f uhinit.f ztmfc1.f\n".format(main_file)

        file.close()

        with open(makefile_file, 'w') as file:
            file.writelines(lines)
        
        file.close()

        os.chdir("../ZebiakCaneFiles")

        #os.system("make")
        command_make=["make", "-f", makefile_file]
        result_make=subprocess.run(command_make,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output_make=result_make.stderr
        output_make=output_make.decode('utf-8')
        os.chdir("../PythonCode")



        while(True):

            if(tfind>=end_month):
                break

            with open(path_configuration, 'r') as file:
                lines = file.readlines()
            
            lines[1]="./OutputEx/Drag{}/it{}in{}.out\n".format(drag,iterations,initial_save)
            lines[2]="./placeholder\n"
            lines[3]="NSTART =  2\n"
            lines[4]="TFIND  =  {}\n".format(tfind)
            lines[11]="TZERO  =  {}\n".format(0.5)
            lines[12]="TENDD  =  {}\n".format(tfind+lead_time)
            lines[53]="DRAG   =  {}\n".format(drag)
            lines[51]="NIC    =  0\n"
            lines[25]='NTAPE  =  0\n'
            file.close()

            with open(path_configuration, 'w') as file:
                file.writelines(lines)

            bounds = Bounds([-1000,-1000,-1000], [1000,1000,1000])
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
            
            first=True
            counter_failures=0
            
            for i,initial_condition in enumerate(initial_conditions):

                res=minimize(lambda x: self.objective_function_different_conditions(x,
                configuration_file=configuration_file,main_file_field=main_file_field,
                nino_file=nino_file),
                initial_condition,constraints=nc,
                options={'maxiter': 1250, 'tol': 1e-4},method="COBYLA")
                
                if(res.success==False):
                    print("optimization failed")
                    print("distance:{}".format(res.fun))
                    print("Perturbation amplitude:{}".format(math.sqrt((res.x[0]/w1)**2+(res.x[1]/w2)**2+(res.x[2]/w2)**2)))
                    counter_failures=counter_failures+1
                    continue

                if(first):
                    best_result=res
                    first=False
                else:
                    if(res.fun<=best_result.fun):
                        best_result=res

                list_initial_condistions.append([res.x[0],res.x[1],res.x[2]])
                list_final_results.append(res.fun)
                print("distance:{}".format(res.fun))
                print("Perturbation amplitude:{}".format(math.sqrt((res.x[0]/w1)**2+(res.x[1]/w2)**2+(res.x[2]/w2)**2)))

            print("number failures:{}".format(counter_failures))
            dictionary_results['TE'].append(best_result.x[0])
            dictionary_results['HW'].append(best_result.x[1])
            dictionary_results['HE'].append(best_result.x[2])
            dictionary_results['perturbation_amplitude'].append(math.sqrt((best_result.x[0]/w1)**2+(best_result.x[1]/w2)**2+(best_result.x[2]/w2)**2))
            dictionary_results['distance'].append(best_result.fun)   

            print("*********************************")
            print("\n")
            print("Maximal distance estimated Cobyla:{}".format(best_result.fun))
            print("Perturbation amplitude:{}".format(math.sqrt((best_result.x[0]/w1)**2+(best_result.x[1]/w2)**2+(best_result.x[2]/w2)**2)))
            print("Optimal Perturbation:{}".format(best_result.x))
            print("Initial moment:{}".format(tfind))
            print("\n")
            print("*********************************")

            rms,correlation=self.compute_rms_correlation_single_event(tfind,[best_result.x[0],
            best_result.x[1],best_result.x[2]],
            configuration_file=configuration_file,
            main_file_field=main_file_field,
            nino_file=nino_file)
            dictionary_results['rms'].append(rms)
            dictionary_results['correlation'].append(correlation)
            tfind=tfind+12

            print(best_result)
        
        os.chdir("../PythonCode")
        current_directory = os.getcwd()
        print(current_directory)
        directory = Path("./resultsCNOPsZebiakCane/Drag{}".format(drag))
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        
        json_file_name="CNOPsIm{}Delta{}Lead{}Interval[{},{}]Attemps{}.json".format(initial_month_cnop,
        delta,lead_time,initial_year,end_year,trials)
        file=directory.joinpath(json_file_name)
        with open(file, 'w') as json_file:
            json.dump(dictionary_results, json_file)

    def compute_rms_correlation_single_event(self,tfind,x,
    configuration_file,main_file_field,nino_file):

        w1=2
        w2=50

        os.chdir("../ZebiakCaneFiles")

        path_configuration="./configuration_files/"+configuration_file

        with open(path_configuration, 'r') as file:
            lines = file.readlines()
            lines[54]="PF     =  0.0\n"
            lines[5]="P_NINO =  0.0\n"
            lines[6]="P_THW  =  0.0\n"
            lines[7]="P_THE  =  0.0\n"
        
        file.close()
        
        with open(path_configuration, 'w') as file:
            file.writelines(lines)

        #os.system("./"+main_file_field+" > /dev/null 2>&1") 
        #os.system("./"+main_file_field)

        command_main=["./"+main_file_field]
        result_main=subprocess.run(command_main,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        err_main=result_main.stderr
        err_main=err_main.decode('utf-8')
        output_main=result_main.stdout
        output_main=output_main.decode('utf-8')

        nino_not_perturbed=pd.read_csv(nino_file,header=None)
        nino_not_perturbed=np.array(nino_not_perturbed)
        nino_not_perturbed_weekly=pd.read_csv("nino_weekly.gdat",header=None)
        nino_not_perturbed_weekly=np.array(nino_not_perturbed_weekly)
        thermocline_east_not_perturbed=pd.read_csv('H_east_weekly.gdat',header=None)
        thermocline_west_not_perturbed=pd.read_csv('H_west_weekly.gdat',header=None)
        thermocline_east_not_perturbed=np.array(thermocline_east_not_perturbed)
        thermocline_west_not_perturbed=np.array(thermocline_west_not_perturbed)

        thermocline_east_not_perturbed_H1=pd.read_csv('H_east_regridded_weekly.gdat',header=None)
        thermocline_west_not_perturbed_H1=pd.read_csv('H_west_regridded_weekly.gdat',header=None)
        thermocline_east_not_perturbed_H1=np.array(thermocline_east_not_perturbed_H1)
        thermocline_west_not_perturbed_H1=np.array(thermocline_west_not_perturbed_H1)

        file.close()

        with open(path_configuration, 'r') as file:
            lines = file.readlines()
            lines[54]="PF     =  1.0\n"
            lines[5]="P_NINO =  {}\n".format(x[0])
            lines[6]="P_THW  =  {}\n".format(x[1])
            lines[7]="P_THE  =  {}\n".format(x[2])

        with open(path_configuration, 'w') as file:
            file.writelines(lines)
        
        file.close()

        #os.system("./"+main_file_field+" > /dev/null 2>&1") 
        #os.system("./"+main_file_field)
        command_main_pertubed=["./"+main_file_field]
        result_main_pertubed=subprocess.run(command_main_pertubed,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output_main_pertubed=result_main_pertubed.stdout
        output_main_pertubed=output_main_pertubed.decode('utf-8')
        nino_perturbed=pd.read_csv(nino_file,header=None)
        nino_perturbed=np.array(nino_perturbed)
        thermocline_east_perturbed=pd.read_csv('H_east_weekly.gdat',header=None)
        thermocline_west_perturbed=pd.read_csv('H_west_weekly.gdat',header=None)
        thermocline_east_perturbed=np.array(thermocline_east_perturbed)
        thermocline_west_perturbed=np.array(thermocline_west_perturbed)
        thermocline_east_perturbed=np.squeeze(thermocline_east_perturbed)
        thermocline_west_perturbed=np.squeeze(thermocline_west_perturbed)
        thermocline_east_not_perturbed=np.squeeze(thermocline_east_not_perturbed)
        thermocline_west_not_perturbed=np.squeeze(thermocline_west_not_perturbed)
        nino_not_perturbed=np.squeeze(nino_not_perturbed)
        nino_perturbed=np.squeeze(nino_perturbed)
        distance=np.sqrt(np.sum((nino_not_perturbed-nino_perturbed)**2))
        nino_perturbed_weekly=pd.read_csv("nino_weekly.gdat",header=None)
        nino_perturbed_weekly=np.array(nino_perturbed_weekly)
        nino_not_perturbed_weekly=np.squeeze(nino_not_perturbed_weekly)
        nino_perturbed_weekly=np.squeeze(nino_perturbed_weekly)

        thermocline_east_perturbed_H1=pd.read_csv('H_east_regridded_weekly.gdat',header=None)
        thermocline_west_perturbed_H1=pd.read_csv('H_west_regridded_weekly.gdat',header=None)
        thermocline_east_perturbed_H1=np.array(thermocline_east_perturbed_H1)
        thermocline_west_perturbed_H1=np.array(thermocline_west_perturbed_H1)
        thermocline_east_perturbed_H1=np.squeeze(thermocline_east_perturbed_H1)
        thermocline_west_perturbed_H1=np.squeeze(thermocline_west_perturbed_H1)
        thermocline_east_not_perturbed_H1=np.squeeze(thermocline_east_not_perturbed_H1)
        thermocline_west_not_perturbed_H1=np.squeeze(thermocline_west_not_perturbed_H1)


        rms=np.sqrt(np.mean((nino_not_perturbed-nino_perturbed)**2))  
        correlation=pearsonr(nino_not_perturbed,nino_perturbed)
        correlation=correlation[0]

        print("**********************************")
        print("\n")
        print("Values saved in dictionary:")
        print("distance:{}".format(distance))
        print("amplitude_perturbation:{}".format(math.sqrt(((x[0]/w1)**2)+((x[1]/w2)**2)+((x[2]/w2)**2))))
        print("correlation:{}".format(correlation))
        print("rms:{}".format(rms))
        print("Optimal Perturbation:{}".format(x))
        print("Initial moment:{}".format(tfind))
        print("\n")
        print("**********************************")
        return rms,correlation

#function to compute the CNOPs for the Zebiak and Cane model the checkpoint to reinitialize the model 
#for the 2 timeseries considered are contained in the directory ZebiakCaneFiles/OutputEx
delta=0.05
drag=0.77
drag_str=str(drag)[2]
trials=10

if(len(str(delta))==3):
    delta_str=str(delta)[2]
else:
    delta_str=str(delta)[2]+str(delta)[3]
drag_str=str(drag)[2]

interval=[955,1000]
for initial_month in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for lead_time_arg in [3,6,9]:    
        configuration_file="fctest{}{}{}{}.data".format(initial_month,drag_str,delta_str,lead_time_arg)
        main_file_field="ZC{}{}{}{}_main".format(initial_month,drag_str,delta_str,lead_time_arg)
        os.system("cp ../ZebiakCaneFiles/configuration_files/fctest.data ../ZebiakCaneFiles/configuration_files/"+configuration_file)

        nino_file="nino{}{}{}{}.gdat".format(initial_month,drag_str,delta_str,lead_time_arg)
        main_file="main{}{}{}{}.f".format(initial_month,drag_str,delta_str,lead_time_arg)
        makefile_file="makefile{}{}{}{}".format(initial_month,drag_str,delta_str,lead_time_arg)

        os.system("cp ../ZebiakCaneFiles/main.f ../ZebiakCaneFiles/"+main_file)
        os.system("cp ../ZebiakCaneFiles/makefile ../ZebiakCaneFiles/"+makefile_file)

        print("********************")
        print("\n")
        print("Delta:{}".format(delta))
        print("Drag:{}".format(drag))
        print("Initial month:{}".format(initial_month))
        print("lead time:{}".format(lead_time_arg))
        print("Start year{}".format(interval[0]))
        print("End year{}".format(interval[1]))
        print("\n")
        print("*********************")

        print(nino_file)
        print(main_file)
        print(configuration_file)
        print(main_file_field)
        print(makefile_file)
        os.chdir("../ZebiakCaneFiles")
        solver=CNOPSolverZebiak()
        solver.conditional_non_linear_perturbation_different_conditions(delta,
        initial_month,lead_time_arg,drag,
        initial_year=interval[0],end_year=interval[1],
        configuration_file=configuration_file,main_file_field=main_file_field,
        nino_file=nino_file,makefile_file=makefile_file,trials=trials)