import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import os


rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 30
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 30
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"
rcParams['font.size']=30

class ZebiakCaneModel:

    def __init__(self,data_directory,drag,transitional_years,train_years,
    validation_interval,variables,normalize):
        
        
        self.dictionary_variables={"Nino3":1,"ThE":2,"ThW":3,"Wind":4}
        self.annual_period=12

        self.time_line=np.linspace(0.5,12000.5,36001)
        self.time_line=np.round(self.time_line,4)
        self.time_line_mean=np.int8(self.time_line%12)
        
        nino3=pd.read_csv(data_directory+"/Drag"+str(drag)+'/nino_n_components_1_iterations12000.5_noise_amplitude1.0_weekly.gdat',header=None)
        thermocline_east=pd.read_csv(data_directory+"/Drag"+str(drag)+'/H_east_n_components_1_iterations12000.5_noise_amplitude1.0_weekly.gdat',header=None)
        thermocline_west=pd.read_csv(data_directory+"/Drag"+str(drag)+'/H_west_regridded_n_components_1_iterations12000.5_noise_amplitude1.0_weekly.gdat',header=None)
        wind_zonal=pd.read_csv(data_directory+"/Drag"+str(drag)+'/zonal_wind_n_components_1_iterations12000.5_noise_amplitude1.0_weekly.gdat',header=None)
        cycle=np.sin((2*np.pi/self.annual_period)*self.time_line) 
        
        nino3=np.array(nino3)
        nino3=np.squeeze(nino3)
        thermocline_east=np.array(thermocline_east)
        thermocline_east=np.squeeze(thermocline_east)
        thermocline_west=np.array(thermocline_west)
        thermocline_west=np.squeeze(thermocline_west)
        wind_zonal=np.array(wind_zonal)
        wind_zonal=np.squeeze(wind_zonal)
        cycle=cycle[np.newaxis,:]

        self.drag=drag
        self.transitional_years=transitional_years
        self.train_years=train_years
        self.validation_interval=validation_interval
        self.normalize=normalize
        self.original_std_nino=np.std(nino3)

        if(normalize):
            nino3=nino3/2
            thermocline_east=thermocline_east/50
            thermocline_west=thermocline_west/50

        nino3=nino3[np.newaxis,:]
        thermocline_east=thermocline_east[np.newaxis,:]
        thermocline_west=thermocline_west[np.newaxis,:]
        wind_zonal=wind_zonal[np.newaxis,:]

        self.data=np.concatenate((cycle,nino3,thermocline_east,thermocline_west,wind_zonal),axis=0)
        indexes_variables=[0]+[self.dictionary_variables[variable] for variable in variables]

        print("#################################")
        print("Shape dataset ZC data:{}".format(self.data.shape))
        print("#################################")

        self.data=self.data[indexes_variables,:]

        train_indexes=[np.where(self.time_line>=transitional_years*self.annual_period)[0][0],
        np.where(self.time_line>=(transitional_years+train_years)*self.annual_period)[0][0]]

        validation_indexes=[np.where(self.time_line>=validation_interval[0]*self.annual_period)[0][0],
        np.where(self.time_line>=validation_interval[1]*self.annual_period)[0][0]]

        self.X_train=self.data[:,train_indexes[0]:train_indexes[1]]
        self.X_validation=self.data[:,validation_indexes[0]:validation_indexes[1]]
        self.time_line_train=self.time_line[train_indexes[0]:train_indexes[1]]
        self.time_line_validation=self.time_line[validation_indexes[0]:validation_indexes[1]]

        print("###################################")
        print("Shape train dataset:{}".format(self.X_train.shape))
        print("Shape validation dataset:{}".format(self.X_validation.shape))
        print("###################################")
    
    def plot(self,variable,start_year,end_year):

        if(variable=="nino"):
            data_tmp=self.data[1,:]
            label="NINO3 [°C]"

        elif(variable=="H_east"):
            data_tmp=self.data[2,:]
            label="Thermocline anomalies east [m]"

        elif(variable=="H_west"):
            data_tmp=self.data[3,:]
            label="Thermocline anomalies west [m]"
    
        elif(variable=="wind_zonal"):
            data_tmp=self.data[4,:]
            label="Zonal wind anomalies [m/s]"

        fig=plt.figure(figsize=(24,12))
        plt.plot(self.time_line[36*start_year:36*end_year]/12,data_tmp[36*start_year:36*end_year],
        linewidth=3,c="k")
        plt.axvspan(start_year, (self.transitional_years+self.train_years), 
        color='#d4f4dd', alpha=0.8, label='Training')    # Training section
        plt.axvspan(self.validation_interval[0], self.validation_interval[1], 
        color='#d9eaf7', alpha=0.8, label='Validation')  # Validation section
        plt.axvspan(self.validation_interval[1], end_year, color='#fde2e4', 
        alpha=0.8, label='Test') 
        plt.legend()
        plt.xlabel("time (years)")
        plt.ylabel(label)
        plt.savefig(f'./imagesPaperErrorGrowth/Nino3Drag{self.drag}.png',bbox_inches='tight')
        plt.show()

#plot the training validation and test dataset
#drag=0.77
#zc=ZebiakCaneModel(data_directory="../ZebiakCaneFiles/NoisyTrajectories_for_experiments_paper",
#drag=drag,transitional_years=300,
#train_years=300,validation_interval=[600,800],variables=["Nino3","ThE","ThW","Wind"],normalize=False)
#zc.plot(variable="nino",start_year=300,end_year=1000)