# -*- coding: utf-8 -*-

"""dataManagement.py
   Author: Jorge Cardenas

   1. Simulation data logging in CSV Files
   2. Simulation data Retrieval

   Future developments:
   1. Local or remote data storage
"""

import csv
from .global_ import *
import os
import pandas as pd
import os.path as path

DBNAME = "db.csv"

DB_LOCATION = direccion_archivos+"/output/"+DBNAME
DB_DATA_TO_PLOT= direccion_archivos+"/output/"
class DBManager:

   file=None
   df = None
   def __init__(self, simulation_ID ):
        self.simulation_ID=simulation_ID
   
   def load_df(self):

      if path.exists('output/output.csv'):
         self.df = pd.read_csv('output/output.csv', header=0)
      else:
            
         column_names = ["sim_id", "created_at", "sim_setup","sim_results", "pbest","gbest","best_particle_id","best_particle","iteration"]
         
         self.df = pd.DataFrame(columns = column_names)

         self.df['sim_id']=self.df['sim_id'].astype( 'object')
         self.df['created_at']=self.df['created_at'].astype( 'datetime64[ms]') ##ojo con esto actualizar 
         self.df['sim_setup']=self.df['sim_setup'].astype( 'object')
         self.df['pbest']=self.df['pbest'].astype( 'float64')
         self.df['gbest']=self.df['gbest'].astype( 'float64')
         self.df['best_particle_id']=self.df['best_particle_id'].astype( 'int64')
         self.df['iteration']=self.df['iteration'].astype( 'int64')


   def save_data():
      pass
   def save_data_to_plot(self,data_to_plot,iteration,particle_id):
      
      df=pd.DataFrame.from_dict(data_to_plot,orient='index').transpose()
      df.to_csv('output/'+str(self.simulation_ID)+"/files/"+r"Derivative_"+str(iteration)+"_"+str(particle_id)+".csv", index=False,sep=',')


   def fill_df(self,data_struct):
      output = pd.DataFrame()
      output = pd.concat(output, data_struct, ignore_index= True)
      
      df = pd.concat([self.df, output])
      
      df.to_csv('output/output.csv', index=False,sep=',')
      
  
      
     


