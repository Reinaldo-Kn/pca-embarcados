import numpy as np
import pandas as pd
import os
dataset_patch = os.path.join('..','dataset_new')
atual_df = os.path.join(dataset_patch,'concatV4-names.csv')
df = pd.read_csv(atual_df)

#rename the columns
df = df.rename(columns={
    'VOC (ppb)': 'D0', 
    'VOC (ug m-3)': 'D1', 
    'VOC (Temperatura)': 'D2', 
    'VOC (Umidade)': 'D3',
    'H2S (ppb)': 'D4', 
    'H2S (ug m-3)': 'D5', 
    'H2S (Temperatura)': 'D6', 
    'H2S (Umidade)': 'D7',
    'SO2 (ppb)': 'D8', 
    'SO2 (ug m-3)': 'D9', 
    'SO2 (Temperatura)': 'D10', 
    'SO2 (Umidade)': 'D11',
    'NH3 (ppb)': 'D12', 
    'NH3 (ug m-3)': 'D13', 
    'NH3 (Temperatura)': 'D14', 
    'NH3 (Umidade)': 'D15',
    'CH3SH (ppb)': 'D16', 
    'CH3SH (ug m-3)': 'D17', 
    'CH3SH (Temperatura)': 'D18', 
    'CH3SH (Umidade)': 'D19',
    'BME280 - TEMPERATURA': 'D26', 
    'BME280 - PRESSAO': 'D27', 
    'BME280 - UMIDADE': 'D28'
})

new_df = os.path.join(dataset_patch,'dirty-concatV4.csv')
#save as new csv
df.to_csv(new_df, index=False)