import numpy as np
import pandas as pd
import os
dataset_patch = os.path.join('..','dataset_lab')


df_1 = os.path.join(dataset_patch,'7-Perfil_Compostagem(meio).csv')
df_1 = pd.read_csv(df_1) 
df_1['Class'] = 'Compostagem(meio)'
df_1 = df_1.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

#rename the columns
df_1 = df_1.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})

df_2 = os.path.join(dataset_patch,'12-Perfil_Compostagem(inicio).csv')
df_2 = pd.read_csv(df_2)
df_2['Class'] = 'Compostagem(inicio)'
df_2 = df_2.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

#rename the columns
df_2 = df_2.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})

    
df_3 = os.path.join(dataset_patch,'14-Perfil_Compostagem(final).csv')
df_3 = pd.read_csv(df_3)
df_3['Class'] = 'Compostagem(final)'
df_3 = df_3.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

#rename the columns
df_3 = df_3.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})

df_4 = os.path.join(dataset_patch,'20-Perfil_Racao(frango).csv')
df_4 = pd.read_csv(df_4)
df_4['Class'] = 'Racao(frango)'
df_4 = df_4.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

   
df_4 = df_4.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})


df_5 = os.path.join(dataset_patch,'17-Perfil_PauAlho(flor).csv')  
df_5 = pd.read_csv(df_5)
df_5['Class'] = 'Pau de alho(flor)'
df_5 = df_5.drop(['D20','D21','D22','D23','D24','D25'], axis=1)

df_5 = df_5.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})


#now concatenate the dataframes and save as new csv
df = pd.concat([df_1,df_2,df_3,df_4,df_5], axis=0)

#save as new csv
df_concat = os.path.join(dataset_patch,'concatV6.csv')
df.to_csv(df_concat, index=False)
