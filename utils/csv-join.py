import numpy as np
import pandas as pd
import os
#read 3 csv files 
dataset_patch = os.path.join('..','dataset_new')
# df_1_atual_df = os.path.join(dataset_patch,'Perfil_Compostagem(final)-concatV2.csv')
# df_1 = pd.read_csv(df_1_atual_df)
# df_2_atual_df = os.path.join(dataset_patch,'Perfil_Compostagem(meio)-concatV2.csv')
# df_2 = pd.read_csv(df_2_atual_df)
# #read more 4 csv files
# df_3_atual_df = os.path.join(dataset_patch,'Perfil_Compostagem(inicio)-concatV2.csv')
# df_3 = pd.read_csv(df_3_atual_df)
df_4_atual_df = os.path.join(dataset_patch,'Perfil-Racao-Alho.csv')
df_4 = pd.read_csv(df_4_atual_df)
df_5_atual_df = os.path.join(dataset_patch,'Perfil-Geral-Compostagem-v2.csv')
df_5 = pd.read_csv(df_5_atual_df)

 
df = pd.concat([df_4,df_5], axis=0)
df_concat = os.path.join(dataset_patch,'concatV5.csv')
df.to_csv(df_concat, index=False)

