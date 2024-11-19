import numpy as np
import pandas as pd

df = pd.read_csv('./dataset/compostagem.csv')

# create a new column 
df['Class'] = 'Compostagem'


#delete the unused columns
df = df.drop(['D0', 'D3','D4','D7','D8','D11','D12','D15','D16','D19','D23','D24',
              'D25','D26','D27','D28','D30','D31','D33','D34'], axis=1)

#rename the columns
df = df.rename(columns={'D1': 'H2S(WE)', 
                        'D2': 'H2S(AE)',
                        'D5': 'SO2(WE)',
                        'D6': 'SO2(AE)',
                        'D9': 'CO(WE)',
                        'D10': 'CO(AE)',
                        'D13': 'VOC(WE)',
                        'D14': 'VOC(AE)',
                        'D17': 'NH3(WE)',
                        'D18': 'NH3(AE)',
                        'D20': 'MICS6814(CO)',
                        'D21': 'MICS6814(NH3)',
                        'D22': 'MICS6814(NO2)',
                        'D29': 'SEMEATECH(NH3)',
                        'D32': 'SEMEATECH(CH3SH)',
})


#save as new csv
df.to_csv('./dataset/compostagem_adjust.csv', index=False)