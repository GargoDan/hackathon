import pandas as pd
import os
import json
import numpy as np
from predict import predict

def test_function():
    print('vse norm')

def rolling_window(df_series, window_size=3):
    df_list = []
    for i in range(df_series.shape[0]):
        window_f = pd.DataFrame(df_series.iloc[i, :].shift(1).rolling(min_periods=1, 
        center=False, window=window_size).mean().iloc[window_size:]).T
        window_f.columns = [str(x) + f'w_{i}' for i, x in enumerate(window_f.columns)]
        df_list.append(window_f)
    df_to_return = pd.concat(df_list, axis=0)
    return df_to_return

def clean_data(input_data):
    df = pd.read_csv(input_data)
    df.drop(columns=['year_'], inplace=True) # Этот слобец не нужен
    df.drop(columns=['name'], inplace=True) # Этот слобец не нужен
    df['prod_norm_name'] = df['prod_norm_name'].astype(str).apply(lambda x: x.split()[-1][1:]).astype(int) # Приводим к числовому виду
    df['ЖНВЛП'] = df['ЖНВЛП'].apply(lambda x: int(x == "Да")) # Создаем числовые признаки
    df['Ковид'] = df['Ковид'].apply(lambda x: int(x == "Да")) # Создаем числовые признаки
    df['code'] = df['code'].astype(int) # Это целые числа
    df = df.fillna(0) 
    df = pd.pivot_table(df, index=['prod_norm_name', 'prod_form_norm_name', "code", 'prod_pack_1_2', 'prod_pack_1_size',
     'origin', 'ЖНВЛП', 'Ковид', 'prod_d_norm_name'], values='Count_3', columns='month_', aggfunc=max)
    df.to_csv('tmp.csv')
    df = pd.read_csv('tmp.csv')
    os.remove('tmp.csv')
    meta_data = pd.read_csv('meta_data_regions.csv')
    df = df.merge(meta_data,on='code')
    wind = rolling_window(df.iloc[:, 9:19])
    df = pd.concat([df,wind], axis=1)
    covi = pd.read_csv('covid_Incident_Rate.csv')
    covi.columns = ['code', '1_covi','2_covi','3_covi','4_covi','5_covi','6_covi','7_covi','8_covi','9_covi','10_covi']
    covd = pd.read_csv('covid_death.csv')
    covd.columns = ['code', '1_covd','2_covd','3_covd','4_covd','5_covd','6_covd','7_covd','8_covd','9_covd','10_covd']
    caseF = pd.read_csv('ovid_Case_Fatality_Ratio.csv')
    caseF.columns = ['code', '1_caseF','2_caseF','3_caseF','4_caseF','5_caseF','6_caseF',
    '7_caseF','8_caseF','9_caseF','10_caseF']
    conf = pd.read_csv('covid_confirmed.csv')
    conf.columns = ['code', '1_conf','2_conf','3_conf','4_conf','5_conf','6_conf','7_conf','8_conf','9_conf','10_conf']
    df = df.merge(conf,on='code').merge(caseF,on='code').merge(covd,on='code').merge(covi,on='code')
    return df

def format_output(data, path):
    with open('dict.json') as f:
        d = json.load(f)
    regions = pd.DataFrame(d.items())
    regions.columns = ['Region', 'code']
    output_columns = ['prod_norm_name', 'prod_form_norm_name', 'code', 'prod_pack_1_2',
       'prod_pack_1_size', 'origin', 'ЖНВЛП', 'Ковид','prod_d_norm_name', 'predict']
    regions.merge(data[output_columns],on='code').to_csv(path,index=False)

def backend(input_data):
    return format_output(predict(clean_data(input_data)), input_data)
if __name__ == "__main__":
    clean_data('test.csv')
