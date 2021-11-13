import pandas as pd

def clean_data(input_data):
    df = pd.read_csv(input_data)
    df.drop(columns=['year_'], inplace=True) # Этот слобец не нужен
    df.drop(columns=['name'], inplace=True) # Этот слобец не нужен
    df['prod_norm_name'] = df['prod_norm_name'].astype(str).apply(lambda x: x.split()[-1][1:]).astype(int) # Приводим к числовому виду
    df['ЖНВЛП'] = df['ЖНВЛП'].apply(lambda x: int(x == "Да")) # Создаем числовые признаки
    df['Ковид'] = df['Ковид'].apply(lambda x: int(x == "Да")) # Создаем числовые признаки
    df['code'] = df['code'].astype(int)
    meta_data = pd.read_csv('meta_data_regions.csv')
    df = df.merge(meta_data,on='code')
    return df

print(clean_data('test.csv').head(5))
