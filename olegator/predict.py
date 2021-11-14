import pandas as pd
import numpy as np
from scipy.stats import randint, norm

def make_bootstrap(X):
    Y = randint.rvs(low = 0, high = len(X),  size = len(X))
    
    return [X[i] for i in Y]

#для среднего бутстреп
def mean_bt(X):
    B = 500
    distr = []
    for i in range(B):
        distr.append(np.mean(make_bootstrap(X)))

    D = np.var(distr)
    se = D**0.5

    alpha = 0.05
    z = norm.ppf(1-alpha/2)

    #дов интервал 0,95
    return np.mean(X) + z*se

#бутстреп для 0,95 квантиля
def quantile_bt(X):
    B = 500
    distr = []
    for i in range(B):
        distr.append(np.quantile(make_bootstrap(X), 0.95))

    D = np.var(distr)
    se = D**0.5

    alpha = 0.05
    z = norm.ppf(1-alpha/2)

    #дов интервал
    return np.quantile(X, 0.95) + z*se

def predict(data):
    df1 = data[data.iloc[:, 9:19].sum(axis=1) == 0]
    df2 = data[data.iloc[:, 9:19].sum(axis=1) != 0]
    df1 ['predict'] = np.zeros(len(df1))
    ind = pd.read_csv('/Users/user/hackathon/olegator/ind.csv')
    df2 = df2.merge(ind,on=['prod_norm_name', 'prod_form_norm_name', 'code', 'prod_pack_1_2',
       'prod_pack_1_size', 'origin', 'prod_d_norm_name'],how='left')



    df_catboost = df2[df2['const'] == 1]
    col_to_test_X = ['prod_norm_name', 'prod_form_norm_name', 'code', 'prod_pack_1_2',
                  'prod_pack_1_size', 'origin', 'ЖНВЛП', 'Ковид', 
                  '2.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0',
                  '4_conf', '5_conf', '6_conf', '7_conf', '8_conf', '9_conf', '10_conf',
                  '4_covd', '5_covd', '6_covd', '7_covd', '8_covd', '9_covd', '10_covd', 
                  '4_covi', '5_covi', '6_covi', '7_covi', '8_covi', '9_covi', '10_covi',
                  '4_caseF', '5_caseF', '6_caseF','7_caseF', '8_caseF', '9_caseF', '10_caseF',
                  '6.0w_2', '7.0w_3', '8.0w_4', '9.0w_5', '10.0w_6']

    # колонки которые знает катбуст
    feature_col_X = ['prod_norm_name', 'prod_form_norm_name', 'code', 'prod_pack_1_2',
                'prod_pack_1_size', 'origin', 'ЖНВЛП', 'Ковид', 
                '1_mounth', '2_mounth', '3_mounth', '4_mounth', '5_mounth', '6_mounth', '7_mounth', 
                '1_conf', '2_conf', '3_conf', '4_conf', '5_conf', '6_conf', '7_conf',
                '1_covd', '2_covd', '3_covd', '4_covd', '5_covd', '6_covd', '7_covd', 
                '1_covi', '2_covi', '3_covi', '4_covi', '5_covi', '6_covi', '7_covi',
                '1_caseF', '2_caseF', '3_caseF','4_caseF', '5_caseF', '6_caseF', '7_caseF',
                '1_w', '2_w', '3_w', '4_w', '5_w']
    float_features = ['prod_pack_1_size', '1_mounth', '2_mounth', '3_mounth', '4_mounth', '5_mounth', '6_mounth', '7_mounth', 
                '1_conf', '2_conf', '3_conf', '4_conf', '5_conf', '6_conf', '7_conf',
                '1_covd', '2_covd', '3_covd', '4_covd', '5_covd', '6_covd', '7_covd', 
                '1_covi', '2_covi', '3_covi', '4_covi', '5_covi', '6_covi', '7_covi',
                '1_caseF', '2_caseF', '3_caseF','4_caseF', '5_caseF', '6_caseF', '7_caseF',
                '1_w', '2_w', '3_w', '4_w', '5_w']
    cat_features = ['prod_norm_name', 'code', 'prod_pack_1_2', 
                'origin', 'ЖНВЛП', 'Ковид', 'prod_form_norm_name']

    # приводим prod_form_norm_name к кластеру
    mapping_name = pd.DataFrame(df_catboost.prod_form_norm_name.unique()).reset_index().set_index(0).to_dict()['index']
    df_catboost.prod_form_norm_name = df_catboost.prod_form_norm_name.map(mapping_name)

    # выбираем только нужные колонки
    test_X = df_catboost.loc[:, col_to_test_X]

    # переменовываем их для катбуста
    test_X.columns = feature_col_X

    # логарифмируем целевую переменную
    test_X.loc[:, ['1_mounth', '2_mounth', '3_mounth', '4_mounth', '5_mounth', '6_mounth', '7_mounth']] = (test_X.loc[:, ['1_mounth', '2_mounth', '3_mounth', '4_mounth', '5_mounth', '6_mounth', '7_mounth']] + 1).apply(np.log)

    # приводим к нужному виду
    test_X['origin'] = test_X['origin'].astype(int)
    test_X['prod_pack_1_2'] = test_X['prod_pack_1_2'].astype(int)
    test_X['code'] = test_X['code'].astype(int)
    test_X['prod_norm_name'] = test_X['prod_norm_name'].astype(int)
    from_file = CatBoostRegressor()
    from_file.load_model('/Users/user/hackathon/olegator/model.cbm')
    df_catboost['predict'] 


    df_stat = df2[df2['const'] != 1]
    df_stat_jn = df_stat[df_stat['ЖНВЛП'] == 1]
    df_stat_not_jn = df_stat[df_stat['ЖНВЛП'] == 0]
    df_jn = np.apply_along_axis(lambda x: quantile_bt(x), axis=1, arr=df_stat_jn.iloc[:,9:19].values)
    df_not_jn = np.apply_along_axis(lambda x: mean_bt(x), axis=1, arr=df_stat_not_jn.iloc[:,9:19].values)
    df_stat_jn['predict'] = df_jn
    df_stat_not_jn['predict'] = df_not_jn
    return pd.concat([df1,df_catboost,df_stat_jn,df_stat_not_jn], axis=0).drop('const',axis=1)

if __name__ == "__main__":
    df = pd.read_csv('test_predict.csv')
    print(predict(df)['predict'])