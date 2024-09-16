def warn(*args, **kwargs):
    pass
from itertools import groupby
from typing import final
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)

df1 = pd.read_csv("Bangalore House Prices Prediction/dataset/bengaluru_house_prices.csv")
#print(df1)

#print(df1.groupby('area_type')['area_type'].agg('count'))

df2 = df1.drop(['area_type','society','balcony','availability'], axis ='columns')

df3 = df2.dropna()
df3.isnull().sum()


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)]


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+ float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    

df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)


location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x : 'other' if x  in location_stats_less_than_10 else x)
#print(df5.head(20)) 

df6 = df5[~(df5.total_sqft/df5.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
#print(df7.shape)

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.bhk ==2)]
    bhk3 = df[(df.location == location) & (df.bhk ==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price, color='blue', label= '2 BHK')
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")
plot_scatter_chart(df7,"Hebbal")
#plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
#print(df8.shape)

df9 = df8[df8.bath < df8.bhk +2]
#print(df9.shape)

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
#print(df10.head(3))

#print(df10.location.value_counts())

dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis= 'columns')], axis= 'columns')

df12 = df11.drop('location', axis= 'columns')

x = df12.drop('price', axis= 'columns')
#print(x)


y = df12.price

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest =  train_test_split(x , y , test_size= 0.2, random_state= 10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(xtrain, ytrain)
#print(lr_clf.score(xtest, ytest))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits= 5 , test_size= 0.2, random_state= 0)
#print(cross_val_score(LinearRegression() , x , y , cv = cv))


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

#print(find_best_model_using_gridsearchcv(x,y))

def predict_price(location, sqft, bath, bhk):
    global x
    global lr_clf
    
    # Find the index of the location in the DataFrame columns
    loc_index = np.where(x.columns == location)[0][0]

    # Initialize a zero array with the same number of columns as the DataFrame
    input_data = np.zeros(len(x.columns))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk
    if loc_index >= 0:
        input_data[loc_index] = 1

    # Return the prediction
    return lr_clf.predict([input_data])[0] # type: ignore

# Assuming x and lr_clf are defined globally
#print(predict_price('Indira Nagar',1000, 3, 3))

import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))