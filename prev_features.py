import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.ensemble import RandomForestRegressor


# Import and clean data of 2019
df = pd.read_csv('testData_2019_Central.csv')
df['Date'] = pd.to_datetime (df['Date'], format='%Y-%m-%d %H:%M:%S')
df['Hour'] = df['Date'].dt.hour
df['WeekEnd'] = df['Date'].map(lambda x: 1 if x.weekday() < 5 else 0)
df['Power-1']=df['Central (kWh)'].shift(1)
df=df.dropna()

method=['Filter_regression','Filter_mutual_info','Embedded']

# function that, for a liste of given features and a methos, gives the score of each feature in a list
def plot_features_scores(list_feat=['Power-1','Hour','temp_C','solarRad_W/m2','WeekEnd'],
                         data=df,
                         method='Filter_mutual_info'):
    list_feat1=['Date','Central (kWh)']+list_feat
    data=data.filter(items=list_feat1)
    Z=data.values
    Y=Z[:,1]
    X=Z[:,range(2,len(Z[1]))]
    if method == 'Filter_regression':
        features=SelectKBest(k='all',score_func=f_regression)
        fit=features.fit(X,Y)
        return fit.scores_

    elif method == 'Filter_mutual_info': 
        features=SelectKBest(k='all',score_func=mutual_info_regression)
        fit=features.fit(X,Y) 
        return fit.scores_
    
    elif method == 'Embedded':
        model = RandomForestRegressor()
        model.fit(X, Y)
        return model.feature_importances_
