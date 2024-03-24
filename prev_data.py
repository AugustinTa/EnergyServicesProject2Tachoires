import pandas as pd
from sklearn import  metrics
import numpy as np
from dash import html

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import  linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Import and clean data of 2017-2018-2019
df_raw_meteo = pd.read_csv('IST_meteo_data_2017_2018_2019.csv') # loads a csv file into a dataframe
df_raw_bat2018 = pd.read_csv('IST_Central_Pav_2018.csv') # loads a csv file into a dataframe
df_raw_bat2017 = pd.read_csv('IST_Central_Pav_2017.csv') # loads a csv file into a dataframe

df_raw_meteo = df_raw_meteo.rename(columns={'yyyy-mm-dd hh:mm:ss': 'Date'})
df_raw_meteo['Date'] = pd.to_datetime(df_raw_meteo['Date'], format='%Y-%m-%d %H:%M:%S')
df_raw_meteo = df_raw_meteo.set_index ('Date', drop = True)
df_raw_meteo=df_raw_meteo.resample('H').mean()

df_raw_bat=pd.concat([df_raw_bat2017, df_raw_bat2018], axis=0)
df_raw_bat = df_raw_bat.rename(columns={'Date_start': 'Date'})
df_raw_bat['Date'] = pd.to_datetime(df_raw_bat['Date'], format='%d-%m-%Y %H:%M')
df_raw_bat['Hour'] = df_raw_bat['Date'].dt.hour
df_raw_bat.tail()
df_raw_bat = df_raw_bat.set_index ('Date', drop = True)

df_merged = pd.merge(df_raw_bat, df_raw_meteo, on='Date', how='inner')
df_merged.sort_values('Date', inplace=True)

df_merged['WeekEnd'] = df_merged.index.map(lambda x: 1 if x.weekday() < 5 else 0)
df_merged['Power-1']=df_merged['Power_kW'].shift(1)
df_train=df_merged.dropna()


df = pd.read_csv('testData_2019_Central.csv')
df['Date'] = pd.to_datetime (df['Date'], format='%Y-%m-%d %H:%M:%S')
df['Hour'] = df['Date'].dt.hour
df['WeekEnd'] = df['Date'].map(lambda x: 1 if x.weekday() < 5 else 0)
df['Power-1']=df['Central (kWh)'].shift(1)
df_2019=df.dropna()


def generate_table(dataframe, max_rows=15):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])



prev={}
mod_params={    
    'Bootstrapping': {
    'n_estimators': 100,  # Nombre d'estimateurs (modèles de base) dans le BaggingRegressor
    'max_samples': 1.0,  # Fraction de données d'entraînement à utiliser pour chaque échantillon bootstrap
    'max_features': 1.0,  # Fraction de caractéristiques à utiliser pour chaque modèle de base
    'bootstrap': True,  # Indique si les échantillons bootstrap doivent être utilisés lors de la construction des modèles de base
    'random_state': 42  # Seed pour la reproductibilité des résultats
},
    'Decision Tree Regressor': {
    'criterion': 'squared_error',  # La fonction pour mesurer la qualité d'une division. 'mse' pour l'erreur quadratique moyenne.
    'splitter': 'best',  # La stratégie utilisée pour choisir la division à chaque nœud. 'best' pour choisir la meilleure division, 'random' pour une division aléatoire.
    'max_depth': None,  # La profondeur maximale de l'arbre. Utilisez None pour ne pas définir de limite de profondeur.
    'min_samples_split': 2,  # Le nombre minimum d'échantillons requis pour diviser un nœud interne.
    'min_samples_leaf': 1,  # Le nombre minimum d'échantillons requis pour être à un nœud feuille.
    'min_weight_fraction_leaf': 0.0,  # La fraction pondérée minimale des échantillons requise pour être à un nœud feuille.
    'max_features': None,  # Le nombre de caractéristiques à considérer lors de la recherche de la meilleure division. Utilisez None pour considérer toutes les caractéristiques.
    'random_state': 42  # La graine aléatoire pour la reproductibilité des résultats.
},
    'Gradient Boosting Regressor': {
    'n_estimators': 100,  # Nombre d'estimateurs (arbres) dans le boosting
    'learning_rate': 0.1,  # Taux d'apprentissage - contrôle la contribution de chaque arbre
    'max_depth': 3,  # Profondeur maximale des arbres de base
    'min_samples_split': 2,  # Nombre minimal d'échantillons requis pour diviser un nœud
    'min_samples_leaf': 1,  # Nombre minimal d'échantillons requis pour être à un nœud feuille
    'subsample': 1.0,  # Fraction des échantillons utilisés pour ajuster chaque arbre
    'loss': 'squared_error',  # Fonction de perte à optimiser ('ls' pour la régression)
    'random_state': 42  # Seed pour la reproductibilité des résultats
},
    'Neural Network Regressor': {'hidden_layer_sizes': (100,),  # Nombre de neurones dans chaque couche cachée
    'activation': 'relu',  # Fonction d'activation pour les couches cachées ('relu' est une bonne valeur par défaut)
    'solver': 'adam',  # Algorithme d'optimisation ('adam' est généralement une bonne option pour les réseaux de neurones)
    'alpha': 0.0001,  # Terme de régularisation pour contrôler le surapprentissage
    'learning_rate': 'adaptive',  # Taux d'apprentissage adaptatif pour un ajustement fin pendant l'entraînement
    'max_iter': 500,  # Nombre maximal d'itérations pour l'entraînement
    'batch_size': 32,  # Taille des mini-lots pour la descente de gradient stochastique
    'random_state': 42,  # Seed pour la reproductibilité des résultats
},
    'Random Forest Regressor': {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 20,
    # 'min_samples_leaf': 10
},
    'Support Vector Regressor (Linear)': {'kernel':'linear'},
    'Support Vector Regressor (Polynomial)': {'kernel':'poly'},
    'Support Vector Regressor (RBF)': {'kernel':'rbf'},
    'Linear Regression': {},
}


mod_func = {
    'Bootstrapping': BaggingRegressor,
    'Decision Tree Regressor': DecisionTreeRegressor,
    'Gradient Boosting Regressor': GradientBoostingRegressor,
    'Linear Regression': linear_model.LinearRegression,
    'Neural Network Regressor': MLPRegressor,
    'Random Forest Regressor': RandomForestRegressor,
    'Support Vector Regressor (Linear)': SVR,
    'Support Vector Regressor (Polynomial)': SVR,
    'Support Vector Regressor (RBF)': SVR
    }



list_features=['Power-1','Hour','temp_C','solarRad_W/m2','WeekEnd']
# list_methods=['Decision Tree Regressor']
list_methods=[
            #   'Bootstrapping',
            #   'Decision Tree Regressor',
            #   'Gradient Boosting Regressor',
              'Linear Regression'
            #   'Neural Network Regressor',
            #   'Random Forest Regressor',
            #   'Support Vector Regressor (Linear)',
            #   'Support Vector Regressor (Polynomial)',
            #   'Support Vector Regressor (RBF)'
              ]

# this function gives, for a list of features and of method of regression, the metric and the prediction
def feature_to_prev(features=list_features,methods=list_methods,data_train=df_train,data_pred=df_2019):
    
    metric={'Methods':[],'MAE':[],'MBE':[],'MSE':[],'RMSE':[],'cvRMSE':[],'NMBE':[]}
    forecast={'Date':data_pred['Date'].values}

    Z=df_train[['Power_kW']+features].values
    y_train=Z[:,0]
    X_train=Z[:,range(1,len(Z[1]))]
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train_ss = ss_X.fit_transform(X_train)
    y_train_ss = ss_y.fit_transform(y_train.reshape(-1,1))        

    X_test=data_pred[features].values
    X_test_ss=ss_X.fit_transform(X_test)

    y=data_pred['Central (kWh)'].values
       
    for method in methods:
        metric['Methods'].append(method)
        param=mod_params[method]
        model=mod_func[method](**param)
        model.fit(X_train_ss, y_train_ss)

        y_pred = model.predict(X_test_ss)
        y_pred = ss_y.inverse_transform(y_pred.reshape(-1,1))

        forecast[method]=y_pred.flatten()

        MAE=metrics.mean_absolute_error(y,y_pred)
        MBE=np.mean(y-y_pred) 
        MSE=metrics.mean_squared_error(y,y_pred)  
        RMSE= np.sqrt(metrics.mean_squared_error(y,y_pred))
        cvRMSE=RMSE/np.mean(y)
        NMBE=MBE/np.mean(y)
        met_name=['MAE','MBE','MSE','RMSE','cvRMSE','NMBE']
        met_value=[MAE,MBE,MSE,RMSE,cvRMSE,NMBE]        

        for met in range(len(met_name)):
            metric[met_name[met]].append(met_value[met])
    
    df_metrics = pd.DataFrame(data=metric)  
    df_forecast=pd.DataFrame(data=forecast)
    df_results=pd.merge(data_pred.iloc[:,[0,1]],df_forecast, on='Date')
    return df_metrics, df_forecast, df_results


