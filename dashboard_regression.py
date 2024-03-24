import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px


import prev_features as pf
import prev_data as pda

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize fig2
df2=pda.df_2019.iloc[:,[0,1,12,10,2,7,11]] # date, power and the 5 features
X2=df2.values
X2=X2[:,1:6]
fig1 = px.line(df2, x="Date", y=df2.columns[2:7])# Creates a figure with the raw data
df_metrics, df_forecast, df_results=pda.feature_to_prev(features=pda.list_features,methods=pda.list_methods,data_train=pda.df_train,data_pred=pda.df_2019)
fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:11])



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H1('IST Energy Forecast tool (kWh), by Augustin Tachoires'),
    html.P('Representing Data and Forecasting for 2019'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
            html.Div([
                html.H4('Select Features: the best ones are Power-1, Hour, temp_C, solarRad_W/m2 and WeekEnd'),
                dcc.Checklist(
                    id='metric-selector',
                    options=[{'label': 'Power-1', 'value': 'Power-1'},
                    {'label': 'Hour', 'value': 'Hour'},
                    {'label': 'temp_C', 'value': 'temp_C'},
                    {'label': 'solarRad_W/m2', 'value': 'solarRad_W/m2'},
                    {'label': 'WeekEnd', 'value': 'WeekEnd'},

                    {'label': 'HR', 'value': 'HR'},
                    {'label': 'windSpeed_m/s', 'value': 'windSpeed_m/s'},
                    {'label': 'windGust_m/s', 'value': 'windGust_m/s'},
                    {'label': 'pres_mbar', 'value': 'pres_mbar'},
                    {'label': 'rain_mm/h', 'value': 'rain_mm/h'},
                    {'label': 'rain_day', 'value': 'rain_day'}                    
                    ],
                    value=['Power-1', 'Hour','temp_C','solarRad_W/m2','WeekEnd'],  
                    inline=True
                ),
                html.Button('Calculate', id='calculate-button', n_clicks=0),
                dcc.Graph(id='selected-metrics-graph')
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('IST Electricity Forecast (kWh)'),
            html.Div([
                html.H4('Select variables to take into account: the best ones are Power-1, Hour, temp_C, solarRad_W/m2 and WeekEnd'),
                dcc.Checklist(
                    id='word-selector',
                    options=[{'label': 'Power-1', 'value': 'Power-1'},
                    {'label': 'Hour', 'value': 'Hour'},
                    {'label': 'temp_C', 'value': 'temp_C'},
                    {'label': 'solarRad_W/m2', 'value': 'solarRad_W/m2'},
                    {'label': 'WeekEnd', 'value': 'WeekEnd'},

                    {'label': 'HR', 'value': 'HR'},
                    {'label': 'windSpeed_m/s', 'value': 'windSpeed_m/s'},
                    {'label': 'windGust_m/s', 'value': 'windGust_m/s'},
                    {'label': 'pres_mbar', 'value': 'pres_mbar'},
                    {'label': 'rain_mm/h', 'value': 'rain_mm/h'},
                    {'label': 'rain_day', 'value': 'rain_day'}                    
                    ],
                    value=['Power-1', 'Hour','temp_C','solarRad_W/m2','WeekEnd'],              
                    inline=True
                ),
                html.Div([
                    html.H4('Select model(s) : be careful, the more you choose, the longer it will take'),
                    dcc.Checklist(
                        id='additional-selector',
                        options=[
                            {'label': 'Bootstrapping', 'value': 'Bootstrapping'},
                            {'label': 'Decision Tree Regressor', 'value': 'Decision Tree Regressor'},
                            {'label': 'Gradient Boosting Regressor', 'value': 'Gradient Boosting Regressor'},
                            {'label': 'Linear Regression', 'value': 'Linear Regression'},
                            {'label': 'Neural Network Regressor', 'value': 'Neural Network Regressor'},
                            {'label': 'Random Forest Regressor', 'value': 'Random Forest Regressor'},
                            {'label': 'Support Vector Regressor (Linear)', 'value': 'Support Vector Regressor (Linear)'},
                            {'label': 'Support Vector Regressor (Polynomial)', 'value': 'Support Vector Regressor (Polynomial)'},
                            {'label': 'Support Vector Regressor (RBF)', 'value': 'Support Vector Regressor (RBF)'}                            
                        ],
                        value=['Linear Regression'],
                        inline=True
                    ),
                ]),
                html.Button('Calculate', id='calculate-forecast-button', n_clicks=0),
                dcc.Graph(id='forecast-graph', figure=fig2),
                html.Div(id='forecast-table')
            ])
        ])        

@app.callback(
    Output('selected-metrics-graph', 'figure'),
    Input('calculate-button', 'n_clicks'),
    Input('metric-selector', 'value')
)

#update graph of the metrics when you click on Calculate
def update_graph(n_clicks, selected_metrics):
    if n_clicks > 0:   
        score_metrics=pd.DataFrame()
        score_metrics['Selected Metrics']=selected_metrics

        score_metrics['Filter_mutual_info']=pf.plot_features_scores(list_feat=selected_metrics,data=pf.df,method='Filter_mutual_info')
        score_metrics['Embedded']=pf.plot_features_scores(list_feat=selected_metrics,data=pf.df,method='Embedded')

        fig = px.bar(score_metrics, x=score_metrics.columns[0],y=score_metrics.columns[1:3])
        fig.update_layout(barmode='group')
        fig.update_layout(title='Selected Metrics')
        return fig
    else:
        return {}

@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-table', 'children')],
    [Input('calculate-forecast-button', 'n_clicks'),
     Input('word-selector', 'value'),
     Input('additional-selector', 'value')]
)

#update graph of the previsions when you click on Calculate
def update_forecast_graph(n_clicks, selected_features2,additional_features):
    if n_clicks > 0:
        df_metrics, df_forecast, df_results=pda.feature_to_prev(features=selected_features2,methods=additional_features,data_train=pda.df_train,data_pred=pda.df_2019)      
        fig = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:11])
        return fig, pda.generate_table(df_metrics)
    else:
        return {}

if __name__ == '__main__':
    app.run_server()
