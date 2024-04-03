from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from copy import deepcopy
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder


# Initialise the app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Load the dataset
df_train = pd.read_csv('~/Schreibtisch/Data/Kaggle/playground-series-s4e4/train.csv')


df_train.set_index('id', inplace=True)


# Sex categorical
categorical_columns = df_train.columns[df_train.nunique() == 3].tolist()
# Define numerical columns
numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
# exclude the binary columns, because they are categorial

target_variable = 'Rings'

#---------------------------- Data preparation for correlatin ------------------------------
# Splitting the dataset into features and target variable
X = df_train.drop(target_variable, axis=1)
y = df_train[target_variable]

# -----------------------------End-------------------------------------

# Define the layout of the dashboard
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H1('Data Visualisation Dashboard')
        ])
    ]),
    # Description of the variables
    dbc.Row([
        dcc.Markdown('Description of the variables can be found [here](https://www.kaggle.com/code/chetanaggarwal01/obesity-or-cvd-risk-classify-regressor-cluster)')
    ]),
    # plot missing values
    dbc.Row([
        dcc.Markdown('### Missing values: Plot the missing values in the dataset with a bar chart'),
        dcc.Graph(figure=px.bar(df_train.isna().sum()), style={'height': '500px'})
    ]),
    # Histogram for numerical columns
    dbc.Row([
        dcc.Markdown('### Histogram: Distribution of the numerical and categorial variables'),
        # create 2 dropdowns aside each other, 1 for the column, and 1 for the color
    dbc.Col([
        dcc.Dropdown(
            id='histogram_column',
            options=[{'label': col, 'value': col} for col in df_train.columns],
            value=numerical_columns[0]
        )], width=6),
        # now the color
        dbc.Col([
            dcc.Dropdown(
                id='histogram_color',
                options= ['Rings', 'Sex'],
                value=target_variable
            )], width=6),

        dcc.Graph(id='histogram', style={'height': '800px'})
    ]),
    # Scatterplot of two columns for numerical values
    dbc.Row([
        dcc.Markdown('### 2D Scatterplot: Plot two numerical columns against each other'),
        dcc.Markdown('Select two columns to plot against each other'),
        dbc.Col([
            dcc.Dropdown(
                id='2d_plot_x',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value=numerical_columns[0]
            ),
            dcc.Dropdown(
                id='2d_plot_y',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value=numerical_columns[1]
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='2d_plot', style={'height': '800px'})
        ])
    ]),
    # parallel coordinates
    dbc.Row([
        dcc.Markdown('### Parallel Coordinates: Parallel coordinates plot of the numerical columns'),
        dcc.Graph(figure=px.parallel_coordinates(df_train, dimensions= ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1','Whole weight.2', 'Shell weight', 'Rings'], color='Rings'))
    ])
])



# Callback to update the histogram
@app.callback(
    Output('histogram', 'figure'),
    Input('histogram_column', 'value'),
    Input('histogram_color', 'value')
)
def update_histogram(y, color):
    fig = px.histogram(df_train, x=y, color=color, marginal='violin')
    return fig

# Callback to update the scatterplot
@app.callback(
    Output('2d_plot', 'figure'),
    Input('2d_plot_x', 'value'),
    Input('2d_plot_y', 'value')
)
def update_2d_plot(x, y):
    fig = px.scatter(df_train, x=x, y=y, color=target_variable, marginal_y='violin', marginal_x='violin')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)