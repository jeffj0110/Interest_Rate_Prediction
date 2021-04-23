# Visualize prediction data
#
#
# This uses 'plotly' and 'chart studios' to create interactive graphs of the interest rate predictions
#
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio
username = 'jeffj0110'
api_key = '4oajckjUAa6sPUhgmSok'

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

import chart_studio.plotly as py
import chart_studio.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import data
IT_Predictions = 'IR_Validation_Results.csv'
IR_Expected_Results = 'IR_Validation_Data.csv'
IR_Predictions_DataFrame = pd.read_csv(IT_Predictions)
IR_Expected_Results_DataFrame = pd.read_csv(IR_Expected_Results)
IR_Predictions_DataFrame.drop('Days', 1, inplace=True)

#Just show the last 3 graphs
df = IR_Expected_Results_DataFrame.tail(3)
df2 = IR_Predictions_DataFrame.tail(3)

graph_X = pd.array(["3 Mth", "6 MTH", "1 YR", "2 YR", "3 YR", "5 YR", "10 YR"], dtype=str)

#When plotting online, the plot and data will be saved to your cloud account. 
#There are two methods for plotting online: py.plot() and py.iplot(). 
#Both options create a unique url for the plot and save it in your Plotly account.

#Use py.plot() to return the unique url and optionally open the url.
#Use py.iplot() when working in a Jupyter Notebook to display the plot in the notebook.
#Plotly allows you to create graphs offline and save them locally. There are also two methods for interactive plotting offline: plotly.io.write_html() and plotly.io.show().

#Use plotly.io.write_html() to create and standalone HTML that is saved locally and opened inside your web browser.
#Use plotly.io.show() when working offline in a Jupyter Notebook to display the plot in the notebook.


rowcount = len(df)

fig = make_subplots(rows=rowcount, cols=1,
                   subplot_titles = ['temp_subtitle' for var1 in np.arange(len(df))]
                   )

for x in range(0, rowcount, 1):
    R_Transpose = []
    P_Transpose = []
    TimeStamp = df.iloc[x,0]
    date_time_obj = datetime.datetime.strptime(TimeStamp, '%m/%d/%Y')
    Rates = df.iloc[x:x+1, 7:14]
    R_Transpose = Rates.transpose()
    R_Transpose.columns = ['Actual Rate']
    R_Transpose['Term'] = graph_X
    
    Predictions = df2.iloc[x:x+1, 7:14]
    P_Transpose = Predictions.transpose()
    R_Transpose['Pred Rate'] = P_Transpose
    if x == 0 :
        fig.add_trace(go.Scatter( 
        x=R_Transpose['Term'],
        y=R_Transpose['Actual Rate'],
        showlegend=True,
        line=dict(
                color='rgb(0,0,255)',
                width=2
            ),
        name="Actual Rate Curve"),
        row=x+1, 
        col=1
        )
        fig.add_trace(go.Scatter( 
        x=R_Transpose['Term'],
        y=R_Transpose['Pred Rate'],
        showlegend=True,
        line=dict(
              color='rgb(255,0,0)',
              width=2
            ),
        name="Predicted Rates"),
        row=x+1, 
        col=1
        )
    else : 
        fig.add_trace(go.Scatter( 
        x=R_Transpose['Term'],
        y=R_Transpose['Actual Rate'],
        showlegend=False,
        line=dict(
            color='rgb(0,0,255)',
            width=2
        ),
        name="Actual Rate Curve"),
        row=x+1, 
        col=1
        )
        fig.add_trace(go.Scatter( 
        x=R_Transpose['Term'],
        y=R_Transpose['Pred Rate'],
        showlegend=False,
        line=dict(
            color='rgb(255,0,0)',
            width=2
            ),
        name="Predicted Rates"),
        row=x+1, 
        col=1
        )
        
    fig.update_layout(
        title='Treasury Yield Curve Actual / Predictions',
        height=rowcount*200,
        width=1000)
    
    # to change subtitle, address subplot
    annotation_counter = 0
    for date_var in df['Date_MMDDYYYY'] :
        fig['layout']['annotations'][annotation_counter].update(text=date_var)
        annotation_counter += 1
        
fig.write_html('yield_graph.html')