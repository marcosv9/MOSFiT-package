import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash import Dash, Input, Output, ctx, html, dcc
import pandas as pd
import dash_bootstrap_components as dbc
import chaosmagpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import glob
import matplotlib.gridspec as gridspec
import main_functions as mvs
import data_processing_tools as dpt
import utilities_tools as utt
import support_functions as spf
#import pwlf
from datetime import datetime,date
import timeit
from datetime import datetime 
import pwlf
import plots_obs as obs
import seaborn as sns
import os



def mosfit_dash():    
    csv = dict({'VSS': 'VSS',
                'NGK': 'NGK',
               'KAK':'KAK'})
    
    #creating dict for pillars
    samples = dict({'hourly':'H',
                    'daily': 'D',
                    'monthly': 'M',
                    'annual': 'Y',
                    'Secular Variation':'SV'})
    
    data_selection = dict({'Empty':'Not',
                           'nighttime':'NT',
                          'quiet-days':'QD',
                          'disturbed-days':'DD',
                          'Kp-index':'KP'})
    
    
    
    # Build App
    app = JupyterDash(__name__)
    
    fig = go.Figure(data=[go.Scatter(x=[], y=[])])
    
    fig.update_layout(paper_bgcolor="#2A3F54")
    
    fig2 = go.Figure(data = [go.Table()])
    
    fig2.update_layout(paper_bgcolor="#2A3F54",
                          plot_bgcolor = "#2A3F54",
                          width = 250,
                          height = 100)
    
    
    
    app.layout = html.Div(
        className="content",
        children =[
            html.H1('Secular Variation Dashboard', id = 'title1'),
            html.Div(
                className = 'left-content',
                children=[
                    html.H1('Data processing options', id = 'left-content-title'),
                    html.Div(
                        className = 'Input',
                        children = [
                            html.Label('IAGA CODE'),
                            html.Br(),
                            dcc.Input(
                                id='imos-input',
                                type = 'text',
                                minLength = 3,
                                maxLength = 3,
                                debounce=True,
                                size = '5',
                                value = 'VSS')
                        ]),
                    html.Div(
                        className = 'dropdowns',
                        children = [
                            html.Label('Sample rate'),
                            dcc.Dropdown(
                                id='samples-dropdown',                                        
                                clearable=False,
                                value = 'H',
                                options=[{"label": i, "value": j} for i, j in samples.items()]),
                            html.Br(),
                            html.Label('use resample condition?'),
                            dcc.RadioItems(
                                id = 'resample-cond',
                                options = ['Yes', 'No'],
                                value = 'No'),
                            html.Label('Data Selection'),
                            dcc.Dropdown(
                                id='selection-dropdown',                                        
                                clearable=False,
                                value = 'Not',
                                options=[{"label": i, "value": j} for i, j in data_selection.items()])
                        ]),
                    html.Div(
                        className = 'radio-itens',
                        children=[
                            html.Label('CHAOS-7 correction'),
                            html.Br(),
                            dcc.RadioItems(
                                id = 'chaos',
                                options = ['Yes', 'No'],
                                value = 'No'),
                            html.Label('Plot CHAOS'),
                            dcc.RadioItems(id = 'chaos-plot',
                                           options = ['Yes', 'No'],
                                           value = 'No')
                        ]),
                    html.Div(
                        className = 'jerk_input',
                        children = [
                            html.H1('Geomagnetic jerk detection', id = 'title-detection'),
                            html.Label('Start-date'),
                            html.Br(),
                            dcc.Input(
                                id='start-date-input',
                                placeholder="yyyy-mm",
                                minLength = 7,
                                maxLength = 7,
                                debounce = True),
                            html.Br(),
                            html.Label('End-date'),
                            html.Br(),
                            dcc.Input(
                                id='end-date-input',
                                placeholder = 'yyyy-mm',
                                minLength = 7,
                                maxLength = 7,
                                debounce = True),
                            html.Br(),
                            html.Button(
                                'Detect',
                                id = 'button2',
                                n_clicks = 0)    
                        ]),
                    
    
                    html.Div(className = 'jerk_stats', children=[
                        html.H3('Jerk detection statistics', id = 'jerk-stats-title'),
                        html.Div(
                        className = 'Graph_stats',
                        children = [
                            dcc.Graph(
                                id='graph2',
                                figure=fig2)
                        
                    ]),
                    html.Div(
                        className = 'Button',
                        children=[
                            html.Button(
                                'Save data',
                                id='button1',
                                n_clicks = 0)
                        ]),
                    
                    ])
                ]),
            html.Div(
                className = 'right-content', 
                children=[
                    html.Div(
                        className = 'Graph',
                        children = [
                            dcc.Graph(
                                id='graph',
                                figure=fig)])
                ])
        ])
    
    
    #html.Div([
    #    html.Div([
    #        dcc.Graph(id='graph',
    #                figure = fig)])])])
    #
    
    
    
    @app.callback([Output('graph', 'figure'),
                   Output('graph2', 'figure')],
                  [Input("imos-input", "value"),
                  Input('samples-dropdown','value'),
                  Input('resample-cond','value'),
                  Input('selection-dropdown','value'),
                  Input('chaos', 'value'),
                  Input('chaos-plot','value'),
                  Input('button1', 'n_clicks'),
                  Input('start-date-input','value'),
                  Input('end-date-input','value'),
                  Input('button2', 'n_clicks')]
    )
    
    def update_figure(imo, sample, condition, selection, chaos, plot, n_click, starttime, endtime, n_click2):
    
        triggered_id = ctx.triggered_id
        
        mode = 'lines'
        
        annotations = []
        
        if condition == 'Yes':
            apply_percentage = True
        else:
            apply_percentage = False
        
        if selection in ['DD','QD', 'NT', 'KP']:
            apply_percentage = False
        
        fig2 = go.Figure(data = [go.Table()])
    
        fig2.update_layout(paper_bgcolor="#2A3F54",
                           plot_bgcolor = "#2A3F54",
                           width = 250,
                           height = 100)
        
        fig2.update_xaxes(showline=False, showgrid = False)
        fig2.update_yaxes(showline=False, showgrid = False)
        
        df_jerk_window = pd.DataFrame()
        
        df = pd.read_csv(f'C:\\Users\marco\\Downloads\\Thesis_notebooks\\hourly_data\\{imo}_hourly_data.txt', sep = '\t')
        df.index = pd.to_datetime(df['Date'], format= '%Y-%m-%d %H:%M:%S.%f')
        df.pop('Date')
        
        df_chaos = pd.read_csv(f'C:\\Users\marco\\Downloads\\Thesis_notebooks\\hourly_data\\{imo}_chaos_hourly_data.txt', sep = '\t')
        df_chaos.index = pd.to_datetime(df_chaos['Unnamed: 0'], format= '%Y-%m-%d %H:%M:%S.%f')
        df_chaos.pop('Unnamed: 0')
        
        if chaos == 'Yes':    
            
            df, df_chaos = dpt.external_field_correction_chaos_model(imo,
                                                                     '2000-01-01',
                                                                     '2022-08-30',
                                                                     df,
                                                                     df_chaos,
                                                                     apply_percentage = apply_percentage)
        else:
            pass
            
            
        if selection == 'DD':
            df = dpt.remove_disturbed_days(df)
        if selection == 'QD':
            df = dpt.keep_quiet_days(df)
        if selection == 'KP':
            df = dpt.kp_index_correction(df,
                                         kp = 2)
        if selection == 'NT':
            df = dpt.night_time_selection(station = imo,
                                          dataframe = df)
        
        df_detection = df
        
        if sample != 'SV':    
            df = dpt.resample_obs_data(df, sample, apply_percentage=apply_percentage)
        else:
            df = dpt.calculate_sv(df, apply_percentage = apply_percentage)
            mode = 'markers'
        
        if sample == 'Y':
            mode = 'lines+markers'
        
    
        
        
        if starttime != None and endtime != None and sample == 'SV' and triggered_id == 'button2':
            
            
            
            df_jerk_window, df_slopes, breakpoints, r2 = dpt.jerk_detection_window(station = imo,
                                                                                window_start = str(starttime),
                                                                                window_end = str(endtime),
                                                                                starttime = str(df.index.date[0]),
                                                                                endtime = str(df.index.date[-1]),
                                                                                df_station = df_detection,
                                                                                df_chaos = df_chaos,
                                                                                chaos_correction = False, 
                                                                                plot_chaos_prediction=False,
                                                                                plot_detection=False)
            
            df_jerk_stats = pd.DataFrame()
            
            df_jerk_stats['Comp'] = ['SV_X','SV_Y','SV_Z']
                
            df_jerk_stats['Occur'] =  breakpoints.iloc[1].values.round(2)
    
            df_jerk_stats['Amplitude'] = df_slopes.diff().iloc[1].values.round(2)
            
            df_jerk_stats['R2'] = r2
            
            fig2 = go.Figure(data=[go.Table(
                header=dict(values=list(df_jerk_stats.columns),
                            fill_color='#2A3F54',
                            align='left'),
                cells=dict(values=[df_jerk_stats.Comp, df_jerk_stats.Occur, df_jerk_stats.Amplitude, df_jerk_stats.R2], # 2nd column
                           line_color='#2A3F54',
                           fill_color='#2A3F54',
                           align='left'))
            ])
            
            fig2.update_layout(width = 250,
                               height = 100,
                               margin=dict(l=0, r=0, t=10, b=0),
                               paper_bgcolor="#2A3F54",
                               plot_bgcolor = '#2A3F54',
                               font_color = 'white')
            
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                row_heights=[0.55, 0.55, 0.55])
        
        #fig.update_layout(xaxis_rangeslider_visible=True)
        
        
        #coltroling subplots
        if sample != 'SV':
            for col, row, color in zip(df.columns, [1,2,3], ['#1616A7','#1CA71C','#222A2A']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], mode = mode, line_color=color, name= col),
                    row=row, col=1)
        
        #setting axis label
        
        for col, row in zip(df.columns, [1,2,3]):
            fig.update_yaxes(title_text= f'{col} (nT)', row=row, col=1, color = 'white')
    
        if sample == 'SV':
            for col, row, color  in zip(df.columns, [1,2,3], ['#1616A7','#1CA71C','#222A2A']):
                
                fig.update_yaxes(
                    title_text= f'SV {col} (nT/Yr)', row=row, col=1, color = 'white')
                
                fig.add_trace(
                    go.Scatter(x= df.index, y=df[col], mode = mode, line_color=color, name= f' SV {col}'),
                    row=row, col=1)
    
        fig.update_xaxes(title_text= 'Date', row=3, col=1, color = 'white')
        
        #adding chaos SV prediction
        if plot == 'Yes' and sample == 'SV':
            
            df_chaos = dpt.calculate_sv(df_chaos, columns = ['X_int','Y_int','Z_int'])
            
            fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['X_int'], mode = 'lines', line_color='#D62728', name = 'CHAOS SV X'),row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['Y_int'], mode = 'lines', line_color='#D62728', name = 'CHAOS SV Y'),row=2, col=1)
            fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['Z_int'], mode = 'lines', line_color='#D62728', name = 'CHAOS SV Z'),row=3, col=1)
            
         
        if df_jerk_window.empty == False:
            
            fig.add_trace(go.Scatter(x=df_jerk_window.index, y=df_jerk_window['X'], mode = 'lines', line_color='rgb(255,215,0)', name = 'SV X detection'),row=1, col=1)
            fig.add_trace(go.Scatter(x=df_jerk_window.index, y=df_jerk_window['Y'], mode = 'lines', line_color='rgb(255,215,0)',name = 'SV Y detection'),row=2, col=1)
            fig.add_trace(go.Scatter(x=df_jerk_window.index, y=df_jerk_window['Z'], mode = 'lines', line_color='rgb(255,215,0)', name = 'SV Z detection'),row=3, col=1)
            
        #updating entire figure layout
    
        
        fig.update_layout(title = 'Secular variation plot',
                          title_font_color=' white',
                          title_font_size = 20,
                          font_color= 'white',
                          height=600,
                          annotations= annotations,
                          width=1100,
                          autosize=False,
                          margin=dict(l=0, r=0, t=35, b=0),
                          paper_bgcolor="#2A3F54",
                          hovermode='x unified',
                          hoverlabel = dict(
                          bgcolor = 'white',
                          font_color = 'black'))  #plot_bgcolor='rgba(153,153,153,153)'  < - plot color
        #fig.update_traces(hovertemplate=hovertemp)
        
        if triggered_id == 'button1':
            
            df.to_csv('teste_save.txt', sep = ' ')
            
        #raise dash.exceptions.PreventUpdate
        
       
        return fig, fig2
    return app.run_server(mode='external', port = 3) 

#if __name__ == '__main__':
    #mosfit_dash()