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

app.layout = html.Div(
    className="content",
    children =[
        html.H1('Secular Variation Dashboard'),
        html.Div(
            className = 'left-content',
            children=[
                html.Div(
                    className = 'Input',
                    children[
                        html.Label('IAGA CODE'),
                        dcc.Input(
                            id='imos-input',
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
                            options=[{"label": i, "value": j} for i, j in samples.items()],
                            style={'width':'33%'}),
                        html.Br(),
                        html.Label('Data Selection'),
                        dcc.Dropdown(
                            id='selection-dropdown',                                        
                            clearable=False,
                            value = 'Not',
                            options=[{"label": i, "value": j} for i, j in data_selection.items()],
                            style={'width':'33%'})
                    ]),
                html.Div(
                    className = 'radio-itens',
                    children=[
                        html.Label('CHAOS-7 correction'),
                        dcc.RadioItems(
                            id = 'chaos',
                            options = ['Yes', 'No'],
                            value = 'No',
                            style={'width':'33%'}, className = 'radio-1'),
                        html.Label('Plot CHAOS'),
                        dcc.RadioItems(id = 'chaos-plot',
                                       options = ['Yes', 'No'],
                                       value = 'No')
                    ]),
                html.Div(
                    className = 'Button',
                    children=[
                        html.Button(
                            'Save data',
                            id='button1',
                            n_clicks = 0)
                    ])
            ]),
        html.Div(
            className = 'right-content', 
            children=[
                html.Div(
                    className = 'Graph',
                    children[
                        dcc.Graph(
                            id='graph',
                            figure=fig)
                    ])
            ])
    ])


#html.Div([
#    html.Div([
#        dcc.Graph(id='graph',
#                figure = fig)])])])
#


@app.callback(Output('graph', 'figure'),
              Input("imos-input", "value"),
              Input('samples-dropdown','value'),
              Input('selection-dropdown','value'),
              Input('chaos', 'value'),
              Input('chaos-plot','value'),
              Input('button1', 'n_clicks')
)



def update_figure(imo, sample, selection, chaos, plot, n_click):
    
    mode = 'lines'
    
    
    df = pd.read_csv(f'hourly_data/{imo}_hourly_data.txt', sep = '\s+')
    df.index = pd.to_datetime(df.index + ' ' + df['Date'], format= '%Y-%m-%d %H:%M:%S.%f')
    df.pop('Date')
    
    df_chaos = pd.read_csv(f'hourly_data/{imo}_chaos_hourly_data.txt', sep = '\t')
    df_chaos.index = pd.to_datetime(df_chaos['Unnamed: 0'], format= '%Y-%m-%d %H:%M:%S.%f')
    df_chaos.pop('Unnamed: 0')
    
    if chaos == 'Yes':    
        
        df, df_chaos = dpt.external_field_correction_chaos_model(imo,
                                                                 '2010-01-01',
                                                                 '2022-02-28',
                                                                 df,
                                                                 df_chaos)
        
        
    if selection == 'DD':
        df = dpt.remove_Disturbed_Days(df)
    if selection == 'QD':
        df = dpt.keep_Q_Days(df)
    if selection == 'KP':
        df = dpt.Kp_index_correction(df, kp = 3)
    if selection == 'NT':
        df = dpt.night_time_selection(df)
        
    if sample != 'SV':    
        df = dpt.resample_obs_data(df, sample, apply_percentage=True)
    else:
        df = dpt.calculate_SV(df)
        mode = 'markers'
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01,
            row_heights=[0.55, 0.55, 0.55])
    
    #fig.update_layout(xaxis_rangeslider_visible=True)
    
    
    #updating entire figure layout
    fig.update_layout(title = 'Secular variotion plot',
                      height=700,
                      width=1200)  #plot_bgcolor='rgba(153,153,153,153)'  < - plot color
    
    #coltroling subplots
    for col, row, color in zip(df.columns, [1,2,3], ['#1616A7','#1CA71C','#222A2A']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode = mode, line_color=color),
            row=row, col=1)
    
    #setting axis label
    
    for col, row in zip(df.columns, [1,2,3]):
        fig.update_yaxes(title_text= f'{col} (nT)', row=row, col=1)

    if sample == 'SV':
        for col, row in zip(df.columns, [1,2,3]):
            fig.update_yaxes(title_text= f'SV {col} (nT/Yr)', row=row, col=1)

    fig.update_xaxes(title_text= 'Date', row=3, col=1)
    
    #adding chaos SV prediction
    if plot == 'Yes' and sample == 'SV':
        
        df_chaos = dpt.calculate_SV(df_chaos, columns = ['X_int','Y_int','Z_int'])
        
        fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['X_int'], mode = mode, line_color='#D62728'),row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['Y_int'], mode = mode, line_color='#D62728'),row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chaos.index, y=df_chaos['Z_int'], mode = mode, line_color='#D62728'),row=3, col=1)
        
     
    triggered_id = ctx.triggered_id
    print(triggered_id)
    
    if triggered_id == 'button1':
        
        df.to_csv('teste_save.txt', sep = ' ')
        
    #raise dash.exceptions.PreventUpdate

    
    return fig

# Run app and display result inline in the notebook
app.run_server(mode='external',port=2222)