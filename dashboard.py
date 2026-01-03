import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df_stocks = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
df_predictions = pd.read_csv('predictions_summary.csv')

# Try to load backtest results
try:
    df_backtest = pd.read_csv('backtest_summary.csv')
    has_backtest = True
except FileNotFoundError:
    df_backtest = pd.DataFrame()
    has_backtest = False

# Try to load trading signals
try:
    df_signals = pd.read_csv('daily_signals.csv')
    has_signals = True
except FileNotFoundError:
    df_signals = pd.DataFrame()
    has_signals = False

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stock Prediction Dashboard"

# Define colors
colors = {
    'background': '#0e1117',
    'text': '#ffffff',
    'card': '#1e2130',
    'accent': '#00d4ff',
    'green': '#00ff88',
    'red': '#ff4444'
}

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px', 'minHeight': '100vh'}, children=[
    html.Div([
        html.H1('📈 Stock Prediction & Trading Dashboard', 
                style={'textAlign': 'center', 'color': colors['accent'], 'marginBottom': '10px'}),
        html.P('Multi-Sector Stock Analysis, AI Predictions & Backtest Results', 
               style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '18px', 'marginBottom': '30px'})
    ]),
    
    # Tabs for different views
    dcc.Tabs(id='main-tabs', value='predictions', children=[
        dcc.Tab(label='📊 Predictions & Charts', value='predictions', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🚦 Trading Signals', value='signals', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🎯 Backtest Results', value='backtest', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
    ], style={'marginBottom': '30px'}),
    
    html.Div(id='tab-content')
])

# Callback to render tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'predictions':
        return html.Div([
            # Filters Row
    html.Div([
        html.Div([
            html.Label('Select Sector:', style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': 'All Sectors', 'value': 'ALL'}] + 
                        [{'label': sector, 'value': sector} for sector in sorted(df_stocks['Sector'].unique())],
                value='ALL',
                style={'backgroundColor': colors['card'], 'color': colors['text']},
                className='dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label('Select Stock:', style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='stock-dropdown',
                style={'backgroundColor': colors['card'], 'color': colors['text']},
                className='dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label('Chart Period:', style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='period-dropdown',
                options=[
                    {'label': '1 Month', 'value': 30},
                    {'label': '3 Months', 'value': 90},
                    {'label': '6 Months', 'value': 180},
                    {'label': '1 Year', 'value': 365},
                    {'label': 'All Data', 'value': 9999}
                ],
                value=180,
                style={'backgroundColor': colors['card'], 'color': colors['text']},
                className='dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'marginBottom': '30px'}),
    
    # KPI Cards Row
    html.Div(id='kpi-cards', style={'marginBottom': '30px'}),
    
    # Charts Row
    html.Div([
        # Candlestick Chart
        html.Div([
            dcc.Graph(id='candlestick-chart', style={'height': '500px'}),
        ], style={'width': '100%', 'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
    ], style={'marginBottom': '30px'}),
    
    # Volume Chart
    html.Div([
        dcc.Graph(id='volume-chart', style={'height': '200px'}),
    ], style={'width': '100%', 'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
    
    # Predictions Table
    html.Div([
        html.H3('🎯 AI Predictions Summary', style={'color': colors['accent'], 'marginBottom': '20px'}),
        html.Div(id='predictions-table')
    ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
    
    # All Stocks Summary Table
    html.Div([
        html.H3('📊 All Stocks Overview', style={'color': colors['accent'], 'marginBottom': '20px'}),
        dash_table.DataTable(
            id='stocks-table',
            columns=[
                {'name': 'Stock', 'id': 'Stock'},
                {'name': 'Ticker', 'id': 'Ticker'},
                {'name': 'Sector', 'id': 'Sector'},
                {'name': 'Price', 'id': 'Latest_Price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': '1d Pred', 'id': '1d_Direction'},
                {'name': '1d Prob', 'id': '1d_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                {'name': '7d Pred', 'id': '7d_Direction'},
                {'name': '7d Prob', 'id': '7d_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                {'name': '30d Pred', 'id': '30d_Direction'},
                {'name': '30d Prob', 'id': '30d_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                {'name': '90d Pred', 'id': '90d_Direction'},
                {'name': '90d Prob', 'id': '90d_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            ],
            data=df_predictions.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'backgroundColor': colors['card'],
                'color': colors['text'],
                'textAlign': 'left',
                'padding': '10px',
                'border': '1px solid #444'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{1d_Direction} = "UP ↑"', 'column_id': '1d_Direction'},
                    'color': colors['green'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{1d_Direction} = "DOWN ↓"', 'column_id': '1d_Direction'},
                    'color': colors['red'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{7d_Direction} = "UP ↑"', 'column_id': '7d_Direction'},
                    'color': colors['green'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{7d_Direction} = "DOWN ↓"', 'column_id': '7d_Direction'},
                    'color': colors['red'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{30d_Direction} = "UP ↑"', 'column_id': '30d_Direction'},
                    'color': colors['green'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{30d_Direction} = "DOWN ↓"', 'column_id': '30d_Direction'},
                    'color': colors['red'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{90d_Direction} = "UP ↑"', 'column_id': '90d_Direction'},
                    'color': colors['green'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{90d_Direction} = "DOWN ↓"', 'column_id': '90d_Direction'},
                    'color': colors['red'],
                    'fontWeight': 'bold'
                },
            ],
            filter_action="native",
            sort_action="native",
            page_size=20,
        )
    ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
        ])
    
    elif tab == 'backtest':
        if not has_backtest:
            return html.Div([
                html.H3('⚠️ No Backtest Results Available', 
                       style={'textAlign': 'center', 'color': colors['red'], 'marginTop': '50px'}),
                html.P('Run backtests first using: python backtest_trading.py', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '16px'})
            ])
        
        return html.Div([
            # Backtest Summary KPIs
            html.Div([
                html.Div([
                    html.H4('Avg Total Return', style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.H2(f"{df_backtest['Total_Return'].mean():.1%}", 
                           style={'color': colors['green'], 'margin': '0'}),
                    html.P(f'Across {len(df_backtest)} strategies', 
                          style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Avg Excess Return', style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.H2(f"{df_backtest['Excess_Return'].mean():.1%}", 
                           style={'color': colors['accent'], 'margin': '0'}),
                    html.P('vs Buy & Hold', 
                          style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Avg Win Rate', style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.H2(f"{df_backtest['Win_Rate'].mean():.1%}", 
                           style={'color': colors['green'], 'margin': '0'}),
                    html.P('Successful trades', 
                          style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Avg Sharpe Ratio', style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.H2(f"{df_backtest['Sharpe_Ratio'].mean():.2f}", 
                           style={'color': colors['accent'], 'margin': '0'}),
                    html.P('Risk-adjusted', 
                          style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Best Strategy', style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.H2(f"{df_backtest['Total_Return'].max():.1%}", 
                           style={'color': colors['green'], 'margin': '0'}),
                    html.P(f"{df_backtest.loc[df_backtest['Total_Return'].idxmax(), 'Stock']}", 
                          style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '18%', 'display': 'inline-block', 'textAlign': 'center'}),
            ], style={'marginBottom': '30px'}),
            
            # Performance Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='backtest-returns-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    dcc.Graph(id='backtest-sector-chart')
                ], style={'width': '49%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='backtest-winrate-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    dcc.Graph(id='backtest-sharpe-chart')
                ], style={'width': '49%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            # Top Performers Table
            html.Div([
                html.H3('🏆 Top 10 Performing Strategies', style={'color': colors['accent'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='top-performers-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Horizon', 'id': 'Horizon'},
                        {'name': 'Total Return', 'id': 'Total_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Excess Return', 'id': 'Excess_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Win Rate', 'id': 'Win_Rate', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Trades', 'id': 'Total_Trades', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                        {'name': 'Sharpe', 'id': 'Sharpe_Ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Max DD', 'id': 'Max_Drawdown', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                    ],
                    data=df_backtest.nlargest(10, 'Total_Return').to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'backgroundColor': colors['card'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Total_Return'},
                            'color': colors['green'],
                            'fontWeight': 'bold'
                        },
                    ],
                )
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # Full Backtest Results Table
            html.Div([
                html.H3('📊 All Backtest Results', style={'color': colors['accent'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='all-backtest-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Horizon', 'id': 'Horizon'},
                        {'name': 'Total Return', 'id': 'Total_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Buy & Hold', 'id': 'Buy_Hold_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Excess Return', 'id': 'Excess_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Win Rate', 'id': 'Win_Rate', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Trades', 'id': 'Total_Trades', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                        {'name': 'Sharpe', 'id': 'Sharpe_Ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Max DD', 'id': 'Max_Drawdown', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Final Value', 'id': 'Final_Value', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                    ],
                    data=df_backtest.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'backgroundColor': colors['card'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{Excess_Return} > 0',
                                'column_id': 'Excess_Return'
                            },
                            'color': colors['green'],
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{Excess_Return} < 0',
                                'column_id': 'Excess_Return'
                            },
                            'color': colors['red'],
                        },
                    ],
                    filter_action="native",
                    sort_action="native",
                    page_size=20,
                )
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
        ])
    
    elif tab == 'signals':
        if not has_signals:
            return html.Div([
                html.H3('⚠️ No Trading Signals Available', 
                       style={'textAlign': 'center', 'color': colors['red'], 'marginTop': '50px'}),
                html.P('Generate signals first using: python generate_daily_signals.py', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '16px'})
            ])
        
        # Calculate signal summary
        signal_summary = df_signals.groupby(['Horizon', 'Signal']).size().unstack(fill_value=0)
        
        # Get top BUY and SELL signals for 7d horizon
        df_7d = df_signals[df_signals['Horizon'] == '7d'].copy()
        df_buy = df_7d[df_7d['Signal'] == 'BUY'].nlargest(10, 'Signal_Strength')
        df_sell = df_7d[df_7d['Signal'] == 'SELL'].nlargest(10, 'Signal_Strength')
        
        return html.Div([
            # Signal Update Info
            html.Div([
                html.H3(f"🚦 Latest Trading Signals", 
                       style={'color': colors['accent'], 'display': 'inline-block', 'marginRight': '20px'}),
                html.P(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", 
                      style={'color': colors['text'], 'display': 'inline-block', 'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Signal Distribution by Horizon
            html.Div([
                html.Div([
                    html.H4('1-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['1d', 'BUY'] if '1d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['1d', 'HOLD'] if '1d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['1d', 'SELL'] if '1d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('7-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['7d', 'BUY'] if '7d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['7d', 'HOLD'] if '7d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['7d', 'SELL'] if '7d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('30-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['30d', 'BUY'] if '30d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['30d', 'HOLD'] if '30d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['30d', 'SELL'] if '30d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('90-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['90d', 'BUY'] if '90d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['90d', 'HOLD'] if '90d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['90d', 'SELL'] if '90d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            # Top BUY Signals (7-day)
            html.Div([
                html.H3('🟢 Top BUY Opportunities (7-Day Horizon)', style={'color': colors['green'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='buy-signals-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal Strength', 'id': 'Signal_Strength', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Prob UP', 'id': 'Probability_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Confidence', 'id': 'Model_Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Accuracy', 'id': 'Model_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Position Size', 'id': 'Position_Size', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                        {'name': 'Reason', 'id': 'Reason'},
                    ],
                    data=df_buy.to_dict('records') if not df_buy.empty else [],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'backgroundColor': colors['card'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Signal_Strength'},
                            'color': colors['green'],
                            'fontWeight': 'bold'
                        },
                    ],
                )
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # Top SELL Signals (7-day)
            html.Div([
                html.H3('🔴 Top SELL Warnings (7-Day Horizon)', style={'color': colors['red'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='sell-signals-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal Strength', 'id': 'Signal_Strength', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Prob DOWN', 'id': 'Probability_Down', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Confidence', 'id': 'Model_Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Accuracy', 'id': 'Model_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Reason', 'id': 'Reason'},
                    ],
                    data=df_sell.to_dict('records') if not df_sell.empty else [],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'backgroundColor': colors['card'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Signal_Strength'},
                            'color': colors['red'],
                            'fontWeight': 'bold'
                        },
                    ],
                )
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # All Signals Table with Filters
            html.Div([
                html.H3('📊 All Trading Signals', style={'color': colors['accent'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='all-signals-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Horizon', 'id': 'Horizon'},
                        {'name': 'Signal', 'id': 'Signal'},
                        {'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Strength', 'id': 'Signal_Strength', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Prob UP', 'id': 'Probability_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Prob DOWN', 'id': 'Probability_Down', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Confidence', 'id': 'Model_Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Accuracy', 'id': 'Model_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Position', 'id': 'Position_Size', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                        {'name': 'Reason', 'id': 'Reason'},
                    ],
                    data=df_signals.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'backgroundColor': colors['card'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{Signal} = "BUY"',
                                'column_id': 'Signal'
                            },
                            'color': colors['green'],
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{Signal} = "SELL"',
                                'column_id': 'Signal'
                            },
                            'color': colors['red'],
                            'fontWeight': 'bold'
                        },
                    ],
                    filter_action="native",
                    sort_action="native",
                    page_size=20,
                )
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
        ])

# Callback for backtest charts
@app.callback(
    Output('backtest-returns-chart', 'figure'),
    Output('backtest-sector-chart', 'figure'),
    Output('backtest-winrate-chart', 'figure'),
    Output('backtest-sharpe-chart', 'figure'),
    Input('main-tabs', 'value')
)
def update_backtest_charts(tab):
    if tab != 'backtest' or not has_backtest:
        return {}, {}, {}, {}
    
    # 1. Total Returns by Stock
    top_20 = df_backtest.nlargest(20, 'Total_Return')
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Bar(
        y=top_20['Stock'],
        x=top_20['Total_Return'] * 100,
        orientation='h',
        marker_color=colors['green'],
        text=[f"{x:.1f}%" for x in top_20['Total_Return'] * 100],
        textposition='outside'
    ))
    fig_returns.update_layout(
        title='Top 20 Strategies by Total Return',
        xaxis_title='Total Return (%)',
        yaxis_title='',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        height=600,
        margin=dict(l=150)
    )
    
    # 2. Average Return by Sector
    sector_avg = df_backtest.groupby('Sector')['Total_Return'].mean().sort_values(ascending=False)
    fig_sector = go.Figure()
    fig_sector.add_trace(go.Bar(
        x=sector_avg.index,
        y=sector_avg.values * 100,
        marker_color=colors['accent'],
        text=[f"{x:.1f}%" for x in sector_avg.values * 100],
        textposition='outside'
    ))
    fig_sector.update_layout(
        title='Average Return by Sector',
        xaxis_title='Sector',
        yaxis_title='Average Return (%)',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        height=400
    )
    
    # 3. Win Rate vs Return Scatter
    fig_winrate = go.Figure()
    for sector in df_backtest['Sector'].unique():
        sector_data = df_backtest[df_backtest['Sector'] == sector]
        fig_winrate.add_trace(go.Scatter(
            x=sector_data['Win_Rate'] * 100,
            y=sector_data['Total_Return'] * 100,
            mode='markers',
            name=sector,
            marker=dict(size=10, opacity=0.7),
            text=sector_data['Stock'],
            hovertemplate='<b>%{text}</b><br>Win Rate: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
    fig_winrate.update_layout(
        title='Win Rate vs Total Return by Sector',
        xaxis_title='Win Rate (%)',
        yaxis_title='Total Return (%)',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        height=400
    )
    
    # 4. Sharpe Ratio by Sector
    sharpe_avg = df_backtest.groupby('Sector')['Sharpe_Ratio'].mean().sort_values(ascending=False)
    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Bar(
        x=sharpe_avg.index,
        y=sharpe_avg.values,
        marker_color=colors['green'],
        text=[f"{x:.2f}" for x in sharpe_avg.values],
        textposition='outside'
    ))
    fig_sharpe.update_layout(
        title='Average Sharpe Ratio by Sector',
        xaxis_title='Sector',
        yaxis_title='Average Sharpe Ratio',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        height=400
    )
    
    return fig_returns, fig_sector, fig_winrate, fig_sharpe

# Callback to update stock dropdown based on sector
@app.callback(
    Output('stock-dropdown', 'options'),
    Output('stock-dropdown', 'value'),
    Input('sector-dropdown', 'value')
)
def update_stock_dropdown(selected_sector):
    if selected_sector == 'ALL':
        stocks = df_stocks.groupby(['Stock', 'Ticker']).size().reset_index()[['Stock', 'Ticker']]
    else:
        stocks = df_stocks[df_stocks['Sector'] == selected_sector].groupby(['Stock', 'Ticker']).size().reset_index()[['Stock', 'Ticker']]
    
    options = [{'label': f"{row['Stock']} ({row['Ticker']})", 'value': row['Ticker']} for _, row in stocks.iterrows()]
    default_value = options[0]['value'] if options else None
    
    return options, default_value

# Callback to update all charts and tables
@app.callback(
    Output('candlestick-chart', 'figure'),
    Output('volume-chart', 'figure'),
    Output('kpi-cards', 'children'),
    Output('predictions-table', 'children'),
    Input('stock-dropdown', 'value'),
    Input('period-dropdown', 'value')
)
def update_charts(selected_ticker, period_days):
    if not selected_ticker:
        return {}, {}, [], []
    
    # Filter data for selected stock
    df_stock = df_stocks[df_stocks['Ticker'] == selected_ticker].copy()
    
    if df_stock.empty:
        return {}, {}, [], []
    
    # Get prediction data
    pred_data = df_predictions[df_predictions['Ticker'] == selected_ticker].iloc[0]
    
    # Filter by period
    if period_days < 9999:
        df_stock = df_stock.tail(period_days)
    
    stock_name = df_stock['Stock'].iloc[0]
    sector = df_stock['Sector'].iloc[0]
    
    # Calculate technical indicators for the chart
    df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()
    df_stock['SMA_50'] = df_stock['Close'].rolling(window=50).mean()
    
    # Create candlestick chart
    fig_candle = go.Figure()
    
    fig_candle.add_trace(go.Candlestick(
        x=df_stock.index,
        open=df_stock['Open'],
        high=df_stock['High'],
        low=df_stock['Low'],
        close=df_stock['Close'],
        name='Price',
        increasing_line_color=colors['green'],
        decreasing_line_color=colors['red']
    ))
    
    # Add moving averages
    fig_candle.add_trace(go.Scatter(
        x=df_stock.index,
        y=df_stock['SMA_20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig_candle.add_trace(go.Scatter(
        x=df_stock.index,
        y=df_stock['SMA_50'],
        name='SMA 50',
        line=dict(color='purple', width=1)
    ))
    
    fig_candle.update_layout(
        title=f'{stock_name} ({selected_ticker}) - {sector}',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Create volume chart
    fig_volume = go.Figure()
    
    volume_colors = ['red' if close < open else 'green' 
                    for close, open in zip(df_stock['Close'], df_stock['Open'])]
    
    fig_volume.add_trace(go.Bar(
        x=df_stock.index,
        y=df_stock['Volume'],
        name='Volume',
        marker_color=volume_colors,
        opacity=0.7
    ))
    
    fig_volume.update_layout(
        title='Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        showlegend=False
    )
    
    # Calculate statistics
    latest_price = df_stock['Close'].iloc[-1]
    prev_price = df_stock['Close'].iloc[-2] if len(df_stock) > 1 else latest_price
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0
    
    period_high = df_stock['High'].max()
    period_low = df_stock['Low'].min()
    avg_volume = df_stock['Volume'].mean()
    
    # Create KPI cards
    kpi_cards = html.Div([
        # Current Price Card
        html.Div([
            html.H4('Current Price', style={'color': colors['text'], 'marginBottom': '5px'}),
            html.H2(f'${latest_price:.2f}', style={'color': colors['accent'], 'margin': '0'}),
            html.P(f'{price_change:+.2f} ({price_change_pct:+.2f}%)', 
                   style={'color': colors['green'] if price_change >= 0 else colors['red'], 
                          'fontSize': '16px', 'marginTop': '5px'})
        ], style={
            'backgroundColor': colors['background'], 
            'padding': '20px', 
            'borderRadius': '10px', 
            'width': '23%', 
            'display': 'inline-block',
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Period High Card
        html.Div([
            html.H4('Period High', style={'color': colors['text'], 'marginBottom': '5px'}),
            html.H2(f'${period_high:.2f}', style={'color': colors['green'], 'margin': '0'}),
            html.P(f'+{((period_high/latest_price - 1) * 100):.1f}%', 
                   style={'color': colors['text'], 'fontSize': '16px', 'marginTop': '5px'})
        ], style={
            'backgroundColor': colors['background'], 
            'padding': '20px', 
            'borderRadius': '10px', 
            'width': '23%', 
            'display': 'inline-block',
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Period Low Card
        html.Div([
            html.H4('Period Low', style={'color': colors['text'], 'marginBottom': '5px'}),
            html.H2(f'${period_low:.2f}', style={'color': colors['red'], 'margin': '0'}),
            html.P(f'{((period_low/latest_price - 1) * 100):.1f}%', 
                   style={'color': colors['text'], 'fontSize': '16px', 'marginTop': '5px'})
        ], style={
            'backgroundColor': colors['background'], 
            'padding': '20px', 
            'borderRadius': '10px', 
            'width': '23%', 
            'display': 'inline-block',
            'marginRight': '2%',
            'textAlign': 'center'
        }),
        
        # Average Volume Card
        html.Div([
            html.H4('Avg Volume', style={'color': colors['text'], 'marginBottom': '5px'}),
            html.H2(f'{avg_volume/1e6:.1f}M' if avg_volume >= 1e6 else f'{avg_volume/1e3:.1f}K', 
                   style={'color': colors['accent'], 'margin': '0'}),
            html.P(f'Last: {df_stock["Volume"].iloc[-1]/1e6:.1f}M' if df_stock["Volume"].iloc[-1] >= 1e6 
                   else f'Last: {df_stock["Volume"].iloc[-1]/1e3:.1f}K',
                   style={'color': colors['text'], 'fontSize': '16px', 'marginTop': '5px'})
        ], style={
            'backgroundColor': colors['background'], 
            'padding': '20px', 
            'borderRadius': '10px', 
            'width': '23%', 
            'display': 'inline-block',
            'textAlign': 'center'
        }),
    ])
    
    # Create predictions table
    predictions_table = html.Div([
        html.Div([
            # 1-Day Prediction
            html.Div([
                html.H4('1-Day Prediction', style={'color': colors['text'], 'textAlign': 'center'}),
                html.H2(pred_data['1d_Direction'], 
                       style={'color': colors['green'] if 'UP' in pred_data['1d_Direction'] else colors['red'],
                              'textAlign': 'center', 'fontSize': '36px'}),
                html.P(f"Probability: {pred_data['1d_Prob_Up']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Confidence: {pred_data['1d_Confidence']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Accuracy: {pred_data['1d_Accuracy']:.1%}", 
                      style={'color': colors['accent'], 'textAlign': 'center'})
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%',
                     'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'}),
            
            # 7-Day Prediction
            html.Div([
                html.H4('7-Day Prediction', style={'color': colors['text'], 'textAlign': 'center'}),
                html.H2(pred_data['7d_Direction'], 
                       style={'color': colors['green'] if 'UP' in pred_data['7d_Direction'] else colors['red'],
                              'textAlign': 'center', 'fontSize': '36px'}),
                html.P(f"Probability: {pred_data['7d_Prob_Up']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Confidence: {pred_data['7d_Confidence']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Accuracy: {pred_data['7d_Accuracy']:.1%}", 
                      style={'color': colors['accent'], 'textAlign': 'center'})
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%',
                     'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'}),
            
            # 30-Day Prediction
            html.Div([
                html.H4('30-Day Prediction', style={'color': colors['text'], 'textAlign': 'center'}),
                html.H2(pred_data['30d_Direction'], 
                       style={'color': colors['green'] if 'UP' in pred_data['30d_Direction'] else colors['red'],
                              'textAlign': 'center', 'fontSize': '36px'}),
                html.P(f"Probability: {pred_data['30d_Prob_Up']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Confidence: {pred_data['30d_Confidence']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Accuracy: {pred_data['30d_Accuracy']:.1%}", 
                      style={'color': colors['accent'], 'textAlign': 'center'})
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%',
                     'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'}),
            
            # 90-Day Prediction
            html.Div([
                html.H4('90-Day Prediction', style={'color': colors['text'], 'textAlign': 'center'}),
                html.H2(pred_data['90d_Direction'], 
                       style={'color': colors['green'] if 'UP' in pred_data['90d_Direction'] else colors['red'],
                              'textAlign': 'center', 'fontSize': '36px'}),
                html.P(f"Probability: {pred_data['90d_Prob_Up']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Confidence: {pred_data['90d_Confidence']:.1%}", 
                      style={'color': colors['text'], 'textAlign': 'center'}),
                html.P(f"Accuracy: {pred_data['90d_Accuracy']:.1%}", 
                      style={'color': colors['accent'], 'textAlign': 'center'})
            ], style={'width': '23%', 'display': 'inline-block',
                     'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'}),
        ])
    ])
    
    return fig_candle, fig_volume, kpi_cards, predictions_table

# Run the app
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Stock Prediction Dashboard")
    print("="*80)
    print("\n📊 Dashboard will be available at: http://localhost:8050")
    print("\n✨ Features:")
    print("   • Interactive candlestick charts with technical indicators")
    print("   • Filter by sector and stock")
    print("   • AI-powered predictions for 1, 7, 30, and 90 days")
    print("   • Real-time data visualization")
    print("   • Comprehensive stock overview table")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)
