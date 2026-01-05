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

# Try to load recommendations
try:
    df_recommendations = pd.read_csv('stock_recommendations.csv')
    has_recommendations = True
    print("✓ Loaded stock recommendations")
except FileNotFoundError:
    df_recommendations = pd.DataFrame()
    has_recommendations = False
    print("⚠️  No recommendations file found")

# Try to load refined predictions
try:
    df_predictions = pd.read_csv('predictions_refined.csv')
    is_refined = True
    print("✓ Loaded REFINED model predictions with trading metrics")
except FileNotFoundError:
    df_predictions = pd.DataFrame()
    is_refined = False
    print("⚠️  No predictions file found")

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

# Custom CSS for better dropdown visibility
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown styling for better visibility */
            .Select-control {
                background-color: #2d3142 !important;
                border: 2px solid #00d4ff !important;
                border-radius: 5px !important;
            }
            .Select-menu-outer {
                background-color: #2d3142 !important;
                border: 2px solid #00d4ff !important;
                border-radius: 5px !important;
            }
            .Select-option {
                background-color: #2d3142 !important;
                color: #ffffff !important;
                padding: 10px !important;
            }
            .Select-option:hover {
                background-color: #00d4ff !important;
                color: #0e1117 !important;
            }
            .Select-option.is-selected {
                background-color: #00d4ff !important;
                color: #0e1117 !important;
            }
            .Select-value-label {
                color: #ffffff !important;
            }
            .Select-placeholder {
                color: #aaaaaa !important;
            }
            .Select-input > input {
                color: #ffffff !important;
            }
            /* Modern dropdown arrow */
            .Select-arrow {
                border-color: #00d4ff transparent transparent !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Build table columns dynamically based on available prediction data
def build_prediction_columns():
    """Build table columns based on available predictions"""
    base_columns = [
        {'name': 'Stock', 'id': 'Stock'},
        {'name': 'Ticker', 'id': 'Ticker'},
        {'name': 'Sector', 'id': 'Sector'},
        {'name': 'Price', 'id': 'Latest_Price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
    ]
    
    # Add recommendation columns if available
    if has_recommendations:
        base_columns.extend([
            {'name': '🎯 Signal', 'id': 'Signal'},
            {'name': 'Recommendation', 'id': 'Recommendation'},
            {'name': 'Score', 'id': 'Score', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Strength', 'id': 'Strength'},
            {'name': 'Consensus', 'id': 'Consensus'},
        ])
    
    if df_predictions.empty and not has_recommendations:
        return base_columns
    
    pred_columns = []
    
    # Daily predictions (1d, 5d, 21d)
    for h in [1, 5, 21]:
        key = f'd{h}'
        if f'{key}_Direction' in df_predictions.columns:
            pred_columns.extend([
                {'name': f'{h}d Pred', 'id': f'{key}_Direction'},
                {'name': f'{h}d Prob', 'id': f'{key}_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            ])
            # Add accuracy if available (from refined models)
            if f'{key}_Accuracy' in df_predictions.columns:
                pred_columns.append(
                    {'name': f'{h}d Acc', 'id': f'{key}_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}}
                )
    
    return base_columns + pred_columns

def build_conditional_styles():
    """Build conditional styles for prediction columns"""
    styles = []
    
    # Recommendation styling
    if has_recommendations:
        styles.extend([
            # BUY recommendations
            {
                'if': {'filter_query': '{Recommendation} = "BUY"', 'column_id': 'Recommendation'},
                'backgroundColor': '#00ff8820',
                'color': '#00ff88',
                'fontWeight': 'bold'
            },
            # SELL recommendations
            {
                'if': {'filter_query': '{Recommendation} = "SELL"', 'column_id': 'Recommendation'},
                'backgroundColor': '#ff444420',
                'color': '#ff4444',
                'fontWeight': 'bold'
            },
            # HOLD recommendations
            {
                'if': {'filter_query': '{Recommendation} = "HOLD"', 'column_id': 'Recommendation'},
                'backgroundColor': '#ffffff10',
                'color': '#aaaaaa',
                'fontWeight': 'bold'
            },
            # Positive scores
            {
                'if': {'filter_query': '{Score} > 0', 'column_id': 'Score'},
                'color': '#00ff88'
            },
            # Negative scores
            {
                'if': {'filter_query': '{Score} < 0', 'column_id': 'Score'},
                'color': '#ff4444'
            },
        ])
    
    if df_predictions.empty:
        return styles
    
    # Get all direction columns
    dir_columns = [col for col in df_predictions.columns if col.endswith('_Direction')]
    
    for col in dir_columns:
        styles.extend([
            {
                'if': {'filter_query': f'{{{col}}} = "UP ↑"', 'column_id': col},
                'color': colors['green'],
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': f'{{{col}}} = "DOWN ↓"', 'column_id': col},
                'color': colors['red'],
                'fontWeight': 'bold'
            },
        ])
    
    return styles

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
title_text = '📈 Stock Prediction & Recommendation Dashboard'
subtitle_text = 'AI-Powered Trading Recommendations with Multi-Period Predictions'

app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px', 'minHeight': '100vh'}, children=[
    html.Div([
        html.H1(title_text, 
                style={'textAlign': 'center', 'color': colors['accent'], 'marginBottom': '10px'}),
        html.P(subtitle_text, 
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
                style={
                    'backgroundColor': '#2d3142',
                    'color': '#ffffff',
                    'borderRadius': '5px'
                },
                className='dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label('Select Stock:', style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='stock-dropdown',
                style={
                    'backgroundColor': '#2d3142',
                    'color': '#ffffff',
                    'borderRadius': '5px'
                },
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
                style={
                    'backgroundColor': '#2d3142',
                    'color': '#ffffff',
                    'borderRadius': '5px'
                },
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
        html.H3('📊 All Stocks Overview with Recommendations', style={'color': colors['accent'], 'marginBottom': '20px'}),
        html.P('BUY/HOLD/SELL recommendations based on combined 1d, 5d, 21d predictions weighted by confidence',
               style={'color': colors['text'], 'fontSize': '14px', 'marginBottom': '15px', 'fontStyle': 'italic'}),
        dash_table.DataTable(
            id='stocks-table',
            columns=build_prediction_columns(),
            data=df_recommendations.to_dict('records') if has_recommendations else df_predictions.to_dict('records'),
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
            style_data_conditional=build_conditional_styles(),
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
        
        # Add Probability_Down column for display
        df_signals_display = df_signals.copy()
        df_signals_display['Probability_Down'] = 1 - df_signals_display['Probability_Up']
        
        # Calculate signal summary
        signal_summary = df_signals_display.groupby(['Horizon', 'Signal']).size().unstack(fill_value=0)
        
        # Get top BUY and SELL signals for 5d horizon
        df_5d = df_signals_display[df_signals_display['Horizon'] == '5d'].copy()
        df_buy = df_5d[df_5d['Signal'] == 'BUY'].nlargest(10, 'Signal_Strength')
        df_sell = df_5d[df_5d['Signal'] == 'SELL'].nlargest(10, 'Signal_Strength')
        
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
                         'width': '31%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('5-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['5d', 'BUY'] if '5d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['5d', 'HOLD'] if '5d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['5d', 'SELL'] if '5d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '31%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('21-Day Signals', style={'color': colors['text'], 'marginBottom': '10px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H2(f"{signal_summary.loc['21d', 'BUY'] if '21d' in signal_summary.index and 'BUY' in signal_summary.columns else 0}", 
                                   style={'color': colors['green'], 'margin': '0'}),
                            html.P('BUY', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['21d', 'HOLD'] if '21d' in signal_summary.index and 'HOLD' in signal_summary.columns else 0}", 
                                   style={'color': colors['text'], 'margin': '0'}),
                            html.P('HOLD', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                        html.Div([
                            html.H2(f"{signal_summary.loc['21d', 'SELL'] if '21d' in signal_summary.index and 'SELL' in signal_summary.columns else 0}", 
                                   style={'color': colors['red'], 'margin': '0'}),
                            html.P('SELL', style={'color': colors['text'], 'fontSize': '14px'})
                        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                    ])
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '31%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            # Top BUY Signals (5-day)
            html.Div([
                html.H3('🟢 Top BUY Opportunities (5-Day Horizon)', style={'color': colors['green'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='buy-signals-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal Strength', 'id': 'Signal_Strength', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Prob UP', 'id': 'Probability_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Confidence', 'id': 'Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Accuracy', 'id': 'Model_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Position Size', 'id': 'Recommended_Position', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
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
            
            # Top SELL Signals (5-day)
            html.Div([
                html.H3('🔴 Top SELL Warnings (5-Day Horizon)', style={'color': colors['red'], 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='sell-signals-table',
                    columns=[
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal Strength', 'id': 'Signal_Strength', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'Prob DOWN', 'id': 'Probability_Down', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Confidence', 'id': 'Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
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
                        {'name': 'Confidence', 'id': 'Confidence', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Accuracy', 'id': 'Model_Accuracy', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Position', 'id': 'Recommended_Position', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                        {'name': 'Reason', 'id': 'Reason'},
                    ],
                    data=df_signals_display.to_dict('records'),
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
    
    # Calculate Support and Resistance Zones
    def find_support_resistance_zones(df, window=20, num_zones=3):
        """Identify support and resistance zones based on swing highs/lows"""
        zones = []
        
        # Calculate swing highs and lows
        df['swing_high'] = df['High'].rolling(window=window, center=True).max()
        df['swing_low'] = df['Low'].rolling(window=window, center=True).min()
        
        # Identify local maxima (resistance)
        df['is_resistance'] = (df['High'] == df['swing_high']) & (df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])
        
        # Identify local minima (support)
        df['is_support'] = (df['Low'] == df['swing_low']) & (df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])
        
        # Get resistance levels
        resistance_levels = df[df['is_resistance']]['High'].values
        support_levels = df[df['is_support']]['Low'].values
        
        # Cluster similar levels (within 2% of each other)
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            levels = sorted(levels, reverse=True)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[0]) / current_cluster[0] <= threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            clusters.append(np.mean(current_cluster))
            return clusters
        
        # Get top resistance and support zones
        resistance_zones = cluster_levels(resistance_levels)[:num_zones]
        support_zones = cluster_levels(support_levels)[:num_zones]
        
        return support_zones, resistance_zones
    
    support_zones, resistance_zones = find_support_resistance_zones(df_stock.copy(), window=20, num_zones=3)
    
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
    
    # Add Support Zones (green shaded areas)
    for i, support in enumerate(support_zones):
        zone_width = support * 0.01  # 1% zone width
        fig_candle.add_hrect(
            y0=support - zone_width,
            y1=support + zone_width,
            fillcolor="rgba(0, 255, 0, 0.15)",
            line_width=0,
            layer="below",
            annotation_text=f"Support {i+1}" if i == 0 else "",
            annotation_position="right"
        )
        fig_candle.add_hline(
            y=support,
            line_dash="dot",
            line_color="rgba(0, 255, 0, 0.5)",
            line_width=1,
            annotation_text=f"S: ${support:.2f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="green"
        )
    
    # Add Resistance Zones (red shaded areas)
    for i, resistance in enumerate(resistance_zones):
        zone_width = resistance * 0.01  # 1% zone width
        fig_candle.add_hrect(
            y0=resistance - zone_width,
            y1=resistance + zone_width,
            fillcolor="rgba(255, 0, 0, 0.15)",
            line_width=0,
            layer="below",
            annotation_text=f"Resistance {i+1}" if i == 0 else "",
            annotation_position="right"
        )
        fig_candle.add_hline(
            y=resistance,
            line_dash="dot",
            line_color="rgba(255, 0, 0, 0.5)",
            line_width=1,
            annotation_text=f"R: ${resistance:.2f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="red"
        )
    
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
        
        # Average Volume Card OR Model Accuracy (if refined)
        html.Div([
            html.H4('Avg Model Accuracy' if is_refined and 'd21_Accuracy' in pred_data else 'Avg Volume', 
                   style={'color': colors['text'], 'marginBottom': '5px'}),
            html.H2(f"{pred_data.get('d21_Accuracy', pred_data.get('d1_Accuracy', 0)):.1%}" 
                   if is_refined and ('d21_Accuracy' in pred_data or 'd1_Accuracy' in pred_data)
                   else (f'{avg_volume/1e6:.1f}M' if avg_volume >= 1e6 else f'{avg_volume/1e3:.1f}K'), 
                   style={'color': colors['accent'], 'margin': '0'}),
            html.P('21-day prediction accuracy' if is_refined and 'd21_Accuracy' in pred_data
                   else (f'Last: {df_stock["Volume"].iloc[-1]/1e6:.1f}M' if df_stock["Volume"].iloc[-1] >= 1e6 
                   else f'Last: {df_stock["Volume"].iloc[-1]/1e3:.1f}K'),
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
    
    # Create predictions table - show daily predictions (d1, d5, d21)
    prediction_cards = []
    
    # Daily predictions configuration
    pred_configs = [
        ('d1', '1-Day'),
        ('d5', '5-Day'),
        ('d21', '21-Day'),
    ]
    
    # Build cards for available predictions
    for key, label in pred_configs:
        dir_col = f'{key}_Direction'
        prob_col = f'{key}_Prob_Up'
        conf_col = f'{key}_Confidence'
        acc_col = f'{key}_Accuracy'
        
        if dir_col in pred_data.index and pd.notna(pred_data[dir_col]):
            prediction_cards.append(
                html.Div([
                    html.H4(f'{label} Prediction', style={'color': colors['text'], 'textAlign': 'center'}),
                    html.H2(pred_data[dir_col], 
                           style={'color': colors['green'] if 'UP' in str(pred_data[dir_col]) else colors['red'],
                                  'textAlign': 'center', 'fontSize': '36px'}),
                    html.P(f"Probability: {pred_data[prob_col]:.1%}", 
                          style={'color': colors['text'], 'textAlign': 'center'}),
                    html.P(f"Confidence: {pred_data[conf_col]:.1%}", 
                          style={'color': colors['text'], 'textAlign': 'center'}),
                    html.P(f"Accuracy: {pred_data[acc_col]:.1%}", 
                          style={'color': colors['accent'], 'textAlign': 'center'})
                ], style={'width': '31%', 'display': 'inline-block', 'marginRight': '2%',
                         'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'})
            )
    
    predictions_table = html.Div([html.Div(prediction_cards)]) if prediction_cards else html.Div([
        html.P('No predictions available for this stock', 
               style={'color': colors['text'], 'textAlign': 'center', 'padding': '20px'})
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
    print("   • AI-powered predictions for 1, 5, 21 days ahead")
    print("   • BUY/HOLD/SELL recommendations based on multi-period analysis")
    print("   • Real-time data visualization")
    print("   • Comprehensive stock overview table")
    print("   • Backtest results and trading signals")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)
