import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

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

# Try to load risk metrics
try:
    df_risk = pd.read_csv('risk_metrics.csv')
    df_stops = pd.read_csv('stop_losses.csv')
    has_risk = True
    print("✓ Loaded risk management data")
except FileNotFoundError:
    df_risk = pd.DataFrame()
    df_stops = pd.DataFrame()
    has_risk = False

# Try to load earnings calendar
try:
    df_earnings = pd.read_csv('earnings_calendar.csv')
    df_earnings['Earnings_Date'] = pd.to_datetime(df_earnings['Earnings_Date'], errors='coerce')
    has_earnings = True
    print("✓ Loaded earnings calendar")
except FileNotFoundError:
    df_earnings = pd.DataFrame()
    has_earnings = False

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
    
    # Signal styling
    if has_recommendations:
        styles.extend([
            # BUY signals
            {
                'if': {'filter_query': '{Signal} = "BUY"', 'column_id': 'Signal'},
                'backgroundColor': '#00ff8820',
                'color': '#00ff88',
                'fontWeight': 'bold'
            },
            # SELL signals
            {
                'if': {'filter_query': '{Signal} = "SELL"', 'column_id': 'Signal'},
                'backgroundColor': '#ff444420',
                'color': '#ff4444',
                'fontWeight': 'bold'
            },
            # HOLD signals
            {
                'if': {'filter_query': '{Signal} = "HOLD"', 'column_id': 'Signal'},
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
        dcc.Tab(label='� Sentiment Analytics', value='sentiment', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
        dcc.Tab(label='⚠️  Risk Management', value='risk', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
        dcc.Tab(label='📈 Performance Tracking', value='performance', style={'backgroundColor': colors['card'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['accent'], 'color': colors['background'], 'fontWeight': 'bold'}),
        dcc.Tab(label='�🚦 Trading Signals', value='signals', style={'backgroundColor': colors['card'], 'color': colors['text']},
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
                        [{'label': sector, 'value': sector} for sector in sorted(df_predictions['Sector'].unique())],
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
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label('Time Interval:', style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='interval-dropdown',
                options=[
                    {'label': '📅 Daily', 'value': 'D'},
                    {'label': '📊 Weekly', 'value': 'W'},
                    {'label': '📈 Monthly', 'value': 'M'}
                ],
                value='D',
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
    
    elif tab == 'sentiment':
        # Load sentiment data (prioritize complete 10-year backfill)
        sentiment_complete_exists = os.path.exists('data/sentiment_history_complete.csv')
        sentiment_history_exists = os.path.exists('data/sentiment_history.csv')
        sentiment_static_exists = os.path.exists('sentiment_data.csv')
        
        if not sentiment_complete_exists and not sentiment_history_exists and not sentiment_static_exists:
            return html.Div([
                html.H3('⚠️  No Sentiment Data Available', 
                       style={'textAlign': 'center', 'color': colors['red'], 'marginTop': '50px'}),
                html.P('Run: python fetch_sentiment_historical.py', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '16px'})
            ])
        
        # Load sentiment data (prefer complete backfill)
        if sentiment_complete_exists:
            df_sent = pd.read_csv('data/sentiment_history_complete.csv')
            df_sent['date'] = pd.to_datetime(df_sent['date'])
            is_historical = True
        elif sentiment_history_exists:
            df_sent = pd.read_csv('data/sentiment_history.csv')
            df_sent['date'] = pd.to_datetime(df_sent['date'])
            is_historical = True
        else:
            df_sent = pd.read_csv('sentiment_data.csv')
            is_historical = False
        
        # Sentiment summary
        avg_sentiment = df_sent['sentiment_compound'].mean()
        positive_count = (df_sent['sentiment_compound'] > 0.1).sum()
        negative_count = (df_sent['sentiment_compound'] < -0.1).sum()
        neutral_count = len(df_sent) - positive_count - negative_count
        
        if is_historical:
            num_dates = df_sent['date'].nunique()
            date_range = f"{df_sent['date'].min().date()} to {df_sent['date'].max().date()}"
            status_text = f"📊 Historical Sentiment: {num_dates} days of data ({date_range})"
        else:
            status_text = "📊 Static Sentiment (one-time snapshot)"
        
        # Top positive/negative
        latest_date = df_sent['date'].max() if is_historical else None
        if is_historical:
            df_latest = df_sent[df_sent['date'] == latest_date]
        else:
            df_latest = df_sent
        
        top_positive = df_latest.nlargest(5, 'sentiment_compound')
        top_negative = df_latest.nsmallest(5, 'sentiment_compound')
        
        # Create sentiment distribution chart
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_latest['sentiment_compound'],
            nbinsx=30,
            marker_color=colors['accent'],
            opacity=0.7,
            name='Sentiment Distribution'
        ))
        fig_dist.update_layout(
            title='Current Sentiment Distribution Across All Stocks',
            xaxis_title='Sentiment Score',
            yaxis_title='Number of Stocks',
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            showlegend=False
        )
        
        # Sentiment by sector
        df_stocks_info = pd.read_csv('data/multi_sector_stocks.csv')
        df_stocks_info = df_stocks_info[['Ticker', 'Sector']].drop_duplicates()
        df_sent_sector = df_latest.merge(df_stocks_info, left_on='ticker', right_on='Ticker', how='left')
        
        sector_sentiment = df_sent_sector.groupby('Sector')['sentiment_compound'].mean().sort_values()
        
        fig_sector = go.Figure()
        colors_list = [colors['red'] if x < -0.1 else colors['green'] if x > 0.1 else colors['text'] 
                      for x in sector_sentiment.values]
        
        fig_sector.add_trace(go.Bar(
            x=sector_sentiment.values,
            y=sector_sentiment.index,
            orientation='h',
            marker_color=colors_list,
            text=[f"{x:+.3f}" for x in sector_sentiment.values],
            textposition='auto'
        ))
        fig_sector.update_layout(
            title='Average Sentiment by Sector',
            xaxis_title='Sentiment Score',
            yaxis_title='Sector',
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            showlegend=False
        )
        
        # Time series chart (if historical)
        if is_historical and num_dates > 1:
            # Merge sector info for filtering
            df_sent_with_sector = df_sent.merge(df_stocks_info, left_on='ticker', right_on='Ticker', how='left')
            
            # Get unique sectors and stocks for dropdowns
            sectors = ['All Sectors'] + sorted(df_sent_with_sector['Sector'].dropna().unique().tolist())
            all_stocks = sorted(df_sent_with_sector['ticker'].unique().tolist())
            
            timeline_chart = html.Div([
                # Filter controls
                html.Div([
                    html.Div([
                        html.Label('Sector:', style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='sentiment-sector-filter',
                            options=[{'label': s, 'value': s} for s in sectors],
                            value='All Sectors',
                            style={'width': '200px', 'display': 'inline-block'},
                            clearable=False
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label('Stock:', style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='sentiment-stock-filter',
                            options=[{'label': 'All Stocks', 'value': 'All Stocks'}] + [{'label': s, 'value': s} for s in all_stocks],
                            value='All Stocks',
                            style={'width': '200px', 'display': 'inline-block'},
                            clearable=False
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label('View:', style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='sentiment-view-mode',
                            options=[
                                {'label': 'Daily', 'value': 'daily'},
                                {'label': 'Rolling 7-Day Avg', 'value': 'rolling_7'},
                                {'label': 'Rolling 30-Day Avg', 'value': 'rolling_30'}
                            ],
                            value='daily',
                            style={'width': '180px', 'display': 'inline-block'},
                            clearable=False
                        ),
                    ], style={'display': 'inline-block'}),
                ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': colors['background'], 'borderRadius': '10px'}),
                
                # Chart
                dcc.Graph(id='sentiment-timeline-chart', style={'height': '400px'}),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'})
        else:
            timeline_chart = html.Div()
        
        return html.Div([
            # Header
            html.Div([
                html.H3('📰 Sentiment Analytics', 
                       style={'color': colors['accent'], 'display': 'inline-block', 'marginRight': '20px'}),
                html.P(status_text, 
                      style={'color': colors['text'], 'display': 'inline-block', 'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H4('Average Sentiment', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{avg_sentiment:+.3f}", 
                           style={'color': colors['green'] if avg_sentiment > 0 else colors['red'], 'margin': '0'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('Positive', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{positive_count}", style={'color': colors['green'], 'margin': '0'}),
                    html.P(f"{positive_count/len(df_latest)*100:.0f}%", style={'color': colors['text'], 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('Neutral', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{neutral_count}", style={'color': colors['text'], 'margin': '0'}),
                    html.P(f"{neutral_count/len(df_latest)*100:.0f}%", style={'color': colors['text'], 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('Negative', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{negative_count}", style={'color': colors['red'], 'margin': '0'}),
                    html.P(f"{negative_count/len(df_latest)*100:.0f}%", style={'color': colors['text'], 'marginTop': '5px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            # Timeline (if available)
            timeline_chart,
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_dist, style={'height': '400px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    dcc.Graph(figure=fig_sector, style={'height': '400px'}),
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # Top stocks tables
            html.Div([
                html.Div([
                    html.H4('🟢 Most Positive Sentiment', style={'color': colors['green'], 'marginBottom': '15px'}),
                    dash_table.DataTable(
                        columns=[
                            {'name': 'Ticker', 'id': 'ticker'},
                            {'name': 'Sentiment', 'id': 'sentiment_compound', 'type': 'numeric', 'format': {'specifier': '+.3f'}},
                            {'name': 'News Count', 'id': 'news_count', 'type': 'numeric'},
                            {'name': 'Positive %', 'id': 'sentiment_positive', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        ],
                        data=top_positive.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': colors['background'],
                            'color': colors['text'],
                            'fontWeight': 'bold',
                        },
                        style_cell={
                            'backgroundColor': colors['card'],
                            'color': colors['text'],
                            'textAlign': 'center',
                        },
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('🔴 Most Negative Sentiment', style={'color': colors['red'], 'marginBottom': '15px'}),
                    dash_table.DataTable(
                        columns=[
                            {'name': 'Ticker', 'id': 'ticker'},
                            {'name': 'Sentiment', 'id': 'sentiment_compound', 'type': 'numeric', 'format': {'specifier': '+.3f'}},
                            {'name': 'News Count', 'id': 'news_count', 'type': 'numeric'},
                            {'name': 'Negative %', 'id': 'sentiment_negative', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        ],
                        data=top_negative.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': colors['background'],
                            'color': colors['text'],
                            'fontWeight': 'bold',
                        },
                        style_cell={
                            'backgroundColor': colors['card'],
                            'color': colors['text'],
                            'textAlign': 'center',
                        },
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
        ])
    
    
    elif tab == 'risk':
        if not has_risk:
            return html.Div([
                html.H3('⚠️  Risk Management Data Not Available', 
                       style={'textAlign': 'center', 'color': colors['red'], 'marginTop': '50px'}),
                html.P('Run: python risk_manager.py && python earnings_calendar.py', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '16px'})
            ])
        
        # Calculate portfolio risk metrics
        portfolio_volatility = (df_risk['Volatility_Annual'] * df_risk['Risk_Score']/100).mean()
        high_risk_count = (df_risk['Risk_Score'] > 66).sum()
        with_stops_count = len(df_stops)
        
        # Load warnings if available
        try:
            df_warnings = pd.read_csv('risk_warnings.csv')
            warnings_count = len(df_warnings)
        except:
            warnings_count = 0
        
        # Load earnings warnings if available  
        try:
            df_earn_warn = pd.read_csv('earnings_warnings.csv')
            earnings_warnings = len(df_earn_warn)
        except:
            earnings_warnings = 0
        
        # Merge sector info
        df_risk_with_sector = df_risk.merge(
            df_stocks[['Ticker', 'Sector']].drop_duplicates(), 
            on='Ticker', 
            how='left'
        )
        
        # Risk distribution chart
        fig_risk_dist = go.Figure()
        risk_bins = pd.cut(df_risk['Risk_Score'], bins=[0, 33, 66, 100], labels=['Low', 'Medium', 'High'])
        risk_counts = risk_bins.value_counts().sort_index()
        
        fig_risk_dist.add_trace(go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[colors['green'], colors['text'], colors['red']],
            text=risk_counts.values,
            textposition='auto'
        ))
        fig_risk_dist.update_layout(
            title='Risk Distribution Across Portfolio',
            xaxis_title='Risk Category',
            yaxis_title='Number of Stocks',
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            showlegend=False
        )
        
        # Volatility by sector
        sector_volatility = df_risk_with_sector.groupby('Sector')['Volatility_Annual'].mean().sort_values(ascending=False)
        
        fig_vol_sector = go.Figure()
        fig_vol_sector.add_trace(go.Bar(
            x=sector_volatility.values * 100,
            y=sector_volatility.index,
            orientation='h',
            marker_color=colors['accent'],
            text=[f"{x:.1f}%" for x in sector_volatility.values * 100],
            textposition='auto'
        ))
        fig_vol_sector.update_layout(
            title='Annual Volatility by Sector',
            xaxis_title='Volatility (%)',
            yaxis_title='Sector',
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            showlegend=False
        )
        
        # Top risky stocks table
        df_high_risk = df_risk.nlargest(10, 'Risk_Score')[['Ticker', 'Stock', 'Sector', 'Risk_Score', 'Volatility_Annual', 'Max_Drawdown']]
        df_high_risk['Volatility_Annual'] = df_high_risk['Volatility_Annual'] * 100
        df_high_risk['Max_Drawdown'] = df_high_risk['Max_Drawdown'] * 100
        
        # Stop losses table
        df_stops_display = df_stops[['Ticker', 'Stock', 'Current_Price', 'Stop_Loss', 'Stop_Loss_Pct', 'ATR_Pct']].head(35)
        
        return html.Div([
            # Header
            html.H3('⚠️  Risk Management', 
                   style={'color': colors['accent'], 'marginBottom': '30px'}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H4('Portfolio Volatility', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{portfolio_volatility:.1f}%", style={'color': colors['accent'], 'margin': '0'}),
                    html.P('Risk-weighted', style={'color': colors['text'], 'marginTop': '5px', 'fontSize': '14px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('High Risk Stocks', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{high_risk_count}", style={'color': colors['red'] if high_risk_count > 10 else colors['text'], 'margin': '0'}),
                    html.P(f"{high_risk_count/len(df_risk)*100:.0f}% of portfolio", style={'color': colors['text'], 'marginTop': '5px', 'fontSize': '14px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('Risk Warnings', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{warnings_count}", style={'color': colors['red'] if warnings_count > 0 else colors['green'], 'margin': '0'}),
                    html.P('Portfolio alerts', style={'color': colors['text'], 'marginTop': '5px', 'fontSize': '14px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.H4('Earnings Warnings', style={'color': colors['text'], 'marginBottom': '10px'}),
                    html.H2(f"{earnings_warnings}", style={'color': colors['red'] if earnings_warnings > 0 else colors['green'], 'margin': '0'}),
                    html.P('Next 5 days', style={'color': colors['text'], 'marginTop': '5px', 'fontSize': '14px'})
                ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                         'width': '23%', 'display': 'inline-block'}),
            ], style={'marginBottom': '30px'}),
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_risk_dist, style={'height': '400px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
                
                html.Div([
                    dcc.Graph(figure=fig_vol_sector, style={'height': '400px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
            ], style={'marginBottom': '30px'}),
            
            # High risk stocks table
            html.Div([
                html.H4('⚠️  Top 10 Highest Risk Stocks', style={'color': colors['red'], 'marginBottom': '15px'}),
                dash_table.DataTable(
                    data=df_high_risk.to_dict('records'),
                    columns=[
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Sector', 'id': 'Sector'},
                        {'name': 'Risk Score', 'id': 'Risk_Score', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                        {'name': 'Volatility %', 'id': 'Volatility_Annual', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Max Drawdown %', 'id': 'Max_Drawdown', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': f"1px solid {colors['card']}"
                    },
                    style_header={
                        'backgroundColor': colors['card'],
                        'fontWeight': 'bold',
                        'border': f"1px solid {colors['accent']}"
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Risk_Score', 'filter_query': '{Risk_Score} > 75'},
                            'backgroundColor': '#ff444420',
                            'color': colors['red'],
                            'fontWeight': 'bold'
                        },
                    ]
                ),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
            
            # Stop losses table
            html.Div([
                html.H4('🛑 Recommended Stop-Loss Levels', style={'color': colors['accent'], 'marginBottom': '15px'}),
                html.P('Based on 2x ATR (Average True Range)', style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '14px'}),
                dash_table.DataTable(
                    data=df_stops_display.to_dict('records'),
                    columns=[
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Stock', 'id': 'Stock'},
                        {'name': 'Current Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Stop Loss', 'id': 'Stop_Loss', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Stop %', 'id': 'Stop_Loss_Pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'ATR %', 'id': 'ATR_Pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    ],
                    style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
                    style_cell={
                        'backgroundColor': colors['background'],
                        'color': colors['text'],
                        'textAlign': 'left',
                        'padding': '10px',
                        'border': f"1px solid {colors['card']}"
                    },
                    style_header={
                        'backgroundColor': colors['card'],
                        'fontWeight': 'bold',
                        'border': f"1px solid {colors['accent']}"
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Stop_Loss_Pct', 'filter_query': '{Stop_Loss_Pct} < -10'},
                            'backgroundColor': '#ff444420',
                            'color': colors['red']
                        },
                    ],
                    page_size=20
                ),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
        ])

    elif tab == 'performance':
        # Load performance data
        perf_log_exists = os.path.exists('data/predictions_log.csv')
        perf_summary_exists = os.path.exists('data/performance_summary.csv')
        
        if not perf_log_exists:
            return html.Div([
                html.H3('⚠️  No Performance Data Available', 
                       style={'textAlign': 'center', 'color': colors['red'], 'marginTop': '50px'}),
                html.P('Run: python track_performance.py', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '16px'}),
                html.P('Performance tracking requires at least 1-day of predictions to evaluate.', 
                      style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '14px', 'fontStyle': 'italic'})
            ])
        
        df_log = pd.read_csv('data/predictions_log.csv')
        df_log['prediction_date'] = pd.to_datetime(df_log['prediction_date'])
        
        # Summary stats
        num_predictions = len(df_log)
        num_dates = df_log['prediction_date'].nunique()
        date_range = f"{df_log['prediction_date'].min().date()} to {df_log['prediction_date'].max().date()}"
        
        # Load performance summary if available
        if perf_summary_exists:
            df_perf = pd.read_csv('data/performance_summary.csv')
            
            # Latest metrics by horizon
            latest_metrics = df_perf.groupby('horizon').last()
            
            # Create accuracy trend chart
            fig_acc_trend = go.Figure()
            for horizon in ['1d', '5d', '21d']:
                horizon_data = df_perf[df_perf['horizon'] == horizon]
                if not horizon_data.empty:
                    fig_acc_trend.add_trace(go.Scatter(
                        x=horizon_data['evaluation_date'],
                        y=horizon_data['accuracy'] * 100,
                        mode='lines+markers',
                        name=horizon,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
            
            fig_acc_trend.update_layout(
                title='Prediction Accuracy Over Time',
                xaxis_title='Evaluation Date',
                yaxis_title='Accuracy (%)',
                template='plotly_dark',
                paper_bgcolor=colors['card'],
                plot_bgcolor=colors['background'],
                font=dict(color=colors['text']),
                hovermode='x unified',
                yaxis=dict(range=[0, 100])
            )
            
            acc_chart = html.Div([
                dcc.Graph(figure=fig_acc_trend, style={'height': '400px'}),
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'})
            
            # Metrics cards
            metrics_cards = []
            for horizon in ['1d', '5d', '21d']:
                if horizon in latest_metrics.index:
                    metrics = latest_metrics.loc[horizon]
                    metrics_cards.append(
                        html.Div([
                            html.H4(f'{horizon} Predictions', style={'color': colors['text'], 'marginBottom': '15px'}),
                            html.Div([
                                html.P('Accuracy', style={'color': colors['text'], 'fontSize': '12px', 'margin': '0'}),
                                html.H3(f"{metrics['accuracy']:.1%}", 
                                       style={'color': colors['accent'], 'margin': '5px 0'}),
                            ]),
                            html.Div([
                                html.P(f"UP: {metrics['up_accuracy']:.1%}", 
                                      style={'color': colors['green'], 'fontSize': '14px', 'margin': '5px 0'}),
                                html.P(f"DOWN: {metrics['down_accuracy']:.1%}", 
                                      style={'color': colors['red'], 'fontSize': '14px', 'margin': '5px 0'}),
                            ]),
                            html.Div([
                                html.P(f"Avg Return: {metrics['avg_return']:+.2%}", 
                                      style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '10px'}),
                                html.P(f"{int(metrics['total_predictions'])} predictions", 
                                      style={'color': colors['text'], 'fontSize': '12px', 'fontStyle': 'italic'})
                            ])
                        ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 
                                 'width': '31%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'})
                    )
        else:
            acc_chart = html.Div([
                html.P('⏳ Waiting for predictions to mature (need 1-21 days)', 
                      style={'textAlign': 'center', 'color': colors['text'], 'padding': '40px', 'fontSize': '16px'})
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'})
            metrics_cards = []
        
        return html.Div([
            # Header
            html.Div([
                html.H3('📈 Prediction Performance Tracking', 
                       style={'color': colors['accent'], 'display': 'inline-block', 'marginRight': '20px'}),
                html.P(f"{num_predictions} predictions logged across {num_dates} dates ({date_range})", 
                      style={'color': colors['text'], 'display': 'inline-block', 'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Metrics cards
            html.Div(metrics_cards, style={'marginBottom': '30px'}) if metrics_cards else html.Div(),
            
            # Accuracy trend chart
            acc_chart,
            
            # Predictions log table
            html.Div([
                html.H4('📝 Recent Predictions Log', style={'color': colors['accent'], 'marginBottom': '15px'}),
                dash_table.DataTable(
                    columns=[
                        {'name': 'Date', 'id': 'prediction_date'},
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                        {'name': '1d Pred', 'id': 'd1_Direction'},
                        {'name': '1d Prob', 'id': 'd1_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': '5d Pred', 'id': 'd5_Direction'},
                        {'name': '5d Prob', 'id': 'd5_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': '21d Pred', 'id': 'd21_Direction'},
                        {'name': '21d Prob', 'id': 'd21_Prob_Up', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                    ],
                    data=df_log.tail(50).to_dict('records'),
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
                        'textAlign': 'center',
                        'padding': '10px',
                    },
                    page_size=20,
                ),
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
        stocks = df_predictions[['Stock', 'Ticker']].drop_duplicates()
    else:
        stocks = df_predictions[df_predictions['Sector'] == selected_sector][['Stock', 'Ticker']].drop_duplicates()
    
    options = [{'label': f"{row['Stock']} ({row['Ticker']})", 'value': row['Ticker']} for _, row in stocks.iterrows()]
    default_value = options[0]['value'] if options else None
    
    return options, default_value

# Callback to handle table row clicks and update filters
@app.callback(
    Output('sector-dropdown', 'value'),
    Output('stock-dropdown', 'value', allow_duplicate=True),
    Input('stocks-table', 'active_cell'),
    State('stocks-table', 'data'),
    prevent_initial_call=True
)
def update_filters_from_table_click(active_cell, table_data):
    if active_cell is None or table_data is None:
        raise PreventUpdate
    
    # Get the clicked row
    row_index = active_cell['row']
    clicked_row = table_data[row_index]
    
    # Extract ticker and sector from the clicked row
    ticker = clicked_row.get('Ticker')
    sector = clicked_row.get('Sector')
    
    if ticker and sector:
        return sector, ticker
    
    raise PreventUpdate

# Callback to filter table by selected sector
@app.callback(
    Output('stocks-table', 'data'),
    Input('sector-dropdown', 'value')
)
def update_stocks_table(selected_sector):
    df_source = df_recommendations if has_recommendations else df_predictions

    if selected_sector and selected_sector != 'ALL':
        df_filtered = df_source[df_source['Sector'] == selected_sector]
    else:
        df_filtered = df_source

    return df_filtered.to_dict('records')

# Callback to update all charts and tables
@app.callback(
    Output('candlestick-chart', 'figure'),
    Output('volume-chart', 'figure'),
    Output('kpi-cards', 'children'),
    Input('stock-dropdown', 'value'),
    Input('period-dropdown', 'value'),
    Input('interval-dropdown', 'value')
)
def update_charts(selected_ticker, period_days, interval):
    if not selected_ticker:
        return {}, {}, []
    
    # Set defaults if None
    if period_days is None:
        period_days = 180
    if interval is None:
        interval = 'D'
    
    # Filter data for selected stock
    df_stock = df_stocks[df_stocks['Ticker'] == selected_ticker].copy()
    
    # Ensure index is DatetimeIndex and handle timezone issues
    if not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index, utc=True)
    else:
        # Remove timezone info if present to avoid conversion issues
        if df_stock.index.tz is not None:
            df_stock.index = df_stock.index.tz_localize(None)
    
    if df_stock.empty:
        # Get stock name from predictions if available
        pred_row = df_predictions[df_predictions['Ticker'] == selected_ticker]
        stock_name = pred_row['Stock'].iloc[0] if not pred_row.empty else selected_ticker
        
        # Create empty figure with message
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"📊 No historical price data available for {stock_name} ({selected_ticker})<br><br>Run: python download_stock_data.py",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text']),
            align='center'
        )
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background'],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        # Create message card
        message_card = html.Div([
            html.H3('⚠️ No Price Data Available', style={'color': colors['red'], 'textAlign': 'center'}),
            html.P(f'Historical price data for {stock_name} ({selected_ticker}) has not been downloaded yet.',
                   style={'color': colors['text'], 'textAlign': 'center', 'marginTop': '20px'}),
            html.P('Predictions are available, but charts require price history.',
                   style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '14px'}),
        ], style={'backgroundColor': colors['background'], 'padding': '40px', 'borderRadius': '10px', 'textAlign': 'center'})
        
        return empty_fig, empty_fig, message_card
    
    # Get prediction data
    pred_rows = df_predictions[df_predictions['Ticker'] == selected_ticker]
    if pred_rows.empty:
        pred_data = None
    else:
        pred_data = pred_rows.iloc[0]
    
    # Filter by period
    if period_days < 9999:
        df_stock = df_stock.tail(period_days)
    
    # Check if we have data after filtering
    if df_stock.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"No data available for the selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text'])
        )
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['background']
        )
        return empty_fig, empty_fig, []
    
    stock_name = df_stock['Stock'].iloc[0]
    sector = df_stock['Sector'].iloc[0]
    
    # Resample data based on interval
    print(f"[DEBUG] Resampling {selected_ticker} to interval: {interval}")
    try:
        if interval == 'W':
            # Weekly resampling
            print(f"[DEBUG] Before weekly resample: {len(df_stock)} rows")
            df_resampled = df_stock.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            print(f"[DEBUG] After weekly resample: {len(df_resampled)} rows")
            # Preserve metadata
            df_resampled['Stock'] = stock_name
            df_resampled['Ticker'] = selected_ticker
            df_resampled['Sector'] = sector
            df_stock = df_resampled
        elif interval == 'M':
            # Monthly resampling
            print(f"[DEBUG] Before monthly resample: {len(df_stock)} rows")
            df_resampled = df_stock.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            print(f"[DEBUG] After monthly resample: {len(df_resampled)} rows")
            # Preserve metadata
            df_resampled['Stock'] = stock_name
            df_resampled['Ticker'] = selected_ticker
            df_resampled['Sector'] = sector
            df_stock = df_resampled
        # else: interval == 'D', use daily data as-is
        
        # Verify we still have data after resampling
        if df_stock.empty:
            raise ValueError("No data after resampling")
            
    except Exception as e:
        # If resampling fails, fall back to daily data
        print(f"[ERROR] Resampling error for {selected_ticker}: {e}")
        df_stock = df_stocks[df_stocks['Ticker'] == selected_ticker].copy()
        if period_days < 9999:
            df_stock = df_stock.tail(period_days)
        interval = 'D'  # Reset to daily
    
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
    
    # Get interval label for chart title
    interval_labels = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
    interval_label = interval_labels.get(interval, 'Daily')
    
    fig_candle.update_layout(
        title=f'{stock_name} ({selected_ticker}) - {sector} [{interval_label}]',
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
        
        # Period Low & AI Prediction Combined Card (side by side)
        html.Div([
            # Left side - Period Low
            html.Div([
                html.H4('Period Low', style={'color': colors['text'], 'marginBottom': '5px'}),
                html.H2(f'${period_low:.2f}', style={'color': colors['red'], 'margin': '0'}),
                html.P(f'{((period_low/latest_price - 1) * 100):.1f}%', 
                       style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '5px'})
            ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'width': '35%', 'textAlign': 'center'}),
            
            # Divider
            html.Div(style={'display': 'inline-block', 'width': '1px', 'height': '60px', 'backgroundColor': colors['text'], 
                           'opacity': '0.3', 'verticalAlign': 'middle', 'margin': '0 10px'}),
            
            # Right side - AI Prediction (one line)
            html.Div([
                html.H4('AI Prediction', style={'color': colors['text'], 'marginBottom': '8px', 'fontSize': '14px'}),
                html.Div([
                    html.Span('1d: ', style={'color': colors['text'], 'fontSize': '11px'}),
                    html.Span(pred_data.get('d1_Direction', 'N/A') if pred_data is not None else 'N/A', 
                             style={'color': colors['green'] if pred_data is not None and 'UP' in str(pred_data.get('d1_Direction', '')) else colors['red'],
                                    'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(f" {pred_data.get('d1_Prob_Up', 0):.0%}" if pred_data is not None else '', 
                             style={'color': colors['text'], 'fontSize': '10px'}),
                    html.Br(),
                    
                    html.Span('5d: ', style={'color': colors['text'], 'fontSize': '11px'}),
                    html.Span(pred_data.get('d5_Direction', 'N/A') if pred_data is not None else 'N/A', 
                             style={'color': colors['green'] if pred_data is not None and 'UP' in str(pred_data.get('d5_Direction', '')) else colors['red'],
                                    'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(f" {pred_data.get('d5_Prob_Up', 0):.0%}" if pred_data is not None else '', 
                             style={'color': colors['text'], 'fontSize': '10px'}),
                    html.Br(),
                    
                    html.Span('21d: ', style={'color': colors['text'], 'fontSize': '11px'}),
                    html.Span(pred_data.get('d21_Direction', 'N/A') if pred_data is not None else 'N/A', 
                             style={'color': colors['green'] if pred_data is not None and 'UP' in str(pred_data.get('d21_Direction', '')) else colors['red'],
                                    'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(f" {pred_data.get('d21_Prob_Up', 0):.0%}" if pred_data is not None else '', 
                             style={'color': colors['text'], 'fontSize': '10px'}),
                ]),
                html.P(f"Acc: {pred_data.get('d21_Accuracy', pred_data.get('d1_Accuracy', 0)):.0%}" if is_refined and pred_data is not None else '',
                       style={'color': colors['accent'], 'fontSize': '11px', 'marginTop': '5px'})
            ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'width': '55%', 'textAlign': 'left'}),
        ], style={
            'backgroundColor': colors['background'], 
            'padding': '20px', 
            'borderRadius': '10px', 
            'width': '31%', 
            'display': 'inline-block',
            'textAlign': 'center'
        }),
    ])
    
    # predictions_table removed - predictions now shown in KPI cards
    
    return fig_candle, fig_volume, kpi_cards

# Callback to update stock dropdown based on selected sector (Sentiment tab)
@app.callback(
    Output('sentiment-stock-filter', 'options'),
    Output('sentiment-stock-filter', 'value'),
    Input('sentiment-sector-filter', 'value')
)
def update_sentiment_stock_options(selected_sector):
    # Load sentiment data
    sentiment_complete_exists = os.path.exists('data/sentiment_history_complete.csv')
    sentiment_history_exists = os.path.exists('data/sentiment_history.csv')
    
    if sentiment_complete_exists:
        df_sent = pd.read_csv('data/sentiment_history_complete.csv')
    elif sentiment_history_exists:
        df_sent = pd.read_csv('data/sentiment_history.csv')
    else:
        return [{'label': 'All Stocks', 'value': 'All Stocks'}], 'All Stocks'
    
    # Load sector info
    df_stocks_info = pd.read_csv('data/multi_sector_stocks.csv')
    df_stocks_info = df_stocks_info[['Ticker', 'Sector']].drop_duplicates()
    df_sent_with_sector = df_sent.merge(df_stocks_info, left_on='ticker', right_on='Ticker', how='left')
    
    # Filter stocks by sector
    if selected_sector and selected_sector != 'All Sectors':
        available_stocks = sorted(df_sent_with_sector[df_sent_with_sector['Sector'] == selected_sector]['ticker'].unique().tolist())
    else:
        available_stocks = sorted(df_sent_with_sector['ticker'].unique().tolist())
    
    options = [{'label': 'All Stocks', 'value': 'All Stocks'}] + [{'label': s, 'value': s} for s in available_stocks]
    
    return options, 'All Stocks'

# Callback for sentiment timeline chart filtering
@app.callback(
    Output('sentiment-timeline-chart', 'figure'),
    Input('sentiment-sector-filter', 'value'),
    Input('sentiment-stock-filter', 'value'),
    Input('sentiment-view-mode', 'value')
)
def update_sentiment_timeline(selected_sector, selected_stock, view_mode):
    # Load sentiment data
    sentiment_complete_exists = os.path.exists('data/sentiment_history_complete.csv')
    sentiment_history_exists = os.path.exists('data/sentiment_history.csv')
    
    if sentiment_complete_exists:
        df_sent = pd.read_csv('data/sentiment_history_complete.csv')
    elif sentiment_history_exists:
        df_sent = pd.read_csv('data/sentiment_history.csv')
    else:
        return {}
    
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    
    # Load sector info
    df_stocks_info = pd.read_csv('data/multi_sector_stocks.csv')
    df_stocks_info = df_stocks_info[['Ticker', 'Sector']].drop_duplicates()
    df_sent_with_sector = df_sent.merge(df_stocks_info, left_on='ticker', right_on='Ticker', how='left')
    
    # Apply filters
    df_filtered = df_sent_with_sector.copy()
    
    if selected_stock and selected_stock != 'All Stocks':
        df_filtered = df_filtered[df_filtered['ticker'] == selected_stock]
        title = f'Sentiment Trend Over Time - {selected_stock}'
    elif selected_sector and selected_sector != 'All Sectors':
        df_filtered = df_filtered[df_filtered['Sector'] == selected_sector]
        title = f'Sentiment Trend Over Time - {selected_sector} Sector'
    else:
        title = 'Sentiment Trend Over Time (All Stocks Average)'
    
    # Aggregate by date
    daily_sentiment = df_filtered.groupby('date')['sentiment_compound'].mean().reset_index()
    daily_sentiment = daily_sentiment.sort_values('date')
    
    # Apply rolling average if selected
    if view_mode == 'rolling_7':
        daily_sentiment['sentiment_compound'] = daily_sentiment['sentiment_compound'].rolling(window=7, min_periods=1).mean()
        title += ' (7-Day Rolling Average)'
    elif view_mode == 'rolling_30':
        daily_sentiment['sentiment_compound'] = daily_sentiment['sentiment_compound'].rolling(window=30, min_periods=1).mean()
        title += ' (30-Day Rolling Average)'
    
    # Create figure
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['sentiment_compound'],
        mode='lines+markers',
        line=dict(color=colors['accent'], width=2),
        marker=dict(size=4 if view_mode == 'daily' else 3),
        name='Avg Sentiment'
    ))
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig_timeline.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Average Sentiment',
        template='plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        hovermode='x unified'
    )
    
    return fig_timeline

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
