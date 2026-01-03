# Automation Schedule

This document explains the automated workflows for the stock prediction system.

## GitHub Actions Workflow

The system automatically updates daily via GitHub Actions (`.github/workflows/daily_update.yml`).

### Schedule

- **Daily Updates**: 6:00 PM UTC (after market close)
  - Downloads latest stock data
  - Generates trading signals
  - Creates daily report
  
- **Weekly Retraining**: Every Sunday
  - Retrains all prediction models
  - Runs comprehensive backtests
  - Updates performance metrics

### Manual Trigger

You can manually trigger the workflow:
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "Daily Stock Data Update & Model Training"
4. Click "Run workflow"

## Local Testing

Test the automation pipeline locally:

```bash
# 1. Download latest data
python download_stock_data.py

# 2. Generate signals
python generate_daily_signals.py

# 3. Create report
python generate_report.py

# 4. (Weekly) Retrain models
python predict_stock.py

# 5. (Weekly) Run backtests
python backtest_trading.py 2
```

## Automated Files

The following files are automatically updated and committed:

- `data/multi_sector_stocks.csv` - Latest stock data (daily)
- `predictions_summary.csv` - Model predictions (weekly)
- `daily_signals.csv` - Trading signals (daily)
- `backtest_summary.csv` - Backtest results (weekly)
- `models/*.pkl` - Trained models (weekly)
- `DAILY_REPORT.md` - Daily summary report (daily)

## Notifications

- **Success**: Updated files are automatically committed to the repository
- **Failure**: An issue is automatically created in the repository

## Setup Requirements

1. Ensure GitHub Actions are enabled for your repository
2. No additional secrets required (uses default `GITHUB_TOKEN`)
3. The workflow runs on Ubuntu with Python 3.12

## Monitoring

Check workflow status:
- View runs: `https://github.com/[owner]/[repo]/actions`
- Check issues: `https://github.com/[owner]/[repo]/issues`
- Review commits: Look for commits from `github-actions[bot]`

## Resource Usage

- Average runtime: 5-10 minutes (daily updates)
- Average runtime: 30-45 minutes (weekly with retraining)
- GitHub Actions free tier: 2,000 minutes/month (sufficient for this workflow)

## Customization

Edit `.github/workflows/daily_update.yml` to:
- Change schedule (modify cron expression)
- Adjust thresholds in signal generation
- Add email notifications
- Configure Slack/Discord webhooks
