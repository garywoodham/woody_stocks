#!/usr/bin/env python3
"""
Run sentiment backfill with minute-by-minute progress updates.
"""

import subprocess
import sys
import time
import threading
from datetime import datetime

def run_with_updates():
    """Run backfill and provide updates every minute"""
    
    print("="*80)
    print(f"STARTING SENTIMENT BACKFILL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("This will backfill 10 years of sentiment data for missing stocks...")
    print("Updates will be shown every minute.")
    print("="*80)
    print()
    
    # Start the backfill process
    process = subprocess.Popen(
        ['python3', 'backfill_full_sentiment_history.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    last_update = time.time()
    line_count = 0
    
    # Read output line by line
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
            sys.stdout.flush()
            line_count += 1
            
            # Show progress update every minute
            current_time = time.time()
            if current_time - last_update >= 60:
                print()
                print("="*80)
                print(f"⏱️  UPDATE: Still running... ({line_count} lines processed)")
                print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                print("="*80)
                print()
                last_update = current_time
    
    # Wait for process to complete
    process.wait()
    
    print()
    print("="*80)
    if process.returncode == 0:
        print("✅ BACKFILL COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️  BACKFILL EXITED WITH CODE: {process.returncode}")
    print(f"Total lines processed: {line_count}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return process.returncode

if __name__ == "__main__":
    exit_code = run_with_updates()
    sys.exit(exit_code)
