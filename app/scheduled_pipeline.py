import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime


base_path = Path(__file__).parent

ingest_path = base_path / "ingest.py"
analyze_path = base_path / "analyse.py"

RUN_EVERY_MINUTES = 60


def run_script(script_path):
    print(f"\nRunning {script_path.name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"{script_path.name} failed.")
        return False

    return True


def run_pipeline():
    print("=" * 60)
    print("Scheduled Financial News Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ingest_success = run_script(ingest_path)

    if not ingest_success:
        print("Pipeline stopped because ingestion failed.")
        return

    analyze_success = run_script(analyze_path)

    if not analyze_success:
        print("Pipeline stopped because analysis failed.")
        return

    print("=" * 60)
    print("Pipeline completed successfully.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


while True:
    run_pipeline()

    print(f"\nWaiting {RUN_EVERY_MINUTES} minutes until next run...")
    time.sleep(RUN_EVERY_MINUTES * 60)