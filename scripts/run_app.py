"""Lance l'application Streamlit Mon Petit Prono."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
APP  = ROOT / "src" / "app" / "main.py"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    subprocess.run(cmd, cwd=str(ROOT))
