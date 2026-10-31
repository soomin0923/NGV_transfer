from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(R/'tools/check_installation.py'),*sys.argv[1:]],check=True)
