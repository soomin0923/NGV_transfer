from pathlib import Path
import argparse,subprocess,sys
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tools',action='store_true');a=p.parse_args()
subprocess.run([sys.executable,str(R/'tools/check_installation.py'),'--role','tools' if a.tools else 'train'],check=True)
