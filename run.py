import os
import subprocess
import sys

def main():
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run Streamlit app
    os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")

if __name__ == "__main__":
    main()
