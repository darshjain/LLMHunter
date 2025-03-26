import subprocess
import sys


# run command huggingface-cli login
def run_huggingface_login():
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_huggingface_login()

