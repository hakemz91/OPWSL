import os
import subprocess

def main():
    script_location = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_location, 'ingest_complete.txt')

    if os.path.isfile(file_path):
        print("Found ingest_complete.txt. Initiating shutdown...")
        shutdown_command = "/mnt/c/Windows/System32/shutdown.exe -s -t 120"
        subprocess.run(shutdown_command, shell=True)

        print("Deleting ingest_complete.txt...")
        os.remove(file_path)
        print("Shutdown initiated and file deleted.")
    else:
        print("No ingest_complete.txt found. Nothing to do.")

if __name__ == "__main__":
    main()
