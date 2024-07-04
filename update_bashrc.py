import os

# Path to the bashrc file
bashrc_path = os.path.expanduser("~/.bashrc")

# Code to be added to bashrc
bashrc_code = """
# Activate Conda environment
conda activate OPWSL

# Function to run commands
function run_commands() {

echo -e "\e[38;5;141m1. Chat with GUI Interface\e[0m"
echo -e "\e[38;5;214m2. Chat(No Sources)\e[0m"  
echo -e "\e[38;5;117m3. Chat(With Sources)\e[0m"
echo -e "\e[38;5;210m4. Chat(History saved as csv and txt)\e[0m"
echo -e "\e[38;5;118m5. Ingestion\e[0m"
echo -e "\e[38;5;227m6. Ingestion(auto shutdown PC after finish)\e[0m"

read -p "Enter the number of the option: " choice

case $choice in

1)
echo -e "\e[38;5;141mRunning Option 1: Chat with GUI Interface\e[0m"
python runGUI.py
;;

2)
echo -e "\e[38;5;214mRunning Option 2: Chat(No Sources)\e[0m"
python run.py
;;

3) 
echo -e "\e[38;5;117mRunning Option 3: Chat(With Sources)\e[0m"
python run.py --show_sources
;;

4)
echo -e "\e[38;5;210mRunning Option 4: Chat(History saved as csv and txt)\e[0m"
python run.py --save_qa
;;

5)
echo -e "\e[38;5;118mRunning Option 5: Ingestion\e[0m"  
python start_date.py && python ingest.py && python finish_date.py && python ingest_TL.py
;;

6)
echo -e "\e[38;5;227mRunning Option 6: Ingestion(auto shutdown PC after finish)\e[0m"
python start_date.py && python ingest_AS.py && python finish_date.py && python ingest_TL.py && python autoshut.py
;;

*)
echo "Invalid choice. No command will be executed."
;;

esac

read -p "Do you want to run another command? (y/n): " confirm 

if [ "$confirm" == "y" ]; then
run_commands  
else
echo "Exiting..."
fi

}

# Run the commands function
run_commands
"""

# Append code to bashrc
with open(bashrc_path, "a") as bashrc_file:
    bashrc_file.write(bashrc_code)

print("Code added to ~/.bashrc file.")