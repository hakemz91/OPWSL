# OPWSL
Improved version of NuWSL with minor but life changing feature updates.

## Features

1. Now have GUI option (but I don't recommend this because cmd console is much faster, clearer and simpler. Why makes our life harder? :D ).
2. Easy to use with simple options.
3. Option to auto shutdown the PC after the ingestion is done. (useful for overnight bulk file ingestion) 
4. Simple logging of ingestion start and finish time.
5. Option to run the chat and saving the chat history (Q&A pairs) both into csv and txt files (localGPT only save into csv).
6. A bit of colored texts for easy reading.
7. Displaying time taken for generating response.
8. Auto detect the same documents when ingesting so no duplicate ingestion.

![Alt text](https://github.com/hakemz91/NuWSL/blob/main/01_im.png)

![Alt text](https://github.com/hakemz91/NuWSL/blob/main/02_im.png)

![Alt text](https://github.com/hakemz91/NuWSL/blob/main/03_im.png)

## How to Install

1. Git clone this repo to anywhere in your windows.

2. Go to the link at the end of this sentence and download the installer Anaconda3-2023.07-2-Linux-x86_64.sh file. I don't recommmend download the latest version since it using python3.11 and will have problem, unless you know how to install python3.11. So for the peace of mind just use link here: (https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh). You then place the Anaconda3-2023.07-2-Linux-x86_64.sh file into folder called "anaconda_installer" in the repo that you had clone before.

3. Enable the WSL for your Windows and use command the command below to update the wsl kernel to wsl2 (older wsl kernel might work and I am not tested it yet):

```
wsl --update
```

4. Install your Ubuntu distro from Microsoft Store and after that opened it and fill the username and password. After everything set up, close your wsl windows. Then get your Ubuntu distro name by using below command in normal cmd:

```
wsl --list
```

for example my distro id name is Ubuntu-22.04
Keep this for later instruction

5. Right click the launcher.bat and replace the distro id name to the one you used and saved it, for example in this case is Ubuntu-22.04

6. Lauch the launcher.bat file, answer y and it will automatically enter the for example Ubuntu-22.04

7. Now in the WSL windows, run the below command one at a time:

```
sudo apt update -y
sudo apt upgrade -y
sudo apt install build-essential -y
cd anaconda_installer
bash Anaconda3-2023.07-2-Linux-x86_64.sh
```

Then you will need to press enter a lot in order to proceed downwards and need to accept yes for license term in order to install anaconda. If you just accidently skipped license term confirmation, just run the above command again to enter the installation windows (bash Anaconda3-2023.07-2-Linux-x86_64.sh). Then just type yes and enter, and enter again to proceed with installation. And after that it will ask "Do you wish the installer to initialize Anaconda3" so just type yes again and enter.

8. After anaconda is installed, close the wsl windows and then launch the launcher.bat again to enter it. It will now enter (base) anaconda environment. 

9. Now run the below command one at a time:

```
conda create -n OPWSL python=3.10.0 -y

python update_bashrc.py

conda activate OPWSL

pip install -r requirements.txt

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

10. Done install! Now just close the wsl windows and then lauch again the launcher.bat and answer y. Choose which option number you want and then type it and press enter. First run will take some time because of downloading the model. After finish the selected option processing, it will ask if you want to run another option. If you do, let say after ingestion you want to run the chatting, just answer y and press enter and it will ask you again the available option.

## Export to .CSV and .TXT

Choose option 3 to run the chat so that it will save the chat history both into csv and txt files. A folder called local_chat_history will be created for the first run of the option. However, note that over time, the txt file can be so huge like maybe thousands of histories, that it will be slow to open the file. So you can occasionally delete that file or the entire local_chat_history folder and it will be rebuilded next time you run the chat with option 3.

## Ingestion Time Logging

After every ingestion for both without and with auto shutdown system, an Ingestion_time_log.txt file will be updated. So the txt file will contain the start and finish date of latest run process.

## Ingested Files Logging

Each time you run the ingestion, a list of ingested file names are logged into file_done_ingested.log file. This is useful if you forgot whether you ingested certain files already or not. However, again, note that over time, the list can be so huge like maybe thousands, that it will be slow to open the file. So you can delete that file occasionally and it will be rebuilded next time you ingest files.

## How to Reset the Vector Database

Just delete the DB folder and reingesting back using option 4 or 5.

## How to Change Model

The default model is the best model already for retrieving information. However if you want to change, just replace the model id in constants.py with the one you get from TheBloke ( https://huggingface.co/TheBloke ) but must be 4 bit quantized, GPTQ type, and Mistral based model. I do not support other type of model format.

## Forked from awesome original LocalGPT
https://github.com/PromtEngineer/localGPT
