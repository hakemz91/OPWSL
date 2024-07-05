# OPWSL
Improved version of NuWSL with minor but life changing feature updates. It is the same chatbot which can talk to your documents and retrieve information very fast (3 to 15 seconds depending on your question(RTX 3060 12 GB), even for gigabytes of database, ), but only for NVIDIA GPU user and WSL2 Windows. However, it may working fine for Ubuntu or Linux too since WSL is just Ubuntu/Linux.

## Features
1. New! Auto detect the same documents when ingesting so no duplicate ingestion.
2. Easy to use with simple options.
3. Option to auto shutdown the PC after the ingestion is done. (useful for overnight bulk file ingestion) 
4. Simple logging of ingestion start and finish time for both auto shutdown and non-auto ingestion.
5. Option to run the chat and saving the chat history (Q&A pairs) both into csv and txt files.
6. A bit of colored texts for easy reading.
7. Displaying time taken for generating response.

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

10. Done installation and then just close cmd console. Usage as below:

## How to use
1. First put all your pdf, docs, txt and etc. (supported) documents in SOURCE DOCEUMENTS folder, start the launcher and choose Option 4 to ingest your documents. First time run it will download the embedding model so will take some time.
2. Then, run option 1 and let it finish downloading the chat model. It will take some times depending on your internet speed. When it finish, you can chat with that, or type exit, answer y and choose the options again.
3. MOST IMPORTANT please note that if you have DB database folder backup from NuWSL, it will not work with this OPWSL ingest script. So unfortunately you have to re-ingest everything.

## Export to .CSV and .TXT

Choose option 3 to run the chat so that it will save the chat history both into csv and txt files. A folder called local_chat_history will be created for the first run of the option. However, note that over time, the txt file can be so huge like maybe thousands of histories, that it will be slow to open the file. So you can occasionally delete that file or the entire local_chat_history folder and it will be rebuilded next time you run the chat with option 3.

## Ingestion Time Logging

After every ingestion for both without and with auto shutdown system, an Ingestion_time_log.txt file will be updated. So the txt file will contain the start and finish date of latest run process.

## Ingested Files Logging and Duplicate Files Detection

Each time you run the ingestion, a list of ingested file names are logged into file_done_ingested.log file. This is useful if you forgot whether you ingested certain files already or not. However, again, note that over time, the list can be so huge like maybe thousands, that it will be slow to open the file. So you can delete that file occasionally and it will be rebuilded next time you ingest files. However, when you delete that and accidentaly ingest the same files, it will not detect that and will reingest it again. The logs are in "file_done_ingested.log" and "duplicate_not_ingested.log" files. To reingest the same thing, delete these two .log files.

## How to Reset the Vector Database

Just delete the DB folder and reingesting back using option 4 or 5.

## How to Change Model

The default model is the one that is working right now. Somehow if using another 4 bit GPTQ quantized model, it will have problem. But the current model is smart enough for talking with your document and probably overkill. There is a model like Qwen2 1.5B that is very much lighter and faster but still good enough for talking with documents. Maybe in future I will fix that.

## Forked from awesome original LocalGPT
https://github.com/PromtEngineer/localGPT
