import os
import csv
from datetime import datetime

def log_to_csv_and_txt(question, answer):
    log_dir = "local_chat_history"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file paths
    csv_file = "qa_log.csv"
    txt_file = "qa_log.txt"
    csv_path = os.path.join(log_dir, csv_file)
    txt_path = os.path.join(log_dir, txt_file)

    # Check if CSV file exists, if not create and write headers
    if not os.path.isfile(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "question", "answer"])

    # Check if TXT file exists, if not create and write headers
    if not os.path.isfile(txt_path):
        with open(txt_path, mode='w', encoding='utf-8') as txt_file:
            txt_file.write("timestamp\nquestion\nanswer\n\n")

    # Append the log entry to CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, question, answer])

    # Append the log entry to TXT with formatting
    with open(txt_path, mode='a', encoding='utf-8') as txt_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txt_file.write(f"{timestamp}\n{question}\n*{answer}*\n\n")

# Example usage:
#log_to_csv_and_txt("What is your name?", "My name is ChatGPT.")
#log_to_csv_and_txt("How does this work?", "This script logs Q&A pairs.")
