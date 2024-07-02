import os

# Read content from start.txt
with open('start.txt', 'r') as start_file:
    start_content = start_file.read().strip()

# Read content from finish.txt
with open('finish.txt', 'r') as finish_file:
    finish_content = finish_file.read().strip()

# Combine content with space between them
combined_content = f"{start_content}\n\n{finish_content}"

# Write the combined content to a new file named Ingestion_time_log.txt
with open('Ingestion_time_log.txt', 'w') as combined_file:
    combined_file.write(combined_content)

print("Combining files complete. Check 'Ingestion_time_log.txt'.")

# Delete start.txt and finish.txt
try:
    os.remove('start.txt')
    os.remove('finish.txt')
    print("start.txt and finish.txt deleted.")
except FileNotFoundError:
    print("start.txt or finish.txt not found.")
except Exception as e:
    print(f"An error occurred while deleting files: {e}")
