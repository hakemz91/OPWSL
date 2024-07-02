import datetime
import os

# Get the current start date and time
current_datetime = datetime.datetime.now()

# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Create the output string
output_text = f"Start: {formatted_datetime}"

# Print the output to the console
print(output_text)

# Get the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create the path for the output file
output_file_path = os.path.join(script_directory, "start.txt")

# Write the output to the file
with open(output_file_path, "w") as file:
    file.write(output_text)
    
#end
