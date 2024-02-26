import subprocess
import time

N = 5

# Command to run the RL environment script
command_rl = ["python", "test_loader.py"]

# Execute the RL environment script and capture the stdout using subprocess.run
process_rl = subprocess.run(command_rl, capture_output=True, text=True)

# Iterate over the stdout line by line
for line in process_rl.stdout.splitlines():
    # Print the captured output
    print(line.strip())

    # Construct the command to execute based on the current line
    command = ["mkdir", f"{line}"]

    # Execute the command using subprocess.run
    subprocess.run(command)

    # Add a delay of N seconds between each line
    time.sleep(N)
