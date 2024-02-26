import subprocess
import time

N = 5

# Command to run the RL environment script
command_rl = ["python", "test_loader_subprocess.py"]

# Execute the RL environment script and capture the stdout using subprocess.run
process_rl = subprocess.run(command_rl, capture_output=True, text=True)

text = process_rl.stdout.splitlines()

first_line_counter, last_line_counter = None, None
for i in range(len(text)):
    if "Wrapping the env in a DummyVecEnv" in text[i]:
        first_line_counter = i
    if "ActiveThreads" in text[i]:
        last_line_counter = i
        break
        
print(first_line_counter)
print(last_line_counter)

action_arrays = text[first_line_counter+1:last_line_counter]

print("Action array")
print(action_arrays)

# Iterate over the stdout line by line
# for line in process_rl.stdout.splitlines():
#     # Print the captured output
#     print(line.strip())

#     # Construct the command to execute based on the current line
#     command = ["mkdir", f"{line}"]

#     # Execute the command using subprocess.run
#     subprocess.run(command)

#     # Add a delay of N seconds between each line
#     time.sleep(N)
