import subprocess
import time
import re

N = 8

# Command to run the RL environment script
command_rl = ["python", "test_loader_subprocess.py"]

# Execute the RL environment script and capture the stdout using subprocess.run
process_rl = subprocess.run(command_rl, capture_output=True, text=True)

text = process_rl.stdout.splitlines()
print(text)

first_line_counter, last_line_counter = None, None
for i in range(len(text)):
    if "Wrapping the env in a DummyVecEnv" in text[i]:
        first_line_counter = i
 #   if "ActiveThreads" in text[i]:
    if "argv[0]" in text[i]:
        last_line_counter = i
        break
        

action_arrays = text[first_line_counter+1:last_line_counter]

print("Action array")

final_result = []
for i in range(len(action_arrays)):
    print(action_arrays[1])
    if i % 2 == 0:
        first_part = re.sub("\s+", ",", action_arrays[i].strip())
        first_part = first_part[0] + first_part[2:]
        print("first part", first_part)
    else:
        second_part = ',' + action_arrays[i].strip()
        print("second part", second_part)
        combined_part = first_part + second_part
        print(combined_part)
        final_result.append(combined_part)
print(final_result)


for i in final_result:
    # Construct the command to execute based on the current line
    command = ["rostopic pub", f"-1 /joint_position_example_controller/command /std_msgs/Float64MultiArray 'data:{i}'"]

    # Execute the command using subprocess.run
    subprocess.run(command)

    # Add a delay of N seconds between each line
    time.sleep(N)
