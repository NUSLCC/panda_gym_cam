import subprocess
import time
import re

N = 8

# Command to run the RL environment script
command_rl = ["python", "test_loader_subprocess.py"]

# Execute the RL environment script and capture the stdout using subprocess.run
process_rl = subprocess.run(command_rl, capture_output=True, text=True)

text = process_rl.stdout.splitlines()
#print(text)

first_line_counter, last_line_counter = None, None
for i in range(len(text)):
    if "Wrapping the env in a DummyVecEnv" in text[i]:
        first_line_counter = i
 #   if "ActiveThreads" in text[i]:
  #  if "argv[0]" in text[i]:
    if "Current joint" in text[i]:
        last_line_counter = i
        break
        

action_arrays = text[first_line_counter+1:last_line_counter]

#print("Action array")

final_result = []
for i in range(len(action_arrays)):
  # print(action_arrays[1])
    if i % 2 == 0:
        first_part = re.sub("\s+", ",", action_arrays[i].strip())
        first_part = first_part[0] + first_part[2:]
    #    print("first part", first_part)
    else:
        second_part = ',' + action_arrays[i].strip()
    #    print("second part", second_part)
        combined_part = first_part + second_part
      #  print(combined_part)
        final_result.append(combined_part)
#print("Result:", final_result)

actual_joints = []
multiplied_joints = []
neutral_joints = [0, 0.41, 0, -1.85, 0, 2.26, 0.79]
current_joints = neutral_joints.copy()

for element in final_result:
    # Convert string to list of floats
    float_list = [float(num) for num in element.strip('[]').split(',')]

    # Multiply each float by 0.05
    multiplied_list = [num * 0.05 for num in float_list] # panda-gym multiplies joint action by 0.05 to limit change in pose
    
    for i in range(len(current_joints)):
        current_joints[i] += multiplied_list[i]

    # Convert back to string
    multiplied_string= '[' + ','.join(map(str, multiplied_list)) + ']'
    multiplied_absolute_string = '[' + ','.join(map(str, current_joints)) + ']'

    # Append to result list
    actual_joints.append(multiplied_absolute_string)
    multiplied_joints.append(multiplied_string)

print("Actual joint list:", actual_joints)
#print("Modified action:", multiplied_joints)

for i in actual_joints:
    # Construct the command to execute based on the current line
    command = ["rostopic pub -1 /joint_position_example_controller/absolute_command /std_msgs/Float64MultiArray \"data: {i}\""]

    # Execute the command using subprocess.run
    subprocess.run(command, shell=True)

