from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import subprocess

active_image = cv2.imread('a_Color.png')
static_image = cv2.imread('s_Color.png')

static_image = static_image[150:580,300:900,:]

resized_active_image = cv2.resize(active_image, (160,90))
resized_static_image = cv2.resize(static_image, (160,90))

final_image = np.concatenate([resized_active_image, resized_static_image], axis=-1)
final_image = np.transpose(final_image, (2, 0, 1))

# cv2.imshow('Final image', static_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#print("final img shape", final_image.shape)

observation = {
    "observation": final_image.astype(np.uint8),
    "desired_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "achieved_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "state": np.random.uniform(-10, 10, (10,)).astype(np.float32)
}

print("Observation: ", observation)

env = gym.make('PandaReachCamJoints-v3', render_mode = 'rgb_array', control_type="joints") # rgb_array
#print(env.action_space)
# HER must be loaded with the env
model = TQC.load("reach_blacktable_jitter", env=env)

obs, _ = env.reset() # don't use this obs 

action, _states = model.predict(observation, deterministic=True)

action_array = []

for i in action:
    action_array.append(i)

print("Original action array: ", action_array)

multiplied_array = np.array(action_array) * 0.05
#print("Modified action array: ", multiplied_array)
# 0, 0.41, 0, -1.85, 0, 2.26, 0.79
neutral_joints = np.array([0, 0.41, 0, -1.45, 0, 1.96, 0.79])
current_joints = np.array([0.165206640958786, 0.5149869647249579, 0.09249006677418947, -1.4322835529921576, 0.06880630273371935, 1.9900759675446897, 0.9239430361986161])
current_joints += multiplied_array

print(current_joints)

current_joints_array = []
for i in current_joints:
    current_joints_array.append(i)

print("\n \n \n")
print(current_joints_array)

# multiplied_absolute_joints_string = '[' + ','.join(map(str, current_joints_array)) + ']'
# print('Absolute command string', multiplied_absolute_joints_string[1])

command = [f'roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:={current_joints_array[0]} panda_joint2:={current_joints_array[1]} panda_joint3:={current_joints_array[2]} panda_joint4:={current_joints_array[3]} panda_joint5:={current_joints_array[4]} panda_joint6:={current_joints_array[5]} panda_joint7:={current_joints_array[6]} robot_ip:=172.16.0.2']
# command = ['cd ../../catkin_ws && ls'] # this works
print(command[0])
# subprocess.run(command, shell=True)

"""

Run(white table):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.0025436640717089176 panda_joint2:=0.4276746379584074 panda_joint3:=-0.006061297841370106 panda_joint4:=-1.8284860778599978 panda_joint5:=-0.0018731832969933748 panda_joint6:=2.2795764899253843 panda_joint7:=0.7887540423171595 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.000725782010704279 panda_joint2:=0.44734555467963216 panda_joint3:=-0.007726878160610795 panda_joint4:=-1.8138637294992805 panda_joint5:=-0.002335634868359193 panda_joint6:=2.293742219209671 panda_joint7:=0.7874582402501256 robot_ip:=172.16.0.2



Run12:
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.006171283312141895 panda_joint2:=0.43915212512016294 panda_joint3:=-0.0120200514793396 panda_joint4:=-1.8681038435548545 panda_joint5:=0.0011843085521832108 panda_joint6:=2.248720154389739 panda_joint7:=0.8008105065301061 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.009504452580586076 panda_joint2:=0.4663049670308828 panda_joint3:=-0.021082273684442043 panda_joint4:=-1.8848619483411313 panda_joint5:=0.002312058233655989 panda_joint6:=2.238604300878942 panda_joint7:=0.8128793742135167 robot_ip:=172.16.0.2



Run11 (works):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.004554045386612415 panda_joint2:=0.4328426586836576 panda_joint3:=7.992983228177764e-06 panda_joint4:=-1.85413935771212 panda_joint5:=0.002963936422020197 panda_joint6:=2.2587057894468305 panda_joint7:=0.8002302972227335 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.010792607441544533 panda_joint2:=0.44756464831531045 panda_joint3:=0.0040953340012492845 panda_joint4:=-1.8536410568864086 panda_joint5:=0.00694656977429986 panda_joint6:=2.2587760096773852 panda_joint7:=0.8096622498333454 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.01738492911681533 panda_joint2:=0.4567096995562315 panda_joint3:=0.009523570910459966 panda_joint4:=-1.8473155614570715 panda_joint5:=0.010610944125801325 panda_joint6:=2.2621655828677465 panda_joint7:=0.8190369996801019 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.025118614081293344 panda_joint2:=0.46541603080928323 panda_joint3:=0.016759026702857227 panda_joint4:=-1.843899088643957 panda_joint5:=0.01489635743200779 panda_joint6:=2.263395429368538 panda_joint7:=0.8282210267707706 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.031872750259935856 panda_joint2:=0.4738160603120923 panda_joint3:=0.023048830591505975 panda_joint4:=-1.8414319033850917 panda_joint5:=0.019052816089242697 panda_joint6:=2.263857419003907 panda_joint7:=0.8371537485718727 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03467370360158384 panda_joint2:=0.4805054183490574 panda_joint3:=0.024582422362072975 panda_joint4:=-1.8313339759712108 panda_joint5:=0.021667659981176257 panda_joint6:=2.269716150460008 panda_joint7:=0.845798776447773 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03642302844673395 panda_joint2:=0.49055343804880974 panda_joint3:=0.02477096973780135 panda_joint4:=-1.8303355802432635 panda_joint5:=0.024573362665250897 panda_joint6:=2.26906289548584 panda_joint7:=0.8541815084964037 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03777616121806204 panda_joint2:=0.49836808381602166 panda_joint3:=0.024903232444557943 panda_joint4:=-1.8265004267101177 panda_joint5:=0.02723930566571653 panda_joint6:=2.2701349566175484 panda_joint7:=0.8624210934713483 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.039403749513439834 panda_joint2:=0.5047828786447643 panda_joint3:=0.025596982837669202 panda_joint4:=-1.821662728872616 panda_joint5:=0.029858822468668222 panda_joint6:=2.2717573593088307 panda_joint7:=0.8708197697624565 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.04201824183110148 panda_joint2:=0.5124024207144975 panda_joint3:=0.02726277771034802 panda_joint4:=-1.8191931652487257 panda_joint5:=0.033039546804502606 panda_joint6:=2.2718826303878448 panda_joint7:=0.8792179515212775 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.04463273414876312 panda_joint2:=0.5200219627842306 panda_joint3:=0.028928572583026835 panda_joint4:=-1.8167236016248354 panda_joint5:=0.03622027114033699 panda_joint6:=2.272007901466859 panda_joint7:=0.8876161332800985 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.04689124959986657 panda_joint2:=0.5296873504668473 panda_joint3:=0.029848093437976786 panda_joint4:=-1.8194233404588886 panda_joint5:=0.03949451516382396 panda_joint6:=2.2688747833142404 panda_joint7:=0.8958704592287541 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.04828668374102563 panda_joint2:=0.5381358199194073 panda_joint3:=0.030051786359763355 panda_joint4:=-1.8188923166831956 panda_joint5:=0.0424376733135432 panda_joint6:=2.267759142339637 panda_joint7:=0.9040864891931415 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.050117702689021826 panda_joint2:=0.5450881713442504 panda_joint3:=0.03099777167699358 panda_joint4:=-1.8170826302608476 panda_joint5:=0.045295048505067825 panda_joint6:=2.2673858801551976 panda_joint7:=0.9123458694294095 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.04977118215174414 panda_joint2:=0.5526336485892533 panda_joint3:=0.029271656903802068 panda_joint4:=-1.8135672614676879 panda_joint5:=0.04755188897252083 panda_joint6:=2.268225763267546 panda_joint7:=0.9203522037342191 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05103053295169957 panda_joint2:=0.5597286935336887 panda_joint3:=0.029732561752098263 panda_joint4:=-1.8123202428920195 panda_joint5:=0.05050042341463268 panda_joint6:=2.2673628727623143 panda_joint7:=0.928362561352551 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.052071659272769466 panda_joint2:=0.5680367344431578 panda_joint3:=0.02969681982540351 panda_joint4:=-1.813819591794163 panda_joint5:=0.05345012620091438 panda_joint6:=2.2648105839450725 panda_joint7:=0.9363955454528332 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05357093064230867 panda_joint2:=0.5759194254316389 panda_joint3:=0.030292192752312985 panda_joint4:=-1.8151659057708458 panda_joint5:=0.05652420595288277 panda_joint6:=2.262303946232132 panda_joint7:=0.9444232843443752 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.056465481204213575 panda_joint2:=0.5840655866451561 panda_joint3:=0.03219660294598725 panda_joint4:=-1.8199033309007064 panda_joint5:=0.059766746358945966 panda_joint6:=2.257661265944771 panda_joint7:=0.952809300236404 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.057832102436805144 panda_joint2:=0.5884872915595769 panda_joint3:=0.033287427550021675 panda_joint4:=-1.8147057043621317 panda_joint5:=0.062278700759634376 panda_joint6:=2.2595833965294876 panda_joint7:=0.9611352290585637 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.0583590253663715 panda_joint2:=0.5938421910069882 panda_joint3:=0.03325592053261062 panda_joint4:=-1.8116613673279063 panda_joint5:=0.06464047520421445 panda_joint6:=2.259971869801811 panda_joint7:=0.9691963046416641 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05930753794382326 panda_joint2:=0.5995345002412795 panda_joint3:=0.033705577972796164 panda_joint4:=-1.809813873632811 panda_joint5:=0.0671079286839813 panda_joint6:=2.2595296897468504 panda_joint7:=0.9772709933668375 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05899960841634311 panda_joint2:=0.6046871788240968 panda_joint3:=0.03271203408621659 panda_joint4:=-1.8047871634131298 panda_joint5:=0.06925478065386415 panda_joint6:=2.2611917056617674 panda_joint7:=0.9853281471505762 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05890293862466933 panda_joint2:=0.6101357707194983 panda_joint3:=0.03193834490048175 panda_joint4:=-1.800551645248197 panda_joint5:=0.07155445893295109 panda_joint6:=2.262297634106071 panda_joint7:=0.9933715201541782 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05940098310384201 panda_joint2:=0.6163239668495952 panda_joint3:=0.03168301350342517 panda_joint4:=-1.7980961008230225 panda_joint5:=0.07394983852282166 panda_joint6:=2.2621432521098175 panda_joint7:=1.0011401394940913 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05968046331690857 panda_joint2:=0.6238873971998691 panda_joint3:=0.030902425080057583 panda_joint4:=-1.798992045072373 panda_joint5:=0.07639073231257498 panda_joint6:=2.259802774451382 panda_joint7:=1.0088936905004084 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.059959943529975135 panda_joint2:=0.6314508275501429 panda_joint3:=0.030121836656689993 panda_joint4:=-1.7998879893217237 panda_joint5:=0.0788316261023283 panda_joint6:=2.257462296792946 panda_joint7:=1.0166472415067256 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.059787101978145074 panda_joint2:=0.6394905469752847 panda_joint3:=0.02866876773441618 panda_joint4:=-1.8009177996078507 panda_joint5:=0.0812204850371927 panda_joint6:=2.255044181057892 panda_joint7:=1.0244488222151995 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05951061093219323 panda_joint2:=0.6476739433966576 panda_joint3:=0.027011192365534953 panda_joint4:=-1.8011806262889878 panda_joint5:=0.0835262609180063 panda_joint6:=2.2531434724057906 panda_joint7:=1.0322128459438682 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05916340194380609 panda_joint2:=0.6555020663328468 panda_joint3:=0.025339714213259867 panda_joint4:=-1.8007516516605393 panda_joint5:=0.0858020200394094 panda_joint6:=2.2516558239419826 panda_joint7:=1.039927087686956 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05652865312731592 panda_joint2:=0.6686497603915631 panda_joint3:=0.02012987477610295 panda_joint4:=-1.80684988622088 panda_joint5:=0.08771615172736347 panda_joint6:=2.246061749215878 panda_joint7:=1.0478327739611268 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.052388135918590706 panda_joint2:=0.6846678474359214 panda_joint3:=0.012473836968638352 panda_joint4:=-1.8177900654496626 panda_joint5:=0.08916322735603899 panda_joint6:=2.2374564562176236 panda_joint7:=1.0562267591804266 robot_ip:=172.16.0.2



Run10:
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.03763182833790779 panda_joint2:=0.013639042377471924 panda_joint3:=-0.04424343258142471 panda_joint4:=-1.4948020100593566 panda_joint5:=0.0156675036996603 panda_joint6:=1.4265610827505588 panda_joint7:=0.7859372193552554 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.07663534954190254 panda_joint2:=0.01769535917788744 panda_joint3:=-0.08862325549125671 panda_joint4:=-1.5395641446113586 panda_joint5:=0.029449420981109142 panda_joint6:=1.391950058788061 panda_joint7:=0.785412856398616 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.11538583040237427 panda_joint2:=0.021922210920602085 panda_joint3:=-0.13269473984837532 panda_joint4:=-1.5843369893729686 panda_joint5:=0.04196907300502062 panda_joint6:=1.3573081819713115 panda_joint7:=0.7866660381178372 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.1535118855535984 panda_joint2:=0.025753830084577205 panda_joint3:=-0.17650996893644333 panda_joint4:=-1.6293743036687374 panda_joint5:=0.05251498240977526 panda_joint6:=1.3222619546949863 panda_joint7:=0.7890651907282882 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.19163794070482254 panda_joint2:=0.029585449248552324 panda_joint3:=-0.22032519802451134 panda_joint4:=-1.674411617964506 panda_joint5:=0.0630608918145299 panda_joint6:=1.287215727418661 panda_joint7:=0.7914643433387392 robot_ip:=172.16.0.2




Run8:
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.03492414578795433 panda_joint2:=0.11049505237257108 panda_joint3:=-0.04206532984972 panda_joint4:=-1.8940715216100217 panda_joint5:=0.010859549045562744 panda_joint6:=2.2263677550852297 panda_joint7:=0.7750232480093837 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.07162665948271751 panda_joint2:=0.11366413610754535 panda_joint3:=-0.08349738270044327 panda_joint4:=-1.9371394649147988 panda_joint5:=0.02172772865742445 panda_joint6:=2.193120958507061 panda_joint7:=0.7640422302857042 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.10493158549070358 panda_joint2:=0.12429920847294852 panda_joint3:=-0.12106813862919807 panda_joint4:=-1.9789992183446885 panda_joint5:=0.03763781767338514 panda_joint6:=2.1645777805149553 panda_joint7:=0.7507880022376776 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.14339089021086693 panda_joint2:=0.13308625652221961 panda_joint3:=-0.16251318529248238 panda_joint4:=-2.0209938049316407 panda_joint5:=0.05586048308759928 panda_joint6:=2.133605394735932 panda_joint7:=0.738720034211874 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.18125004693865776 panda_joint2:=0.14120170364854856 panda_joint3:=-0.20392169058322906 panda_joint4:=-2.063297490030527 panda_joint5:=0.07352301385253668 panda_joint6:=2.102485439032316 panda_joint7:=0.726089082621038 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.21897338703274727 panda_joint2:=0.14955223149387165 panda_joint3:=-0.24526245892047882 panda_joint4:=-2.105578935146332 panda_joint5:=0.09166187513619661 panda_joint6:=2.071601486057043 panda_joint7:=0.7130745453387499 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.2530500739812851 panda_joint2:=0.15723501409171148 panda_joint3:=-0.28428255021572113 panda_joint4:=-2.148026192933321 panda_joint5:=0.10706149227917194 panda_joint6:=2.04280721232295 panda_joint7:=0.6991148870065809 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.28731250017881393 panda_joint2:=0.164834088019561 panda_joint3:=-0.32360002398490906 panda_joint4:=-2.1902459993958474 panda_joint5:=0.12352011166512966 panda_joint6:=2.0147506056725977 panda_joint7:=0.6871666228026152 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.3223511017858982 panda_joint2:=0.17159630062757059 panda_joint3:=-0.36406153067946434 panda_joint4:=-2.233443407714367 panda_joint5:=0.13870611134916544 panda_joint6:=1.984299458786845 panda_joint7:=0.6733499858155847 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.36039824411273 panda_joint2:=0.1812687579798512 panda_joint3:=-0.4052322693169117 panda_joint4:=-2.2752409376204015 panda_joint5:=0.15782436076551676 panda_joint6:=1.9539820945262907 panda_joint7:=0.6609231028333307 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.39844538643956184 panda_joint2:=0.19094121533213182 panda_joint3:=-0.44640300795435905 panda_joint4:=-2.317038467526436 panda_joint5:=0.17694261018186808 panda_joint6:=1.9236647302657364 panda_joint7:=0.6484962198510766 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.39997366815805435 panda_joint2:=0.18972699141828342 panda_joint3:=-0.4473922327160835 panda_joint4:=-2.31697548776865 panda_joint5:=0.17670418601483107 panda_joint6:=1.9225621803104875 panda_joint7:=0.65001171823591 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.4395490922033787 panda_joint2:=0.19818522485671564 panda_joint3:=-0.48955219611525536 panda_joint4:=-2.358710037916899 panda_joint5:=0.19558401126414537 panda_joint6:=1.8911422660946844 panda_joint7:=0.6391003336384893 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.4394620284438133 panda_joint2:=0.1981059860694222 panda_joint3:=-0.4893685653805733 panda_joint4:=-2.358570522069931 panda_joint5:=0.19644119683653116 panda_joint6:=1.8918229269236324 panda_joint7:=0.6395196318253875 robot_ip:=172.16.0.2


Run7 (always moves to the same corner):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.025731703266501427 panda_joint2:=0.10722868204116821 panda_joint3:=-0.04078081250190735 panda_joint4:=-1.8962753996253014 panda_joint5:=0.012034619227051735 panda_joint6:=2.227180974781513 panda_joint7:=0.7841496043466032 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.061710091307759285 panda_joint2:=0.10891598231624812 panda_joint3:=-0.08207595720887184 panda_joint4:=-1.9390974812209607 panda_joint5:=0.021026880480349064 panda_joint6:=2.194087701141834 panda_joint7:=0.7757073002494872 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.10162031091749668 panda_joint2:=0.11243720423895866 panda_joint3:=-0.1258099526166916 panda_joint4:=-1.9815974183380605 panda_joint5:=0.040265864692628384 panda_joint6:=2.1621864160895345 panda_joint7:=0.766017355453223 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.12726228684186935 panda_joint2:=0.10818476161453873 panda_joint3:=-0.162462767213583 panda_joint4:=-2.026103413105011 panda_joint5:=0.04402108839713037 panda_joint6:=2.129266241788864 panda_joint7:=0.7607133108749986 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.10362240485846996 panda_joint2:=0.11366824597585946 panda_joint3:=-0.15020353440195322 panda_joint4:=-2.0719260297715665 panda_joint5:=0.03857857617549598 panda_joint6:=2.096184214800596 panda_joint7:=0.7522438363730908 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.07546266168355942 panda_joint2:=0.13631978795398025 panda_joint3:=-0.1248093144968152 panda_joint4:=-2.1187928155064584 panda_joint5:=0.03987405332736671 panda_joint6:=2.0567245802283285 panda_joint7:=0.7546466427948326 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.06107084546238184 panda_joint2:=0.16675384879577904 panda_joint3:=-0.12200706172734499 panda_joint4:=-2.1653452202677728 panda_joint5:=0.038699084776453674 panda_joint6:=2.023006832450628 panda_joint7:=0.7564840513141826 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.08135906886309385 panda_joint2:=0.1911099886195734 panda_joint3:=-0.15314585249871016 panda_joint4:=-2.211057635396719 panda_joint5:=0.02419888402801007 panda_joint6:=1.989062721133232 panda_joint7:=0.7697980423318223 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.05147585738450289 panda_joint2:=0.1958965335646644 panda_joint3:=-0.12994128745049238 panda_joint4:=-2.256224049627781 panda_joint5:=0.014459467376582325 panda_joint6:=1.9486937133967874 panda_joint7:=0.7623206931957975 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.09271205682307482 panda_joint2:=0.19937277772929518 panda_joint3:=-0.17427692841738462 panda_joint4:=-2.2985772028565408 panda_joint5:=0.03430526971351355 panda_joint6:=1.9165595418214796 panda_joint7:=0.7559262174973265 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.13302618358284235 panda_joint2:=0.2017911467840895 panda_joint3:=-0.21840423252433538 panda_joint4:=-2.3419852800667287 panda_joint5:=0.051919766585342586 panda_joint6:=1.8834579123556612 panda_joint7:=0.7511592615256086 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.17309305910021067 panda_joint2:=0.20568030817899852 panda_joint3:=-0.2620629472658038 panda_joint4:=-2.3845973931252957 panda_joint5:=0.06850586726795882 panda_joint6:=1.8508181667327879 panda_joint7:=0.7440224756347016 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.2149095581844449 panda_joint2:=0.21219737293664365 panda_joint3:=-0.30599577631801367 panda_joint4:=-2.425533464550972 panda_joint5:=0.08719595975708216 panda_joint6:=1.81824738740921 panda_joint7:=0.7365376400621608 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.2575651304796338 panda_joint2:=0.22043470504228024 panda_joint3:=-0.34988227020949125 panda_joint4:=-2.4657972775399686 panda_joint5:=0.10691137134563178 panda_joint6:=1.7853089927136896 panda_joint7:=0.7298666849592701 robot_ip:=172.16.0.2


Run6:
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.020499063655734062 panda_joint2:=0.10610145972110331 panda_joint3:=-0.034919749945402145 panda_joint4:=-1.8933638952672482 panda_joint5:=-0.0040612551383674145 panda_joint6:=2.2289509514719246 panda_joint7:=0.7811897480487824 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.04243511147797108 panda_joint2:=0.1074513977067545 panda_joint3:=-0.06590047478675842 panda_joint4:=-1.9368960432708264 panda_joint5:=-0.006846586242318153 panda_joint6:=2.194824852272868 panda_joint7:=0.7694939635321498 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.011460425332188606 panda_joint2:=0.13038635977078228 panda_joint3:=-0.03659387119114399 panda_joint4:=-1.982349479943514 panda_joint5:=-0.004704436985775828 panda_joint6:=2.158953529074788 panda_joint7:=0.7648653112910688 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.0199744189158082 panda_joint2:=0.16232956178952007 panda_joint3:=-0.04632555786520243 panda_joint4:=-2.0289648532867433 panda_joint5:=-0.009086913196370006 panda_joint6:=2.119904108569026 panda_joint7:=0.7790082594938577 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.0319758765399456 panda_joint2:=0.19122666843701153 panda_joint3:=-0.06740863341838121 panda_joint4:=-2.074929217249155 panda_joint5:=-0.015688527142629027 panda_joint6:=2.0820503125339744 panda_joint7:=0.7919094706512988 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.03430463606491685 panda_joint2:=0.1972668899828568 panda_joint3:=-0.07778555806726217 panda_joint4:=-2.120554356276989 panda_joint5:=-0.02386353421024978 panda_joint6:=2.0416527541726825 panda_joint7:=0.7925223792763427 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.06579764047637582 panda_joint2:=0.1994310512812808 panda_joint3:=-0.11602027621120214 panda_joint4:=-2.1622273795306683 panda_joint5:=-0.02008076710626483 panda_joint6:=2.0107368493825195 panda_joint7:=0.7873424851195887 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.09570874506607652 panda_joint2:=0.1999937191279605 panda_joint3:=-0.15359368454664946 panda_joint4:=-2.2050349958240987 panda_joint5:=-0.018006203463301063 panda_joint6:=1.9782382848113773 panda_joint7:=0.78237326963339 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.12636520760133862 panda_joint2:=0.20088828950189053 panda_joint3:=-0.19156701397150755 panda_joint4:=-2.247739727050066 panda_joint5:=-0.016086904099211097 panda_joint6:=1.9450776243954895 panda_joint7:=0.7767569949710742 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.16012510703876615 panda_joint2:=0.20229206235613673 panda_joint3:=-0.2314581824466586 panda_joint4:=-2.2905153430998326 panda_joint5:=-0.01207823515869677 panda_joint6:=1.9110301586240528 panda_joint7:=0.770453851283528 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.19589754240587354 panda_joint2:=0.20590129053685813 panda_joint3:=-0.27216027583926916 panda_joint4:=-2.332738547027111 panda_joint5:=-0.005238095531240106 panda_joint6:=1.8770140895992515 panda_joint7:=0.7630781755642966 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.22927713813260198 panda_joint2:=0.21016370203811674 panda_joint3:=-0.3110468788072467 panda_joint4:=-2.3748023867607118 panda_joint5:=-0.0003040886949747801 panda_joint6:=1.8433733733743427 panda_joint7:=0.7537810430722311 robot_ip:=172.16.0.2



Run5 (try from high up):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.0071739316917955875 panda_joint2:=0.10809627828653902 panda_joint3:=-0.02741268090903759 panda_joint4:=-1.8950722239911557 panda_joint5:=-0.003193891141563654 panda_joint6:=2.230770688727498 panda_joint7:=0.775614150762558 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.019580853637307882 panda_joint2:=0.11520536012481898 panda_joint3:=-0.014234683476388454 panda_joint4:=-1.940384878218174 panda_joint5:=-0.006286117481067777 panda_joint6:=2.197481993809342 panda_joint7:=0.7656973767653108 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.005837756674736738 panda_joint2:=0.12908903464209287 panda_joint3:=-0.0419599087908864 panda_joint4:=-1.9865430913865567 panda_joint5:=-0.007491254829801619 panda_joint6:=2.160874366983771 panda_joint7:=0.7709326613321901 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03404370089992881 panda_joint2:=0.14239683888386934 panda_joint3:=-0.027831951156258583 panda_joint4:=-2.0302877560257913 panda_joint5:=-0.014197233249433339 panda_joint6:=2.1265814530104397 panda_joint7:=0.7639020429551602 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03404370089992881 panda_joint2:=0.14239683888386934 panda_joint3:=-0.027831951156258583 panda_joint4:=-2.0302877560257913 panda_joint5:=-0.014197233249433339 panda_joint6:=2.1265814530104397 panda_joint7:=0.7639020429551602 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.00021348008885979652 panda_joint2:=0.14802386892493813 panda_joint3:=-0.06766662932932377 panda_joint4:=-2.0716819636523724 panda_joint5:=-0.014669910015072674 panda_joint6:=2.093977377042174 panda_joint7:=0.7503229425847531 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.042141003999859095 panda_joint2:=0.16120318324770777 panda_joint3:=-0.10931639187037945 panda_joint4:=-2.109112484753132 panda_joint5:=-0.0008665592758916318 panda_joint6:=2.06105516128242 panda_joint7:=0.7387892054021359 robot_ip:=172.16.0.2



Run4 (green cube outside FOV):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.04030979797244072 panda_joint2:=0.42341497898101804 panda_joint3:=-0.04121299460530281 panda_joint4:=-1.8891332618892194 panda_joint5:=0.021772896870970726 panda_joint6:=2.227819757759571 panda_joint7:=0.7800657534599305 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.08134252950549126 panda_joint2:=0.4331674015522003 panda_joint3:=-0.08414696529507637 panda_joint4:=-1.9298331424593926 panda_joint5:=0.042354756966233253 panda_joint6:=2.1932164153456686 panda_joint7:=0.7642640852928162 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.12359117716550827 panda_joint2:=0.443056384883821 panda_joint3:=-0.12806515395641327 panda_joint4:=-1.9707852959632874 panda_joint5:=0.06565413624048233 panda_joint6:=2.157842817008495 panda_joint7:=0.7471542851626873 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.16639995947480202 panda_joint2:=0.45567264951765535 panda_joint3:=-0.17185182496905327 panda_joint4:=-2.010619270056486 panda_joint5:=0.09223119169473648 panda_joint6:=2.123318283110857 panda_joint7:=0.7334354240074754 robot_ip:=172.16.0.2

Run3 (green cube):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.03680632263422012 panda_joint2:=0.4129766380321234 panda_joint3:=0.00937588233500719 panda_joint4:=-1.8953404150903226 panda_joint5:=-0.006169310305267572 panda_joint6:=2.2535855981893835 panda_joint7:=0.7689621476083994 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.06179207190871239 panda_joint2:=0.4485197333525866 panda_joint3:=0.008832705207169056 panda_joint4:=-1.9420623317360879 panda_joint5:=0.010039148386567831 panda_joint6:=2.2409338715858755 panda_joint7:=0.7556994109973312 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.055030545219779015 panda_joint2:=0.45773154924623666 panda_joint3:=0.02360771968960762 panda_joint4:=-1.9407781161367894 panda_joint5:=-0.0007746964693069458 panda_joint6:=2.2234469584561882 panda_joint7:=0.7627491084486246 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.06573226768523455 panda_joint2:=0.5030579684209078 panda_joint3:=0.023995954543352127 panda_joint4:=-1.9864444345235825 panda_joint5:=-0.00041656196117401123 panda_joint6:=2.1966779164411125 panda_joint7:=0.7531016021594406 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.05814064433798194 panda_joint2:=0.5273396974336355 panda_joint3:=-0.003778534010052681 panda_joint4:=-2.0320628307759763 panda_joint5:=0.009613153524696827 panda_joint6:=2.177940039653331 panda_joint7:=0.7406483376026154 robot_ip:=172.16.0.2

Run2 (green cube, moves the same way as the shadow again. Could be the shadow causing problems):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.0016870528925210238 panda_joint2:=0.4431880338490009 panda_joint3:=-0.02705479972064495 panda_joint4:=-1.896151801943779 panda_joint5:=0.005454081576317549 panda_joint6:=2.238752480819821 panda_joint7:=0.7783362352848053 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=0.009170565521344543 panda_joint2:=0.47690669059753416 panda_joint3:=-0.04992498643696308 panda_joint4:=-1.9430316038429738 panda_joint5:=0.009524357505142689 panda_joint6:=2.2128740271925924 panda_joint7:=0.75876060962677 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.02546518319286406 panda_joint2:=0.48197135686874387 panda_joint3:=-0.09467396698892117 panda_joint4:=-1.9877493590116502 panda_joint5:=0.02988277655094862 panda_joint6:=2.179186785817146 panda_joint7:=0.7327617955952883 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.06395632424391806 panda_joint2:=0.48650988129898903 panda_joint3:=-0.14013070054352283 panda_joint4:=-2.0321509897708894 panda_joint5:=0.04930798429995775 panda_joint6:=2.142310751825571 panda_joint7:=0.7048572571575642 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.10445791878737509 panda_joint2:=0.4912824014574289 panda_joint3:=-0.18582151643931866 panda_joint4:=-2.0759558625519277 panda_joint5:=0.06965838093310595 panda_joint6:=2.10450613707304 panda_joint7:=0.6769865834712983 robot_ip:=172.16.0.2

Run1 (yellow cube fail, went the other way...):
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.041773345321416855 panda_joint2:=0.42387134846299884 panda_joint3:=-0.04218156263232231 panda_joint4:=-1.8882800586521626 panda_joint5:=0.025693370029330254 panda_joint6:=2.2299420575052498 panda_joint7:=0.783774765636772 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.08538996055722237 panda_joint2:=0.43864465210586784 panda_joint3:=-0.08544142171740532 panda_joint4:=-1.9255182132124902 panda_joint5:=0.05076695792376995 panda_joint6:=2.1968449617177246 panda_joint7:=0.7753086466901005 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.12906743213534355 panda_joint2:=0.4533021495118737 panda_joint3:=-0.1287928894162178 panda_joint4:=-1.9627636879682542 panda_joint5:=0.07584917172789574 panda_joint6:=2.163606669977307 panda_joint7:=0.7665521758235991 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.17271152511239052 panda_joint2:=0.46568808410316703 panda_joint3:=-0.17382195964455605 panda_joint4:=-2.0029181376099587 panda_joint5:=0.10527684353291988 panda_joint6:=2.127837132886052 panda_joint7:=0.7484322352893651 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.21697348356246948 panda_joint2:=0.4803205000609159 panda_joint3:=-0.2186221480369568 panda_joint4:=-2.0417620301246644 panda_joint5:=0.13598838821053505 panda_joint6:=2.092690132781863 panda_joint7:=0.7344641373120249 robot_ip:=172.16.0.2
roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:=-0.2610311880707741 panda_joint2:=0.495515059530735 panda_joint3:=-0.2630111128091812 panda_joint4:=-2.0803533144295216 panda_joint5:=0.16567306406795979 panda_joint6:=2.0583692828565834 panda_joint7:=0.7232283041439951 robot_ip:=172.16.0.2

Initial pose:

roslaunch franka_example_controllers move_to_start_philip.launch panda_joint2:=0.11 panda_joint6:=1.96 robot_ip:=172.16.0.2
Even higher: roslaunch franka_example_controllers move_to_start_philip.launch panda_joint2:=0.01 panda_joint4:=-1.45 panda_joint6:=1.46 robot_ip:=172.16.0.2


"""

# for element in action_array:
#     # Convert string to list of floats
#     float_list = [float(num) for num in element.strip('[]').split(',')]

#     # Multiply each float by 0.05
#     multiplied_list = [num * 0.05 for num in float_list] # panda-gym multiplies joint action by 0.05 to limit change in pose
    
#     for i in range(len(current_joints)):
#         current_joints[i] += multiplied_list[i]

#     # Convert back to string
#     multiplied_string= '[' + ','.join(map(str, multiplied_list)) + ']'
#     multiplied_absolute_string = '[' + ','.join(map(str, current_joints)) + ']'

#     # Append to result list
#     actual_joints.append(multiplied_absolute_string)
#     multiplied_joints.append(multiplied_string)

# print("Actual joint list:", actual_joints)