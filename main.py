from BuggySimulator import *
import numpy as np
from controller import *
from util import *
import matplotlib.pyplot as plt
from Evaluation import *

class pid:
    def __init__(self,error_I,prev_error,dt):
        self.error_I= error_I
        self.prev_error=prev_error
        self.dt= dt

    def setGains(self,p,i,d):
        self.kp= p
        self.ki=i
        self.kd=d

    def calculateInput(self,des_output,vehicle):
        self.cur_error= des_output-vehicle.state.xd
        self.error_I+= self.cur_error
        self.error_D = (self.cur_error-self.prev_error)/self.dt
        des_input=self.kp*(self.cur_error) + self.ki*(self.error_I)+self.kd*(self.error_D)
        self.prev_error=self.cur_error

        return des_input


# get the trajectory
traj = get_trajectory('buggyTrace.csv')
# initial the Buggy
vehicle = initail(traj,0)

n = 5000
X = []
Y = []
delta = []
xd = []
yd = []
phi = []
phid = []
deltad_list = []
deltad=[]
F = []
minDist =[]
'''
your code starts here
'''


command=vehicle.command
# preprocess the trajectory
passMiddlePoint = False
nearGoal = False
for i in range(n):

    command.F,command.deltad = controller(traj,vehicle)

    
    vehicle.update(command = command)
    # termination check
    disError,nearIdx = closest_node(vehicle.state.X, vehicle.state.Y, traj)
    stepToMiddle = nearIdx - len(traj)/2.0
    if abs(stepToMiddle) < 100.0:
        passMiddlePoint = True
        print('middle point passed')
    nearGoal = nearIdx >= len(traj)-50
    if nearGoal and passMiddlePoint:
        print('destination reached!')
        break
    # record states
    X.append(vehicle.state.X)
    Y.append(vehicle.state.Y)
    delta.append(vehicle.state.delta)
    xd.append(vehicle.state.xd)
    yd.append(vehicle.state.yd)
    phid.append(vehicle.state.phid)
    phi.append(vehicle.state.phi)
    deltad.append(command.deltad)
    F.append(command.F)
    minDist.append(disError)

    cur_state = save_state(vehicle.state)
    cur_state = np.array(cur_state)


    if i == 0:
        cur_state = np.array(cur_state)
        state_saved = cur_state.reshape((1, 7))
    else:
        state_saved = np.concatenate((state_saved, cur_state.reshape((1, 7))), axis=0)


showResult(traj,X,Y,delta,xd,yd,F,phi,phid,minDist)
np.save('24-677_Project_3_BuggyStates_Raftaar.npy', state_saved)
evaluation(minDist,traj,X,Y,taskNum=3)