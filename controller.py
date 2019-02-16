'''
This is a realization of the controller from Vehicle Dynamics and Control by Rajesh Rajamani.
Yaohui Guo
'''
from BuggySimulator import *
import numpy as np
import scipy
import control
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from util import *
import math
import matplotlib.pyplot as plt

class pid:
    def __init__(self,error_I,prev_error,dt):
        self.error_I= error_I
        self.prev_error=prev_error
        self.dt= dt

    def setGains(self,p,i,d):
        self.kp= p
        self.ki=i
        self.kd=d

    def calculateInput(self,des_output,state):
        self.cur_error= des_output-state
        self.error_I+= self.cur_error
        self.error_D = (self.cur_error-self.prev_error)/self.dt
        des_input=self.kp*(self.cur_error) + self.ki*(self.error_I)+self.kd*(self.error_D)
        self.prev_error=self.cur_error

        return des_input




def controller(traj,vehicle):
	m = 2000
	l_r = 1.7
	l_f = 1.1
	C_apha = 15000
	I_z = 3344
	f = 0.01
	dt = 0.05


	#lateral Dynamics

	currentstate = vehicle.state
	xd = currentstate.xd
	yd = currentstate.yd
	phi = currentstate.phi
	phid = currentstate.phid
	X = currentstate.X
	Y = currentstate.Y
	delta = currentstate.delta

	A_lt = np.array([[0,1,0,0],
		            [0,-4*C_apha/(m*xd),4*C_apha/m,2*C_apha*(l_r-l_f)/(m*xd)],
		            [0,0,0,1],
		            [0,-2*C_apha*(l_f-l_r)/(I_z*xd),2*C_apha*(l_f-l_r)/(I_z),-2*C_apha*(l_f**2+l_r**2)/(I_z*xd)]])
    
	B_lt = np.array([[0],[2*C_apha/m],[0],[2*C_apha*l_f/I_z]])
	C_lt = np.array([1,1,1,1])
	D_lt = 0
	sysD_lt = signal.cont2discrete((A_lt,B_lt,C_lt,D_lt),dt=0.05)
	A_ltD = sysD_lt[0]
	B_ltD = sysD_lt[1]
	Q_lt = np.diag([1.2,2.5,4,3])
	R_lt = 1
	K_lt,S_lt,_= dlqr(A_ltD,B_ltD,Q_lt,R_lt)
	


	dist,nearest_idx = closest_node(X,Y, traj)

	
	#CALCULATING CURVATURE
	
	#GAUSSIAN METHOD
	Xd= gaussian_filter1d(traj[:,0],sigma=10,order=1)
	Xdd= gaussian_filter1d(traj[:,0],sigma=10,order=2)
	Yd=gaussian_filter1d(traj[:,1],sigma=10,order=1)
	Ydd=gaussian_filter1d(traj[:,1],sigma=10,order=2)

	curvature= np.divide((np.multiply(Xd,Ydd)-np.multiply(Yd,Xdd)),np.power((np.square(Xd)+np.square(Yd)),1.5))
	
	if nearest_idx<8000 and xd<=6:
	   look_ahead=100
	elif nearest_idx<8000 and xd>6:
		look_ahead=120
	elif nearest_idx>=8000:
		look_ahead=0


	
	phi_des=np.arctan2(Yd[nearest_idx],Xd[nearest_idx])
	#print(nearest_idx)
	
	#CALCULATING ERROR E1
	phi_ahead=np.arctan2(Yd[nearest_idx+look_ahead],Xd[nearest_idx+look_ahead])
	error_distance=np.array([[X,Y]])-traj[nearest_idx+look_ahead,:]
	Y_unit= np.array([[-np.sin(phi_ahead)],[np.cos(phi_ahead)]])
	
	e1=(currentstate.Y-traj[nearest_idx+look_ahead,1])*math.cos(phi_ahead)- (currentstate.X-traj[nearest_idx+look_ahead,0])*math.sin(phi_ahead)
	
	e2=wrap2pi(phi-phi_ahead)
	e1d= yd+xd*(e2)
	
	e2d=phid-xd*np.average(curvature[nearest_idx+look_ahead-20:nearest_idx+look_ahead+20])

	

	observed_error=np.array([[e1],
		                    [e1d],
		                    [e2],
		                    [e2d]])
	
	
	

	delta_next= K_lt@observed_error
	#deltad= (delta_next-delta)/dt
	delta_next = np.asscalar(delta_next)
	deltad_pid= pid(0,0,0.05)
	deltad_pid.setGains(p=0,i=0,d=1)
	deltad= deltad_pid.calculateInput(delta_next,vehicle.state.delta)

	#TRYING PID CONTROL
	F_pid= pid(0,0,0.05)
	F_pid.setGains(p=2000,i=2500,d=1500)

	
	if nearest_idx<1900:
		F= F_pid.calculateInput(35,vehicle.state.xd)
	elif nearest_idx>=1900 and nearest_idx<=2500:
		F= F_pid.calculateInput(7,vehicle.state.xd)
	elif nearest_idx>2500 and nearest_idx<=5500:
		F= F_pid.calculateInput(20.5,vehicle.state.xd)
	elif nearest_idx>5500 and nearest_idx<=6000:
		F= F_pid.calculateInput(6.5,vehicle.state.xd)
	elif nearest_idx>7610 and nearest_idx<=7900:
		F= F_pid.calculateInput(9.3,vehicle.state.xd)
	else:
		F= F_pid.calculateInput(30.5,vehicle.state.xd)


	return F,deltad

def dlqr(A,B,Q,R):
	"""Solve the discrete time lqr controller.
	 
	x[k+1] = A x[k] + B u[k]
	 
	cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
	"""
	
	X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	 
	#compute the LQR gain
	K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(-B.T*X*A))
	 
	eigVals, eigVecs = scipy.linalg.eig(A+B*K)
	 
	return K, X, eigVals                                                                   