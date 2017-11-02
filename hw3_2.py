import numpy as np
from numpy import unravel_index
import math

sum_q = 0
temp = 100
prob = np.zeros(shape = (5,10))
exp_q = np.zeros(shape = (5,10))

world = np.zeros(shape = (5,10))
world[world == 0] = 50

reward = np.zeros(shape = (5,10))
reward[reward == 0] = -1
reward[3,9] = 100
print world


#TODO INITIALIZE STATES AND ASSIGN IT TO WORLD
state = np.empty([5,10])
#state[i,j] = state[3][5]

i = np.random.randint(0,5)
j = np.random.randint(0,10)
# for i  in range(0,4):
# 	for j in range(0,9):
# 		state[i][j] = state[np.random.randint(0,5)][np.random.randint(0,10)]
#		print state
# for i = [1,2,3,4,5]:
# 	for j = [1,2,3,4,5,6,7,8,9,10]:

# 			state[i][j] = state[i][j]


 
#new = i+1,j  i-1,j  i, j-1   i,j+1  i,j
def probability(q):

	# for i in range(0,5):
	# 	for j in range(0, 10):
	position = unravel_index(world.argmax(), world.shape)
	#print position
	#put just position and not iterate over all of i,j
	exp_q[position] = math.exp(q[position]/temp)
	
	for i in range(0,5):
		for j in range(0, 10):
			global sum_q
			sum_q = sum_q + exp_q[i][j]
			
	for i in range(0,5):
	 	for j in range(0,10):
	# 		#again put just position and not iterate over all i,j
			prob[position] = exp_q[position]
			prob[position] = prob[i][j]/sum_q
	return prob[position]
	#print prob[position]

global r
global u
global d
global l
global s
def go_up():
	probability(world)
	state[i,j] = world[i,j]
	world[i,j] = world[i-1,j]
	world[i,j] = world[i][j] + reward[i][j]
	
	probability(world)
	#print u


def go_right():
	probability(world)
	state[i,j] = world[i,j]
	world[i,j] = world[i,j+1]
	world[i,j] = world[i][j] + reward[i][j]
	
	probability(world)
	
	#print r

def go_down():
	probability(world)
	state[i,j] = world[i,j]
	world[i,j] = world[i+1,j]
	world[i,j] = world[i][j] + reward[i][j]
	
	probability(world)
	#print d

def go_left():
	probability(world)
	state[i,j] = world[i,j]
	world[i,j] = world[i,j-1]
	world[i,j] = world[i][j] + reward[i][j]
	
	probability(world)
	#print l

def stay():
	probability(world)
	state[i,j] = world[i,j]
	world[i,j] = world[i,j]
	world[i,j] = world[i][j] + reward[i][j]
	
	probability(world)
	#print s


def motion():	
	if go_up() > (go_right() and go_down() and go_left() and stay()):
		state[i,j] = state[i-1,j]
	elif go_right() > (go_up() and go_down() and go_left() and stay()):
		state[i,j] = state[i,j+1]
	elif go_down() > (go_up() and go_right() and go_left() and stay()):
		state[i,j] = state[i+1,j]
	elif go_left() > (go_up() and go_down() and go_right() and stay()):
		state[i,j] = state[i,j-1]
	else:
		state[i,j] = state[i,j]
	print ("current state = %r") %state[i,j] 
	print state
epoch = 0
while epoch <= 9:
	motion()
	epoch = epoch + 1
# new_iterate = (reward[position]-q[position])/2 + q[position]
# print new_iterate

# q[position] = new_iterate
# print q


#probability(world)

# position = unravel_index(world.argmax(), world.shape)
# print position

# position = unravel_index(state[i+1][j+1].argmax(), world.shape)
# new_iterate = (reward[position]-q[position])/2 + q[position]
# state[i][j] = new_iterate
'''
def maximum_softmax(): 
	probability(q)
	position = prob.argmax()
	print position

	#iterate the value of this
	new_iterate = (reward[position]-q[position])/2 + q[position]
	print new_iterate

	q[position] = new_iterate
	#print prob
	print q
'''