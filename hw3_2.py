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
#TODO: INITIALIZE STATES AND ASSIGN IT TO WORLD
state = np.array(shape = (5,10))
i = np.random.randint(0,5)
j = np.random.randint(0,10)
state
#robot moves left right or stays there
# def motiion():
# 	state = random.rand[i][j]
# 	state = state[i][j]
global r
global u
global d
global l
global s
def go_up():
	state[i][j] = world[i][j]
	world[i][j] = world[i-1][j]
	world[i][j] = world[i][j] + reward[i][j]
	
	u = probability(world)
	return u
	print u


def go_right():
	state[i][j] = world[i][j]
	world[i][j] = world[i][j+1]
	world[i][j] = world[i][j] + reward[i][j]
	
	r = probability(world[i][j])
	return r
	print r

def go_down():
	state[i][j] = world[i][j]
	world[i][j] = world[i+1][j]
	world[i][j] = world[i][j] + reward[i][j]
	
	d =probability(world[i][j])
	return d
	print d

def go_left():
	state[i][j] = world[i][j]
	world[i][j] = world[i][j-1]
	world[i][j] = world[i][j] + reward[i][j]
	
	l =probability(world[i][j])
	return l
	print l

def stay():
	state[i][j] = world[i][j]
	world[i][j] = world[i][j]
	world[i][j] = world[i][j] + reward[i][j]
	
	s =probability(world[i][j])
	return s
	print s

 
# u = go_up()
# r = go_right()
# d = go_down()
# l = go_left()
# s = stay()

if go_up() > go_right() and go_down() and go_left() and stay():
	state[i][j] = state[i-1][j]
elif go_right() > go_up() and go_down() and go_left() and stay():
	state[i][j] = state[i][j+1]
elif go_down() > go_up() and go_right() and go_left() and stay():
	state[i][j] = state[i+1][j]
elif go_left() > go_up() and go_down() and go_right() and stay():
	state[i][j] = state[i][j-1]
else:
	state[i][j] = state[i][j]

def probability(q):

	# for i in range(0,5):
	# 	for j in range(0, 10):
	position = unravel_index(state, world.shape)
	#put just position and not iterate over all of i,j
	exp_q[i][j] = math.exp(q[position]/temp)
	for i in range(0,5):
		for j in range(0, 10):
			global sum_q
			sum_q = sum_q + exp_q[i][j]
	# for i in range(0,5):
	# 	for j in range(0,10):
	# 		#again put just position and not iterate over all i,j
	prob[i][j] = exp_q[i][j]
	prob[i][j] = prob[i][j]/sum_q
	print prob
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