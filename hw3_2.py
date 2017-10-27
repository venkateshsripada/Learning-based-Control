import numpy as np
from numpy import unravel_index
import math

sum_q = 0
temp = 100
prob = np.zeros(shape = (5,10))
exp_q = np.zeros(shape = (5,10))

world = np.zeros(shape = (5,10))
world[world == 0] = -1
world[3,9] = 100
print world

#robot moves left right or stays there
# def motiion():
# 	state = random.rand[i][j]
# 	state = state[i][j]

def probability(q):

	for i in range(0,5):
		for j in range(0, 10):
			exp_q[i][j] = math.exp(q[i][j]/temp)
			global sum_q
			sum_q = sum_q + exp_q[i][j]
	for i in range(0,5):
		for j in range(0,10):
			prob[i][j] = exp_q[i][j]
			prob[i][j] = prob[i][j]/sum_q
	print prob
probability(world)

position = unravel_index(world.argmax(), world.shape)
print position
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