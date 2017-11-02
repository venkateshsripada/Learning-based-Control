import numpy as np
import math
import random
from random import random
from random import sample

temp = 100
x = 0.5
step_size = 0.5
sum_q = 0
exp_q = np.zeros(shape = (5,1))
prob = np.zeros(shape = (5,1))
reward = np.zeros(shape = (5,1))
reward[0] = np.random.normal(1, 5)
reward[1] = np.random.normal(1.5, 1)
reward[2] = np.random.normal(2, 1)
reward[3] = np.random.normal(2, 2)
reward[4] = np.random.normal(1.75, 10)
global q

epoch = 0
q = np.zeros(shape = (5,1))
q[0] = 50
q[1] = 50
q[2] = 50
q[3] = 50
q[4] = 50
while epoch <= 99:	
	print ("EGREEDY ALGO")

 	for i in range(0,5):
 		if x > random():

 			prob[i] = max(q[i])
 		else:
 			r = np.random.randint(0, 5)
 			prob[i] = q[r] 
	print prob
	position = prob.argmax()
	print position

	#iterate the value of this
	new_iterate = (reward[position]-q[position])/2 + q[position]
	print new_iterate

	q[position] = new_iterate
	print q
	print "reward", max(reward)
	
	epoch = epoch + 1 
