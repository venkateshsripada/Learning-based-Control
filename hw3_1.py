import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

temp = 100
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
#initialize with high value of q
q = np.zeros(shape = (5,1))
q[0] = 50
q[1] = 50
q[2] = 50
q[3] = 50
q[4] = 50
#print q

#iterate q values
def iterate_q(q):
	for i in range(0,5):
		q_next = q[i] + step_size*(reward[i] - q[i])
		q[i] = q_next
#	print q
iterate_q(q)

#calculate all probabilites

def probability(q):
	#iterate_q(q)
	for i in range(0,len(q)):
		exp_q[i] = math.exp(q[i]/temp)
#		exp_q[i] = np.array(exp_q, dtype = np.float64)
#		print exp_q[i]
#	for i in range(0,5):
		global sum_q
		sum_q = sum_q + exp_q[i]
#		print sum_q
	for j in range(0,5):
		prob[j] = exp_q[j]
		prob[j] = prob[j]/sum_q
#	print prob
#probability(q)

#take the element with maximum one
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
	print "reward", max(reward)
maximum_softmax()


#again take max of values
epoch = 0
while epoch <= 99:
	
	print ("SOFTMAX ALGO")
	maximum_softmax()
	
	epoch = epoch + 1 