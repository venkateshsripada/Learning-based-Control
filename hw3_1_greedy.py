import numpy as np
import math

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
global q
#initialize with high value of q

#print q

#iterate q values
# def iterate_q(q):
# 	for i in range(0,5):
# 		q_next = q[i] + step_size*(reward[i] - q[i])
# 		q[i] = q_next
# #	print q
# iterate_q(q)

#calculate all probabilites

# def probability(q):
# 	#iterate_q(q)
# 	for i in range(0,5):
# 			for j in range(0,5):
# 				prob[j] = max(q[i])
		 
	
# 	print prob
#probability(q)

#take the element with maximum one
#def maximum_softmax():
epoch = 0
q = np.zeros(shape = (5,1))
q[0] = 50
q[1] = 50
q[2] = 50
q[3] = 50
q[4] = 50
while epoch <= 9:	
	print ("GREEDY ALGO")
	for i in range(0,5):
		for j in range(0,5):
			prob[j] = max(q[i])
	print prob
	position = prob.argmax()
	print position

	#iterate the value of this
	new_iterate = (reward[position]-q[position])/2 + q[position]
	print new_iterate

	q[position] = new_iterate
	print q
	

	epoch = epoch + 1 
# maximum_softmax()


# #again take max of values
# epoch = 0
# while epoch <= 9:
	
# 	print ("SOFTMAX ALGO")
# 	maximum_softmax()
# 	epoch = epoch + 1 