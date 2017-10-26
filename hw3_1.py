import numpy as np
import math

temp = 100
step_size = 0.5
sum_q = 0
exp_q = np.zeros(shape = (5,1))
prob = np.zeros(shape = (5,1))
reward = np.zeros(shape = (5,1))
reward[0] = 5
reward[1] = 4
reward[2] = 3
reward[3] = 7
reward[4] = 2
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
	print q
#iterate_q(q)

#calculate all probabilites

def probability(q):
	iterate_q(q)
	for i in range(0,5):
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
probability(q)

#take the element with maximum one
position = prob.argmax()
print position
#iterate the value of this
for position in q:
	new_iterate = q[position] + step_size*int(reward[position] - q[position])
	q[position] = new_iterate

print q[position]
