import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.plot([1,2,3,4,5,6,7,8,9,10], [4,5,1,2,4,1,3,5,5,1], 'g^', label = "Softmax algorithm")
plt.plot([1,2,3,4,5,6,7,8,9,10], [4,5,5,2,4,5,1,4,4,5], 'rs', label = "E-Greeedy algorithm")
plt.plot([1,2,3,4,5,6,7,8,9,10], [2,5,5,5,4,4,5,5,1,1], 'ko', label = "Greedy Algorithm")

plt.axis([0,11,0,7])
plt.title('Three algorithms: 10 time steps')
plt.xlabel('iterations')
plt.ylabel('action')
plt.legend()
#x = list(range(0,10))
plt.show()