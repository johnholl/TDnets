import numpy as np
from matplotlib import pyplot as plt

learning_data = np.load("/home/john/code/pythonfiles/TDnets/learning_data.npy")
time_vals = [learning_data[i][0] for i in range(len(learning_data))]
Q_vals = [learning_data[i][1] for i in range(len(learning_data))]
reward_vals = [learning_data[i][2] for i in range(len(learning_data))]
max_reward_vals = [learning_data[i][3] for i in range(len(learning_data))]
step_vals = [learning_data[i][4] for i in range(len(learning_data))]
loss = [learning_data[i][5] for i in range(len(learning_data))]




plt.plot(time_vals, Q_vals)
plt.show()


