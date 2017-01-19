import numpy as np
from matplotlib import pyplot as plt

learning_data = np.load("/home/john/code/pythonfiles/TDnets/chainmrp/loss_1482251952.487911.npy")
# time_vals = [learning_data[i][0] for i in range(len(learning_data))]
# Q_vals = [learning_data[i][1] for i in range(len(learning_data))]
# reward_vals = [learning_data[i][2] for i in range(len(learning_data))]
# max_reward_vals = [learning_data[i][3] for i in range(len(learning_data))]
# step_vals = [learning_data[i][4] for i in range(len(learning_data))]
# loss = [learning_data[i][5] for i in range(len(learning_data))]
# prob = [learning_data[i][6] for i in range(len(learning_data))]

x_vals = range(len(learning_data))


# avg = []
# for i in range(10, 655):
#     avg_val = sum(reward_vals[i-10:i+10])/10.
#     avg.append(avg_val)
#
# print(len(avg))
# print(len(time_vals[5:660]))

plt.plot(x_vals, learning_data)
plt.show()

