from ale_python_interface import ALEInterface
import numpy as np
import time
import random

try:
    from Tkinter import *
    import ImageTk
    from PIL import Image

except ImportError:
    print("Machine does not have libraries for rendering")


#
# actions:
# left = 0, right = 1, up = 2, down = 3. Not all actions will be possible in every state
# colors:
# 0 = black, 1 = white, 2 = red, 3 = green, 4 = blue, 5 = yellow, 6 = agent
#
#


class IMaze:

    def __init__(self):
        self.action_space = [0,1,2,3]
        self.global_time = 0

        self.maze = np.zeros(shape=[14, 14])
        self.hall_length = random.choice([5,7,9])
        self.agentpos = [1, 3]
        for i in range(1, 6):
            self.maze[1][i] = 1.
            self.maze[self.hall_length+1][i] = 1.
        for j in range(1, self.hall_length+1):
            self.maze[j][3] = 1.

        self.traffic_light = random.choice([2.,3.])
        self.maze[1][1] = self.traffic_light
        self.maze[1+self.hall_length][1] = 4.
        self.maze[1+self.hall_length][5] = 2.
        self.maze[self.agentpos[0]][self.agentpos[1]] = 5.

        self.obs = self.get_obs()

        self.isDone = False

        self.root = Tk()
        self.window = Canvas(self.root, width = len(self.maze)*20, height=len(self.maze)*20)
        self.window.pack()
        self.draw_gridworld()


    def reset(self):
        self.global_time = 0
        self.maze = np.zeros(shape=[14, 14])
        self.hall_length = random.choice([5,7,9])
        self.agentpos = [1, 3]
        for i in range(1, 6):
            self.maze[1][i] = 1
            self.maze[self.hall_length+1][i] = 1
        for j in range(1, self.hall_length+1):
            self.maze[j][3] = 1

        self.traffic_light = random.choice([2,3])
        self.maze[1][1] = self.traffic_light
        self.maze[1+self.hall_length][1] = 4
        self.maze[1+self.hall_length][5] = 2
        self.maze[self.agentpos[0]][self.agentpos[1]] = 5

        self.obs = self.get_obs()
        self.isDone = False

        return self.obs

    def get_obs(self):
        pos = self.agentpos
        l = self.maze[pos[0]-1][pos[1]]
        r = self.maze[pos[0]+1][pos[1]]
        u = self.maze[pos[0]][pos[1]+1]
        d = self.maze[pos[0]][pos[1]-1]
        obs = (l, r, d, u)
        return obs

    def step(self, action):
        self.global_time += 1
        reward = 0.

        if action == 0:
            obs = self.obs[0]
            if obs == 1:
                self.maze[self.agentpos[0]][self.agentpos[1]] = 1
                self.agentpos[0] -= 1
                self.maze[self.agentpos[0]][self.agentpos[1]] = 5

        elif action == 1:
            obs = self.obs[1]
            if obs == 1:
                self.maze[self.agentpos[0]][self.agentpos[1]] = 1
                self.agentpos[0] += 1
                self.maze[self.agentpos[0]][self.agentpos[1]] = 5

        elif action == 2:
            obs = self.obs[2]
            if obs == 1:
                self.maze[self.agentpos[0]][self.agentpos[1]] = 1
                self.agentpos[1] -= 1
                self.maze[self.agentpos[0]][self.agentpos[1]] = 5

        elif action == 3:
            obs = self.obs[3]
            if obs == 1:
                self.maze[self.agentpos[0]][self.agentpos[1]] = 1
                self.agentpos[1] += 1
                self.maze[self.agentpos[0]][self.agentpos[1]] = 5

        else:
            print("not a valid action")
            return

        reward = -.04

        self.obs = self.get_obs()
        if self.agentpos == [self.hall_length+1, 2] or self.agentpos == [self.hall_length+1, 4]:
            self.isDone = True
            if (2 in self.obs and self.traffic_light==2):
                reward = 1.
            if (2 in self.obs and self.traffic_light==3):
                reward = -1.
            if (4 in self.obs and self.traffic_light==3):
                reward = 1.
            if (4 in self.obs and self.traffic_light==2):
                reward = -1.

        if self.global_time >= 50:
            self.isDone = True

        return self.obs, reward, self.isDone

    def render(self, rate=.1):
        self.draw_gridworld()
        self.root.update_idletasks()
        self.root.update()
        time.sleep(rate)

    def sample_action(self):
        action = random.choice([0, 1, 2, 3])
        return action

    def draw_gridworld(self):
        s=20
        M=self.maze
        for i in range(len(M)):
            for j in range(len(M)):
                if M[i][j] == 0:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="black")
                if M[i][j] == 1:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="white")
                if M[i][j] == 2:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="red")
                if M[i][j] == 3:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="green")
                if M[i][j] == 4:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="blue")
                if M[i][j] == 5:
                    self.window.create_rectangle(s*i, s*j, s*(i+1), s*(j+1), fill="yellow")



# imaze = IMaze()
#
#
# while 1:
#     a = input("enter an action: ")
#     obs, reward, done = imaze.step(a)
#     imaze.render()
#     print(obs)
#     print(reward)
#     print(done)
