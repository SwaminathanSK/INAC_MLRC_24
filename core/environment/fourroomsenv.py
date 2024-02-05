import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

class FourRooms:

    def __init__(self, start = [10, 0], goal = [0, 10], seed = None, discount = 0.9, WORLD_SIZE = 11):

        self._max_episode_steps = np.inf
        self.seed = seed
        self.action_dim = 4
        
        self.WORLD_SIZE = WORLD_SIZE
        self.discount = discount
        self.world = np.zeros((WORLD_SIZE, WORLD_SIZE))
        

        self.state = start
        self.goal = goal
        self.li = []
        reward_goal = 1

        # left, up, right, down
        self.actions = ['L', 'U', 'R', 'D']

        leftgap = [5, 1] # left gap
        rightgap = [6, 8] # right gap
        upgap = [2, 5] # up gap
        downgap = [9, 5] # down gap


        self.actionProb = []
        for i in range(0, WORLD_SIZE):
            self.actionProb.append([])
            for j in range(0, WORLD_SIZE):
                self.actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

        self.nextState = []
        self.actionReward = []
        for i in range(0, WORLD_SIZE):
            self.nextState.append([])
            self.actionReward.append([])
            for j in range(0, WORLD_SIZE):
                next = {'R' : [i, j+1], 'L' : [i, j-1], 'D' : [i+1, j], 'U' : [i-1, j]}
                reward = {'R' : 0, 'L' : 0, 'D' : 0, 'U' : 0}

                if [i, j] == leftgap or [i, j] == rightgap:
                    next['R'] = [i, j]
                    next['L'] = [i, j]
                    if [i + 1, j] == goal:
                        reward['D'] = reward_goal
                    if [i - 1, j] == goal:
                        reward['U'] = reward_goal

                elif [i, j] == upgap or [i, j] == downgap:
                    next['U'] = [i, j]
                    next['D'] = [i, j]
                    if [i, j+1] == goal:
                        reward['R'] = reward_goal
                    if [i, j-1] == goal:
                        reward['L'] = reward_goal

                else:
                    if i == leftgap[0]+1 and j != leftgap[1] and j < downgap[1]:
                        next['U'] = [i, j]
                    elif i == rightgap[0]+1 and j != rightgap[1] and j > downgap[1]:
                        next['U'] = [i, j]
                    elif i == 0:
                        next['U'] = [i, j]
                    elif [i-1, j] == goal:
                        reward['U'] = reward_goal

                    if i == leftgap[0]-1 and j != leftgap[1] and j < upgap[1]:
                        next['D'] = [i, j]
                    elif i == rightgap[0]-1 and j != rightgap[1] and j > upgap[1]:
                        next['D'] = [i, j]
                    elif i == WORLD_SIZE - 1:
                        next['D'] = [i, j]
                    elif [i+1,j] == goal:
                        reward['D'] = reward_goal

                    if j == upgap[1]+1 and i != upgap[0] and i < rightgap[0]:
                        next['L'] = [i, j]
                    elif j == downgap[1]+1 and i != downgap[0] and i > rightgap[0]:
                        next['L'] = [i, j]
                    elif j == 0:
                        next['L'] = [i, j]
                    elif [i, j-1] == goal:
                        reward['L'] = reward_goal

                    if j == upgap[1]-1 and i != upgap[0] and i < leftgap[0]:
                        next['R'] = [i, j]
                    elif j == downgap[1]-1 and i != downgap[0] and i > leftgap[0]:
                        next['R'] = [i, j]
                    elif j == WORLD_SIZE - 1:
                        next['R'] = [i, j]
                    elif [i, j+1] == goal:
                        reward['R'] = reward_goal

                self.nextState[i].append(next)
                self.actionReward[i].append(reward)


        # def draw_image(image):
        #     fig, ax = plt.subplots()
        #     ax.set_axis_off()
        #     tb = Table(ax, bbox=[0,0,1,1])

        #     nrows, ncols = image.shape
        #     width, height = 1.0 / ncols, 1.0 / nrows

        #     # Add cells
        #     for (i,j), val in np.ndenumerate(image):
        #         # Index either the first or second item of bkg_colors based on
        #         # a checker board pattern
        #         idx = [j % 2, (j + 1) % 2][i % 2]
        #         color = 'white'

        #         tb.add_cell(i, j, width, height, text=val,
        #                     loc='center', facecolor=color)

        #     # Row Labels...
        #     for i, label in enumerate(range(len(image))):
        #         tb.add_cell(i, -1, width, height, text=label+1, loc='right',
        #                     edgecolor='none', facecolor='none')
        #     # Column Labels...
        #     for j, label in enumerate(range(len(image))):
        #         tb.add_cell(-1, j, width, height/2, text=label+1, loc='center',
        #                         edgecolor='none', facecolor='none')
        #     ax.add_table(tb)
        #     plt.show()

        # li = []
        ## top wall
        for i in range(0, WORLD_SIZE):
            if i != upgap[0] and i != downgap[0]:
                self.li.append((i, upgap[1]))

        for j in range(0, upgap[1]):
            if j != leftgap[1]:
                self.li.append((leftgap[0], j))

        for j in range(upgap[1], WORLD_SIZE):
            if j != rightgap[1]:
                self.li.append((rightgap[0], j))
    
    def reset(self):
        return self.state
    
    def step(self, a):
        a = self.actions[a]
        i = self.state[0]
        j = self.state[1]
        newPosition = self.nextState[i][j][a]
        self.state = newPosition
        info = None
        reward = self.actionReward[i][j][a]
        if reward == 1:
            done = True
        else:
            done = False

        # state, reward, done, info = self.step(a)
        return np.asarray(self.state), np.asarray(reward), np.asarray(done), info
