from gym import spaces, Env
import numpy as np
import pygame


# I define a class BypassingEnv containing all the methods required to create my custom environment.
class BypassingEnv(Env):

    # In this initialisation I define:
    # - the array
    # - the starting position of the agent which is the AI position at the beginning
    # - the goal position which is the player position
    # - the current position which is equal to the start position at the beginning
    # - the number of rows and columns
    # - the number of actions we need: 4 (up, down, right, left)
    # - the game initialisation to see the results

    def __init__(self, bypass):
        super(BypassingEnv, self).__init__()
        self.bypass = np.array(bypass)
        self.start_pos = np.where(self.bypass == 8)
        self.goal_pos = np.where(self.bypass == 4)
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.bypass.shape

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=225, shape=self.bypass.shape, dtype=np.float64)

        pygame.init()

        self.cell_size = 70
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    # The reset method allows the agent to explore different initial conditions or states.
    def reset(self):
        self.bypass = np.array(self.bypass)
        self.start_pos = np.where(self.bypass == 8)
        self.goal_pos = np.where(self.bypass == 4)
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.bypass.shape
        return self.bypass

    # The step method define all the action (move) my agent could do.
    # I add reward conditions to guide my agent as best I can.
    # The rewards are distributed as follows:
    #       100 if the agent is on the goal place
    #       0 for all other position
    #       +10 if the agent is on the boxes next to the obstacles
    #       -50  if the agent is on the three squares to the left of the starting point
    def step(self, action):
        new_pos = np.array(self.current_pos)
        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] -= 1
        elif action == 3:
            new_pos[1] += 1

        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 100.0
            done = True
        else:
            reward = 0.0

            for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                neighbor_pos = (self.current_pos[0] + i, self.current_pos[1] + j)
                if self._is_in_matrix(neighbor_pos) and self.bypass[neighbor_pos] == 1:
                    reward += 10

            start_x, start_y = self.start_pos[0][0], self.start_pos[1][0]
            if (self.current_pos[1][0] < start_y) and (start_x - 2 < self.current_pos[0][0] < start_x + 2):
                reward -= 50.0
            done = False

        return np.zeros((self.num_rows, self.num_cols)), reward, done, {}

    # I create this method to verify if my agent do a valid move.
    # So my agent can't go into the obstacles or out of the matrix.
    def _is_valid_position(self, pos):
        row, col = pos

        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_rows or self.bypass[row, col] == 1:
            return False
        return True

    def _is_in_matrix(self, pos):
        row, col = pos

        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_rows:
            return False
        return True

    # The render is here to see all the results on our screen.
    def render(self):
        self.screen.fill((255, 255, 255))

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if self.bypass[row, col] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.bypass[row, col] == 8:
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.bypass[row, col] == 4:
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1, 1)):
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()
