try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import random
import math
from collections import defaultdict
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import rand
import Mapbuild

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Hyperparameters

OBS_SIZE = 15
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
Animal_list = ['Pig', 'Cow', 'Sheep']
Entity_LIST = ['Pig', 'Cow', 'Sheep', 'Zombie']
Animal_life = {'Sheep': 8, 'Pig': 10, 'Cow': 10}
Zombie_life = 20
# ,  # Move one block forward
# 0: 'move 1'
#     1: 'turn 1',  # Turn 90 degrees to the right
#     2: 'turn -1',  # Turn 90 degrees to the left
#     3: 'attack 1',  # Destroy block
#     4: 'pitch 0.1',
#     5: 'pitch -0.1',
#     6: 'pitch 0'
ACTION_DICT = {
    0: 'move 1',
    1: 'move -1',
    2: 'turn 1',
    3: 'attack 1'  # Destroy block
}


# helper functions
def get_entities(entities):
    animal_all = defaultdict(dict)
    for ele in entities:
        if ele['name'] in Entity_LIST:
            X = ele['x']
            Z = ele['z']
            ids = ele['id']
            life = ele['life']
            print(life)
            name = ele['name']
            animal_all[ids] = {'name': name, 'x': X, 'z': Z, 'life': life}
    return animal_all


# -------------------------------------------------
# agent recording
class agent:
    def __init__(self):
        self.Xpos = 0.5
        self.Zpos = 0.5
        self.life = 20
        self.entites = {}
        self.yaw = 0
        self.pitch = 0
        self.near = None
        self.disttonear = None
        self.LineOfSight = None
        self.wall = []

    def update(self, observe):
        # print(observe.keys)
        # print(observe)
        self.Xpos = observe['XPos']
        self.Zpos = observe['ZPos']
        self.life = observe['Life']
        self.entites = get_entities(observe['Entities'])
        self.yaw = observe['Yaw']
        if 'LineOfSight' in observe.keys():
            self.LineOfSight = observe['LineOfSight']['type']
        else:
            self.LineOfSight = None
        self.pitch = observe['Pitch']
        self.wall = self.update_wall(observe['floorAll'])
        self.nearchecker()

    def update_wall(self, wall_info):
        obs = np.zeros((1, OBS_SIZE, OBS_SIZE))
        for i in range(OBS_SIZE):
            for j in range(OBS_SIZE):
                index = OBS_SIZE * i + j
                if wall_info[index] == 'sea_lantern':
                    obs[0, i, j] = 1
        return obs

    # check if there is a entity in observation
    def nearchecker(self):
        nearinfo = self.find_nearst()
        # if self.near is not None:
        #     return
        if nearinfo != 'no entity':
            self.near = nearinfo[0]
            self.disttonear = nearinfo[1]
        else:
            self.near = None
            self.disttonear = None

    def cal_distance(self, entity):
        xent = entity['x']
        zent = entity['z']
        distdiff = (self.Xpos - xent) ** 2 + (self.Zpos - zent) ** 2
        value = max(0.01, math.sqrt(distdiff))
        return value

    def find_nearst(self):
        distdiff = 100
        id_ent = 0
        for ele_id, info in self.entites.items():
            if info['name'] in Entity_LIST:
                newdistdiff = self.cal_distance(info)
                if distdiff > newdistdiff:
                    distdiff = newdistdiff
                    id_ent = ele_id
        if id_ent != 0:
            return id_ent, distdiff
        else:
            return 'no entity'

    # find best pitch to attack
    def find_pitch(self, entity):
        x_diff = self.Xpos - entity['x']
        z_diff = self.Zpos - entity['z']
        angle = abs(180 * math.atan2(x_diff, z_diff) / math.pi)
        if self.Xpos < entity['x'] and self.Zpos < entity['z']:
            self.agent.sendCommand("setYaw {}".format(-90 + angle))
        elif self.Xpos < entity['x'] and self.Zpos > entity['z']:
            self.agent.sendCommand("setYaw {}".format(-90 - angle))
        elif self.Xpos > entity['x'] and self.Zpos < entity['z']:
            self.agent.sendCommand("setYaw {}".format(90 - angle))
        elif self.Xpos > entity['x'] and self.Zpos > entity['z']:
            self.agent.sendCommand("setYaw {}".format(90 + angle))

    # find best yaw to attack
    def find_yaw(self, entity):
        x_diff = entity['x'] - self.Xpos
        z_diff = entity['z'] - self.Zpos
        yaw = self.yaw
        # y_diff = (entity['y'] + c.HEIGHT_CHART[entity['name']] / 2) - (y + 1.8)

        yaw_to_mob = -180 * math.atan2(x_diff, z_diff) / math.pi
        delta_yaw = yaw_to_mob - yaw

        while delta_yaw < -180:
            delta_yaw += 360
        while delta_yaw > 180:
            delta_yaw -= 360
        delta_yaw /= 180.0

        return delta_yaw
        # x_diff = self.Xpos - entity['x']
        # z_diff = self.Zpos - entity['z']
        # angle = abs(180 * math.atan2(x_diff, z_diff) / math.pi)
        # if self.Xpos < entity['x'] and self.Zpos < entity['z']:
        #     return -90 + angle
        # elif self.Xpos < entity['x'] and self.Zpos > entity['z']:
        #     return -90 - angle
        # elif self.Xpos > entity['x'] and self.Zpos < entity['z']:
        #     return 90 - angle
        # elif self.Xpos > entity['x'] and self.Zpos > entity['z']:
        #     return 90 + angle


# Q-Value Network
class QNetwork(nn.Module):
    # ------------------------------------
    #
    #   TODO: Modify network architecture
    #
    # -------------------------------------

    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))
        #

    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


def get_action(obs, q_network, epsilon, allow_break_action):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action

    Returns:
        action (int): chosen action [0, action_size)
    """
    # ------------------------------------
    #
    #   TODO: Implement e-greedy policy
    #
    # -------------------------------------

    # Prevent computation graph from being calculated
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        # Remove attack/mine from possible actions if not facing a diamond
        print(action_values)
        if not allow_break_action:
            print(1)
            action_values[0, 3] = -float('inf')
        size_act = len(action_values[0])
        if rand() < (1 - epsilon):
            # Select action with highest Q-value
            action_idx = torch.argmax(action_values).item()
        else:
            action_idx = randint(low=0, high=size_act, size=1)[0]
            while not allow_break_action and action_idx == 3:
                action_idx = randint(low=0, high=size_act, size=1)[0]
    return action_idx


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(Mapbuild.XMLMapGenerator(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "MineFarm")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


# first agent should have its class
#


def get_observation(world_state, agent):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
    The agent is in the center square facing up.

    Args
        world_state: <object> current agent world state

    Returns
        observation: <np.array>
    """
    obs = np.zeros((1, OBS_SIZE, OBS_SIZE))

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')
        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            agent.update(observations)
            grid = agent.entites
            # full observation or 5*5 observation
            for entity in grid.values():
                obs[0, math.floor(entity['x']) + 5, math.floor(entity['z']) + 5] = 1

            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))

            break

    return obs


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)

    return obs, action, next_obs, reward, done


def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('MineFarm Farmer')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value))


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((1, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((1, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        agentinf = agent()
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        obs = get_observation(world_state, agentinf)
        # Run episode
        numani = 0
        numzomb = 0
        for ele in agentinf.entites.values():
            if ele['name'] in Animal_list:
                numani += 1
            elif ele['name'] == 'Zombie':
                numzomb += 1
        while world_state.is_mission_running:
            # Get action
            # parameter change
            animaln = 0
            zombn = 0
            for ele in agentinf.entites.values():
                if ele['name'] in Animal_list:
                    animaln += 1
                elif ele['name'] == 'Zombie':
                    zombn += 1
            numzomb = zombn
            numani = animaln
            if numzomb == 0:
                agent_host.sendCommand('quit')
            if agentinf.near is not None:
                # print(agentinf.entites[agentinf.near])
                # print(agentinf.entites)
                yawtomob = agentinf.find_yaw(agentinf.entites[agentinf.near])
                print(yawtomob)
                agent_host.sendCommand(f"turn {yawtomob}")
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
                newobs = get_observation(world_state, agentinf)
            allow_break_action = False
            if agentinf.LineOfSight == 'Zombie':
                allow_break_action = True
            print(agentinf.LineOfSight, allow_break_action, '----------------')

            lifevalue = agentinf.life
            # allow_break_action = obs[0, int(OBS_SIZE / 2) - 1, int(OBS_SIZE / 2)] == 1
            action_idx = get_action(obs, q_network, epsilon, allow_break_action)
            command = ACTION_DICT[action_idx]
            # Take step
            print(command)
            print('-----------------')
            agent_host.sendCommand(command)
            # If your agent isn't registering reward you may need to increase this
            time.sleep(.5)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS:
                done = True
                time.sleep(2)
                # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state, agentinf)

            # reward part
            # penalty for attack animals are in the XML part
            reward = 0
            if agentinf.near is not None:
                info = agentinf.entites[agentinf.near]
                # info == {} bug, I can't solve it
                xent = info['x']
                zent = info['z']
                distdiff = (agentinf.Xpos - xent) ** 2 + (agentinf.Zpos - zent) ** 2
                if math.sqrt(distdiff) < agentinf.disttonear:
                    reward += 2
                if lifevalue > agentinf.life:
                    lifevalue = agentinf.life
                    reward -= 10
                else:
                    reward += 2
                if info['name'] == "Zombie" and agentinf.LineOfSight == 'zombie' and command == 'attack 1':
                    reward += 10
                elif info['name'] == "Zombie" and command == 'attack 1':
                    reward += 2

                print('reward', reward)
                # this checker may write in the XML
                # if info['name'] == 'zombie' and info['life'] < Zombie_life:
                #     reward += 20

            for r in world_state.rewards:
                print(r.getValue, '--------------------')
                reward += r.getValue()
            episode_return += reward
            # -------------------------------------------------------------------------------
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description(
            'Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
                num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    print(1)
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)
