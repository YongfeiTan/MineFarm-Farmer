try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import Mapbuild
from numpy.random import randint
from numpy.random import rand
from collections import defaultdict

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

# Hyperparameters
OBS_SIZE = 21
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 12000
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


# helper functions
def get_entities(entities):
    animal_all = defaultdict(dict)
    for ele in entities:
        if ele['name'] in Entity_LIST:
            X = ele['x']
            Z = ele['z']
            ids = ele['id']
            life = ele['life']
            name = ele['name']
            animal_all[ids] = {'name': name, 'x': X, 'z': Z, 'life': life}
    return animal_all


# Agent class
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
        if 'Xpos' in observe.keys():
            self.Xpos = observe['XPos']
            self.Zpos = observe['ZPos']
        if 'Life' in observe.keys():
            self.life = observe['Life']
        self.entites = get_entities(observe['Entities'])
        if 'Yaw' in observe.keys():
            self.yaw = observe['Yaw']
        if 'LineOfSight' in observe.keys():
            self.LineOfSight = observe['LineOfSight']['type']
        else:
            self.LineOfSight = None
        # self.pitch = observe['Pitch']
        if 'floorAll' in observe.keys():
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
        value = max(0.0001, math.sqrt(distdiff))
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

    def wallchanger(self):
        Xind = int(self.Xpos)
        Zind = int(self.Zpos)
        stack = False
        for i in range(OBS_SIZE):
            for j in range(OBS_SIZE):
                if self.wall[0, i, j] == 1 and abs(i + j - Xind - Zind) <= 2:
                    stack = True
                    break
        if self.LineOfSight == 'sea_lantern' and stack:
            return -1 * self.yaw / 180.0
        elif stack:
            return self.yaw / 180.0
        return 0

    # find best yaw to attack
    def find_yaw(self, entity):
        x_diff = max(0.000001, entity['x'] - self.Xpos)
        z_diff = max(0.000001, entity['z'] - self.Zpos)
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


# algorithm class
class MineFarmFarmer(gym.Env):
    def __init__(self, env_config):
        self.size = 21
        self.obs_size = 21
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {0: 'move 1',
                            1: 'move -1',
                            2: 'turn 1',
                            3: 'turn 0',
                            4: 'attack 1'}
        # self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, shape=(np.prod([1, self.obs_size, self.obs_size]),), dtype=np.int32)
        self.agent_host = MalmoPython.AgentHost()
        self.agentinf = agent()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
                Resets the environment for the next episode.

                Returns
                    observation: <np.array> flattened initial obseravtion
        """
        world_state = self.init_malmo()
        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
                len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state, self.agentinf)

        return self.obs.flatten()

    def step(self, action):
        """
            Take an action in the environment and return the results.
            Args
                action: <int> index of the action to take
            Returns
                observation: <np.array> flattened array of obseravtion
                reward: <int> reward from taking action
                done: <bool> indicates terminal state
                info: <dict> dictionary of extra information
        """
        # deal with action
        lifevalue = self.agentinf.life
        # action_command = self.action_dict[action]
        if self.agentinf.near is not None:
            yawtomob = self.agentinf.find_yaw(self.agentinf.entites[self.agentinf.near])
            self.agent_host.sendCommand("turn " + str(yawtomob))
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            self.obs = self.get_observation(world_state, self.agentinf)
            time.sleep(0.5)
        allow_break_action = False
        # print(agentinf.LineOfSight)
        # print(agentinf.entites[agentinf.near])
        if self.agentinf.LineOfSight in Entity_LIST:
            allow_break_action = True
        command = self.action_dict[action]
        self.agent_host.sendCommand(command)
        if command == 'attack 1':
            self.agent_host.sendCommand('attack 0')
        time.sleep(.5)
        # if allow_break_action and action[2] > 0:
        #     self.agent_host.sendCommand('move 0')
        #     self.agent_host.sendCommand('turn 0')
        #     self.agent_host.sendCommand('attack 1')
        #     time.sleep(1)
        # else:
        #     self.agent_host.sendCommand('attack 0')
        #     self.agent_host.sendCommand('move {:30.1f}'.format(action[0]))
        #     self.agent_host.sendCommand('turn {:30.1f}'.format(action[1]))
        #     time.sleep(2)

        # if action_command != 'attack 1' or allow_break_action:
        #     self.agent_host.sendCommand(action_command)
        #     time.sleep(.1)
        self.episode_step += 1
        # done part
        done = False
        if self.episode_step >= self.max_episode_steps or not self.agent_host.getWorldState().is_mission_running:
            done = True
            time.sleep(2)
            # obs part
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("\nError:", error.text)
        self.obs = self.get_observation(world_state, self.agentinf)
        # reward part
        reward = 0
        if self.agentinf.near is not None:
            info = self.agentinf.entites[self.agentinf.near]
            # info == {} bug, I can't solve it
            xent = info['x']
            zent = info['z']
            distdiff = (self.agentinf.Xpos - xent) ** 2 + (self.agentinf.Zpos - zent) ** 2
            if math.sqrt(distdiff) < self.agentinf.disttonear:
                reward += 1
            if lifevalue > self.agentinf.life:
                lifevalue = self.agentinf.life
                reward -= 5
            if allow_break_action and info['name'] == "Zombie" and self.agentinf.LineOfSight == 'zombie' and  command == 'attack 1':
                reward += 5
                if 2 < math.sqrt(distdiff) < 3:
                    reward += 3
            elif allow_break_action and info['name'] == "Zombie" and self.agentinf.LineOfSight == 'zombie':
                reward += 1
        for r in world_state.rewards:
            print(r.getValue(), '--------------------')
            reward += r.getValue()
        print(reward)
        self.episode_return += reward
        return self.obs.flatten(), reward, done, dict()

    def getMissonXML(self):
        return Mapbuild.XMLMapGenerator()

    def init_malmo(self):
        """
            Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.getMissonXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, 'MineFarm Farmer')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        self.agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:20}]}")

        return world_state

    def get_observation(self, world_state, agentinf):
        obs = np.zeros((1, OBS_SIZE, OBS_SIZE))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')
            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                agentinf.update(observations)
                grid = agentinf.entites
                # full observation or 5*5 observation
                for entity in grid.values():
                    obs[0, math.floor(entity['x']) + 5, math.floor(entity['z']) + 5] = 1

                # Rotate observation with orientation of agent
                if 'Yaw' in observations.keys():
                    yaw = observations['Yaw']
                else:
                    yaw = agentinf.yaw
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                break

        return obs

    def log_returns(self):
        """
            Log the current returns as a graph and text file
        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('MineFarm Farmer')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=MineFarmFarmer, config={
        'env_config': {},  # No environment parameters to configure
        'framework': 'torch',  # Use pyotrch instead of tensorflow
        'num_gpus': 0,  # We aren't using GPUs
        'num_workers': 0  # We aren't using parallelism
    })

    while True:
        print(trainer.train())
