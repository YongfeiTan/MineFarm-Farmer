from __future__ import print_function
from __future__ import division

# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #3: Drawing

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from builtins import range
from past.utils import old_div

SIZE = 10
OBS_SIZE = 21
zombie_num = 3
animal_num = 5
# Maze_type = 'stained_glass'
Farm_type = 'grass'
zombie_list = ['Zombie']
Animal_list = ['Pig', 'Cow', 'Sheep']
# ACTION_DICT = {
#     0: 'move 1',  # Move one block forward
#     1: 'turn 1',  # Turn 90 degrees to the right
#     2: 'turn -1',  # Turn 90 degrees to the left
#     3: 'attack 1',  # Destroy block
# }

# <Inventory>
#                             <InventoryItem slot="0" type="diamond_sword"/>
# </Inventory>

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)


def EnitySpawn(list_entity, num):
    entity_list = []
    result = ''
    length = len(list_entity)
    for i in range(num):
        x, z = np.random.randint(-SIZE+1, SIZE-1, size=2)
        index = np.random.randint(0, length)
        while (x, z) in entity_list:
            x, z = np.random.randint(-SIZE+1, SIZE-1, size=2)
        result += f"<DrawEntity x='{x}' y='2' z='{z}' type='{list_entity[index]}'/>"
        entity_list.append((x, z))
    return result


# <AgentQuitFromTimeUp timeLimitMs='1000'/>
# grass is relative position in girdobservation
def XMLMapGenerator():
    bounder = SIZE + 1
    bounderneg = -SIZE - 1
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>MineFarm Zombie Killer</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>17000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;2;village"/>
                        <DrawingDecorator>''' + \
           "<DrawCuboid x1='{}' x2='{}' y1='2' y2='5' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
           "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
           f"<DrawCuboid x1='{bounder}' x2='{bounder}' y1='0' y2='5' z1='{bounderneg}' z2='{bounder}' type='sea_lantern'/>" + \
           f"<DrawCuboid x1='{bounderneg}' x2='{bounderneg}' y1='0' y2='5' z1='{bounderneg}' z2='{bounder}' type='sea_lantern'/>" + \
           f"<DrawCuboid x1='{bounderneg}' x2='{SIZE}' y1='0' y2='5' z1='{bounderneg}' z2='{bounderneg}' type='sea_lantern'/>" + \
           f"<DrawCuboid x1='{bounderneg}' x2='{SIZE}' y1='0' y2='5' z1='{bounder}' z2='{bounder}' type='sea_lantern'/>" + \
           EnitySpawn(zombie_list, zombie_num) + \
           EnitySpawn(Animal_list, animal_num) + \
           '''</DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                        <ServerQuitFromTimeUp timeLimitMs="50000" description="limit"/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>CS175MineFarm Farmer</Name>
                    <AgentStart>
                        <Placement x="0.5" y="2" z="0.5" yaw="0" pitch="15"/>
                    </AgentStart>
                    <AgentHandlers>
                        <ContinuousMovementCommands/>
                        <VideoProducer want_depth="true"> 
                            <Width>320</Width> 
                            <Height>240</Height>
                        </VideoProducer>
                        <MissionQuitCommands quitDescription="all_kill"/>
                        <RewardForDamagingEntity>
                            <Mob type='Sheep' reward='-100'/>
                            <Mob type='Cow' reward='-100'/>
                            <Mob type='Pig' reward='-100'/>
                            <Mob type='Zombie' reward='+50'/>
                        </RewardForDamagingEntity>
                        <RewardForTouchingBlockType>
                            <Block reward="-0.5" type="sea_lantern" behaviour="oncePerBlock"/>
                        </RewardForTouchingBlockType>
                        <RewardForMissionEnd rewardForDeath="-500.0">
                            <Reward description="all_kill" reward="0.0"/>
                            <Reward description="limit" reward="-100.0"/>
                        </RewardForMissionEnd>
                        <ChatCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <ObservationFromNearbyEntities>
                                <Range name="Entities" xrange="10" yrange="1" zrange="10"/>
                        </ObservationFromNearbyEntities>

                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                 <min x="-''' + str(int(OBS_SIZE) / 2) + '''" y="0" z="-''' + str(int(OBS_SIZE) / 2) + '''"/>
                                <max x="''' + str(int(OBS_SIZE) / 2) + '''" y="0" z="''' + str(int(OBS_SIZE) / 2) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

