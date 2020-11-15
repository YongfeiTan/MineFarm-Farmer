---
layout: default
title:  Status
---

#### Video
[Video Link](https://drive.google.com/file/d/1vrUaV2EGtWeD2R_JFOYxdc3E_1qZ_ah-/view?usp=sharing)
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ob-9S6akK1U" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary
In this project, we want to employ the AI agent to kill various animals on the map to attain as much profit as possible with the help of the implement of Q-Learning given a limit time. The agent may meet different types of animals and it needs to keep learning and improve its performance on identifying which animal is worth the most rewards and could bring the most potential value to the agent within the limit time.

## Approach
In this project, we use some basic skills of Reinforcement learning which is based on the environment observations and the reward/penalty mechanism to improve the agent’s performance.  

- **Observation**  
We edit the get_observation function which figures out the observation spaces for our project. In this function, it will receive all entities information from World_state.observations. It will collect all entities' positions, names, ids, and live. The function returns a np.array describing the status of each grid. The grid contains all animals information. If an animal inside a block, the block will return 1. Otherwise, it will return 0. 
```
def get_entities(entities):
	animal_all = defaultdict(dict)
	for ele in entities:
		if ele['name'] in ANIMAL_LIST and ele['x'] < int(OBS_SIZE/2) and ele['z']< int(OBS_SIZE/2):
			X = ele['x']
			Y = ele['y']
			Z = ele['z']
			ids = ele['id']
			name = ele['name']
			animal_all[ids] = {'name':name,'x':X,'y':Y,'z':Z}
	return animal_all
```
```
Entities = observations['Entities']
grid = get_entities(Entities)
for entity in grid.values():
    obs[0, math.floor(entity['x'])+5, math.floor(entity['z'])+5] = 1
```

- **Action Space**  
The action space is the exactly the same as the project of assignment 2 which includes move forward/left/right and destroy the block which is for killing the animal on the block. We implement the epsilon-greedy policy to select the following actions. In the next step, we may add `pitch` as a new action since some animals are too small to attack.
```
# may add 'pitch' in the future
ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'attack 1'  # Destroy block
}
```

- **Terminal states**  
Within a specific time limit. Currently, the limit is 60 seconds.

- We also implement the reward mechanism which gives a feedback to the agent when it each time kills the animal. The agent can learn from the score and gain the ability to identify more “valuable” animal to kill. 
  
```
<RewardForDamagingEntity>
    <Mob type='Sheep' reward='+1'/>
</RewardForDamagingEntity>
```


## Evaluation
#### Quantitative
In the final version, we will use AUC score to judge the agent’s performance to see whether it can maximize the profit and avoid killing animals (since killing animals will get penalty). 
We will also graph the agent’s average return and revenue per second to see the learning effect more directly. 
For the current project, we will use AUC score to judge the agent’s performance on identifying the animals. The agent should be trained to identify the most “valuable” animal and kill them first since the time is limit. 

#### Qualitative
In the final version, the agent should learn to kill as many zombies as possible as the time passes by. There should be a positive correlation between the agent’s return and time.
For the current project, the agent should learn to maximize its profit within the time limit. 

## Remaining Goals and Challenges
#### Remaining Goals
In the next four weeks, we will upgrade the map by adding some zombies which can move randomly with superpower to attack the agent if they meet each other. In the final project, the agent should learn to identify zombies and animals and then kill the zombie and protect animals (avoid killing animals). We will add penalty coefficient to penalize the agent if it kills animals by mistake. For attacking the zombie, the agent will move and locate the enemy's location. If the zombie is within the attack area of the agent, there will also be an attacking function in which the agent will attack the specific enemy (zombie) continuously until the enemy is killed. Then the agent will be looking for the next target and identify the enemy. The ultimate goal is to let the agent kill as many zombies as possible while protectin animals and gain as many points as possible within the time limit.  
  
Consider using a RL framework like RLlib with more powerful algorithms than what was used in assignment2. Try different algorithms and hyperparameters in RLlib.

#### Challenges 
When we design the AI, we notice many difficulties when we want AI to achieve goals. We need to calculate distance between AI and other entities and identify different entities. Since we intend to kill as many zombies as possible in a limited time, the agent should track the nearest entity.  
  
It has gaps between our plan and the implementation. We realize many problems after running our codes. It turns out that we ignored some steps when converting human-controlling agent into an AI. For example, our AI could not move since the commands being sent are invalid. The reason is that discrete commands only work if the agent is centered on a block (x and z starting positions must be at X.5 where X is some number). And the scenario is also differen when there are animals in the map. The agent could not move once it bumps into the animal. Now we are also facing the problem that the agent could not attack effectively.
 

## Resources Used
1. codes from Assigment2
2. XML website which designed by Microsoft. [link](https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html)
3. Tutorial files in Malmo project, helping us to understand tags in XML and how to achieve some specific goals.
