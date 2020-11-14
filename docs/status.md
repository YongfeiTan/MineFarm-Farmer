---
layout: default
title:  Status
---

## Project Summary
In this project, our main goal is to employ the AI agent to kill various animals on the map to attain as much profit as possible with the help of the implement of Q-Learning given a limit time. The agent may face different types of animals and it needs to keep learning and improve its performance on identifying which animal is worth the most rewards and could bring the most potential value to the agent within the limit time.

## Approach
In this project, we use some basic skills of Reinforcement learning which is based on the environment observations and the reward/penalty mechanism to improve the agent’s performance.  

- Observation: We edit the get_observation function which figures out the observation spaces for our project. In this function, it will receive all entities information from World_state.observations. It will collect all entities' positions, names, ids, and live. The function returns a np.array describing the status of each grid. The grid contains all animals information. If an animal inside a block, the block will return 1. Otherwise, it will return 0. 

- Action Space: The action space is the exactly the same as the project of assignment 2 which includes move forward/backward/left/right and destroy the block which is for killing the animal on the block. We implement the epsilon-greedy policy to select the following actions.  In the next step, we may add pitch as a new action since some animals are too small to attack. 


We also implement the reward mechanism which gives a feedback to the agent when it each time kills the animal. The agent can learn from the score and gain the ability to identify more “valuable” animal to kill.  



## Evaluation
#### Quantitative
In the final version, we will use AUC score to judge the agent’s performance to see whether it can maximize the profit and avoid killing animals (since killing animals will get penalty). 
We will also graph the agent’s average return and revenue per second to see the learning effect more directly. 
For the current project, we will use AUC score to judge the agent’s performance on identifying the animals. The agent should be trained to identify the most “valuable” animal and kill them first since the time is limit. 

#### Qualitative
In the final version, the agent should learn to kill as many zombies as possible as the time passes by. There should be a positive correlation between the agent’s return and time.
For the current project, the agent should learn to maximize its profit within the time limit. 

## Remaining Goals and Challenges
In the next four weeks, we will upgrade the map by adding some zombies which can move randomly with superpower to attack the agent if they meet each other. In the final project, the agent should learn to identify zombies and animals and then kill the zombie and protect animals (avoid killing animals). We will add penalty coefficient to tell penalize the agent if it kills animals by mistake. For attacking the zombie, the agent will move and locate the enemy's location. If the zombie is within the attack area of the agent, there will also be an attacking function in which the agent will attack the specific enemy (zombie) continuously until the enemy is killed. Then the agent will be looking for the next target and identify the enemy. The ultimate goal is to let the agent kill as many zombies as possible and gain as many points as possible within the time limit. When we design the AI, we notice many difficulties when we want AI to achieve goals. We need to calculate distance between AI and other entities and identify different entities. Since we intend to kill as many zombies as possible in a limited time, the agent should track the nearest entity. It has gaps between our thinking and the goal. We usually realized some problems after we run our code. We ignore some steps when we convert human-controlling agent into an AI. In the future, we intend to training AI to kill zombies. 

## Resources Used
1. we use code from Assigment2
2. XML website which designed by Microsoft. https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html
3. Tutorial files help us understand tags in XML and how to achieve some specific goals. It includes in Malmo project
