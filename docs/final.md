---
layout: default
title:  Status
---

## Video
<iframe width="610" height="335" src="https://www.youtube.com/embed/DqTQjLdnMu4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary
In this project, the agent will learn to identify zombies and animals and then kill the zombies and protect animals (avoid killing animals). We add penalty coefficient to penalize the agent if it kills animals by mistake. For attacking the zombie, the agent will move and locate the enemy’s location. If the zombie is within the attack area of the agent, there will also be an attacking function in which the agent will attack the nearest enemy (zombie) continuously. Then the agent will be looking for the next target and identify the enemy. The ultimate goal is to enable the agent kill as many zombies as possible while protecting animals and gain as many points as possible within the time limit.  
<br/>
In the final map, 3 zombies and 5 animals will be spawned randomly in a `21*21` map. It includes walls which are comprised of `sea_lantern`. Since zombie cannot stay alive in real light, we use `sea_lantern` as an artificial light. We set 50s(real time) as the time limitation to achieve its goal. We intent to kill as many zombies as possible at each time, and prevent to hurt animals. When we achieved our goal, there are many problems. We should design appropriate reward functions, action functions, and observe functions. In our final project, we mainly focus on setting up good `action_dict` list, reward functions, and different methods.  
<br/>
The agent has huge freedom inside Minecraft. It is possible to write a strict functions to make action. It has to explore actions and receives information from environment. Since it has to do a series of actions to achieve our goal, it need to distinguish good actions and bad actions in each state. Therefore, it is necessary to use RL algorithm.  
<br/>
<img width="500" alt="environment" src="https://user-images.githubusercontent.com/24601423/102731627-2e83bd00-42ed-11eb-94b4-5d4e7139370b.png">
<br/>

## Approach
In this project, we use some basic skills of Reinforcement learning which is based on the environment observations and the reward/penalty mechanism to improve the agent’s performance. We tried DQN and PPO algorithm in our final project. We intend to make a comparison between this two algorithms.   
- Baseline: The agent can kill at least two zombies  
- Best result: The agent can kill all zombies in a short time


### Implemetation
- **Observation**  
  We will return an one layer `21*21` blocks. If the block includes an entity, it will equal to 1. Otherwise, it uses 0. Furthermore, the agent will get wall information at each step.

- **Actions**  
Our agent has the following actions:
```
ACTION_DICT = {
    0: 'move 1',
    1: 'move -1',
    2: 'turn 1',
    3: 'turn 0',
    4: 'attack 1'
}
```  
  We didn't include the `attack 0` in the action_dict since it makes the actions not fluent. After the agent receives the command `attack 1`, we will send `attack 0` in train function. Hence, agent has to choose `attack 1` to attack an entity per step.

- **Reward Functions**  
  - R(s) = +50, Agent kills a zombie.   
  - R(s) = +5, Agent attacks zombie.
  - R(s) = +3, Agent attacks zombie with a safe distance.
  - R(s) = +1, Agent looks at a zombie.
  - R(s) = +1, Agent approached to an entity.
  - R(s) = -500, Agent becomes dead.
  - R(s) = -100, Agent runs out of time.
  - R(s) = -100, Agent kills an animal. 
  - R(s) = -5, Agent attacked by zombie.
  - R(s) = -0.5, Agent touches the wall.
  
  In this version, we tried many rewards. When we ran the game, we noticed that the agent would hide in a corner and attack zombies. We thought it could not kill zombies efficiently and safely. Hence, we tried to prevent agent to approach walls. It doesn't work as we want. Hence, we added a negative reward when it touches the walls. We encouraged the agent to keep a smaller distance each time since agent need to keep attacking range to kill zombies efficiently. 

- **Terminal State**  
Within a specific time limit. Currently, the limit is 50 seconds. If agent kills all zombies less than 50s, it will quit.


### Algorithm
We used DQL and PPO methods to help improve the performance.

#### Deep Q Learning (DQN)
<img width="600" alt="DQN" src="https://user-images.githubusercontent.com/24601423/102749955-c8615f00-4319-11eb-80ea-aced6f62fc9b.png">

  - Store the past exploring experience of the agent 
  
  - Determine the maximum output of the last action 
  
  - Update Q-value table using Bellman Equation 

  $$ Q(S, A) = Q(S, A) + \alpha [R + \gamma max_aQ(S', a) - Q(S,A)] $$
  
  S = the State or Observation  

  A = the Action the agent takes  

  R = the Reward from taking an Action  

  - Advantage: Find a good strategy quickly. In our project, It usually start get positive reward at 2000 steps.  
  - Disadvantage: It will get higher overfit because it keeps its strategy after many iterations. When we train the agent, agent will continuously alive at most 3 times. After that, it will be killed with poor behavior. The agent's strategy doesn't change when the state has been changed. It shows agent usually find local maximum but not global maximum. 


#### Proximal Policy Optimization (PPO)  
<img width="600" alt="PPO" src="https://user-images.githubusercontent.com/24601423/102749803-80423c80-4319-11eb-802b-7d8f9cd9b517.png">
  We tried both discrete action and continuous action list. We do train agent with this algorithm, but it has bug on our code. Agent cannot get reward from XML map. Its behavior doesn't as good as we think because of this problem. We cannot finish the training even we fix this problem. We make this comparison according to articles which introduce this two method. And Since PPO is a more complex and suitable method, agent should get better result with this algorithm. 

  $$ L(\theta)=\hat E_t[\min(r_t(\theta)\hat A_t, clip(r_t(\theta),1-\epsilon, 1+\epsilon)\hat A_t)] $$
  - Advantage: PPO should eventually get better behavior than DQN since DQN will get higher overfit.Since PPO will have longer time to explore new actions, it is possible that agent will find a global maximum.
  - Disadvantage: More time(PPO will explores more than DQN method)



## Evaluation
### Quantitative
The plots below show the average reward over time. They show that the performance of the agent improves over time. We try our agent with different reward functions. Normally, it will get about 500-700 reward when it kills one zombie. It is oscillating because its behavior may not perfect. It may hurt animals when agent attacks zombies. We notice the agent will be limit by the wall since it doesn't know it has a wall on the fringe. We add a negative penalty if agent touches wall. It gets better behavior since it has higher reward and better behavior. About 8000 steps, its behavior improves a lot since it can kill 3 zombies in 20s. It won't running over the map like before. In our best version, it will kill 687 mobs and save their life 44 times. On average, it has 30 steps in one iteration. We find this expect value through ratio between episode and step among all our tries. Since it has 13000 steps in the picture. It tries about 434 times in our best version. It alives 44 times and kills 687 mobs. It can kill 1.58 mobs in this version. Our baseline is that agent can kill at least two zombies after it gets training. It shows agent can successfully to kill 2 zombies after 400 iterations.  
- 1 zombie --> 500-700 reward
- 2 zombies --> 1000-1400 reward
- 3 zombies --> 1300-1600 reward

The corner issue really limits the agent's behavior. It will be a good direction to improve our porject in the future.

- DQN without wall penalty:  

<img width="500" alt="DQN_without_wall_penalty" src="https://user-images.githubusercontent.com/24601423/102706504-5d475800-4247-11eb-9a6b-cbcbf7b2ef91.png">

- DQN with wall penalty:  

<img width="500" alt="DQN_wall_penalty" src="https://user-images.githubusercontent.com/24601423/102729138-4d318600-42e4-11eb-93ad-89cea2eeaaff.png">  

The following plot is the average reward for random choices. All the decisions/actions the agent makes are random. There is no increase of the reward, which is much worse than the one using DQN.  
- Ramdom choice:  

<img width="500" alt="random_choice" src="https://user-images.githubusercontent.com/24601423/102749734-60ab1400-4319-11eb-8b78-7af9da6b3972.png">

- Best Result Record:  

<img width="900" alt="record" src="https://user-images.githubusercontent.com/24601423/102742856-d78ce080-430a-11eb-8ac1-8845c4116aeb.png">


### Qualitative
In the beginning of our training process, the agent moves randomly and is always killed easily by zombies. The reward is thus always negative. The Death penalty is 500. In first 70 iterations, the agent is killed without killing any zombie, so the reward is nearly -500. There is no any reward that is possible to counteract with this penalty if the agent doesn't kill any zombie. After roughly 2000 (70 iterations), the agent's performance gradually becomes better, it starts to kill zombies, and it is less attacked by zombies. Sometimes, it can kill all the zombies. Among 434 iterations, it kills at least 2 zombies with more than 70 times. We consider it as killing two zombies if it can get more than 1000 reward. It should have more than it since it has death penalty and the penalty of killing animal. Thus, the reward becomes positive and higher. Comparing these two pictures, we can notice that the agent's behavior still oscillate after 70 iterations. Without wall penalty, it oscillate from 0 to 1300. However, if it has this wall penalty, it usually oscillate from 300 to 1800. It proves behavior of agent improves with this penalty. And they both shows that agent behavior improves as they are training. 

## References
- Malmo Community 
- Malmo Example Codes    
[Python Examples](https://github.com/microsoft/malmo/tree/master/Malmo/samples/Python_examples): hit_test.py, assigment2.py, assigment2rlib.py, chat_reward.py, mob_fun.py
- Library  
[RLlib](https://docs.ray.io/en/latest/rllib.html)
[PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
(supplied by TA)
- Papers  
Fighting Zombies in MineCraft With Deep Reinforcement Learning [Link](http://cs229.stanford.edu/proj2016/report/UdagawaLeeNarasimhan-FightingZombiesInMinecraftWithDeepReinforcementLearning-report.pdf)
