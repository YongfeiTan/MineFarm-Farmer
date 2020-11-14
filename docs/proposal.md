---
layout: default  
title: Proposal  
---

# MineFarm Minecraft Maze

## Summary of the Project

We intend to implement an agent that can solve a maze. The map will be a N x N square which some blocks form the maze. It is different from a normal maze since the agent can explore the whole map. If the agent is not in the maze, it will get punishment. Animals will ramble over the map, and the maze includes zombies and coins. The agent should pick up coins, find a blade, and kill zombies before it leaves the maze. The agent needs to know the difference between zombies and animals.  We will input the whole map, starting point, and ending point. The agent acts randomly at the beginning and finds a series of optimistic actions. The output will be the game score, action list and agent track.

## AI/ML Algorithm
We will use Q-learning to optimize the agent’s actions and machine learning algorithms to distinguish animals and zombies.  

## Evaluation Plan

### Quantitative

- Numeric: We will use AUC to judge the agent’s performance when it classifies zombies and animals. We also use the time that the agent knows it can’t leave the maze and finds out the blade to judge the agent’s performance.  

- Baselines: The agent only move on the maze and finds out the shortest path to the ending point at any point.  

### Qualitative

- Simple example cases: It should find out the blade quickly and kill as many zombies as possible. It should pick up many coins and finds out the best way to leave the maze.  

- Error Analysis: We will use the agent’s action list and its track to check our error. If it doesn’t work, we will output some screenshots which the agent is doing actions.  

- The super-Impressive Example: In a limited time, it can kill all zombies and finds out all coins without hurt any animal.  


## Appointment with the Instructor
Time: 11:30am, Thursday, Oct 22

## Group Meeting
3:00pm, TueThu

