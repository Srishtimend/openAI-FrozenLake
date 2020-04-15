#!/usr/bin/env python
# coding: utf-8

# In[4]:


import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0') #initialize frozen lake environment 
env.reset()
env.render() #loads the environment GUI

epsilon = 0.9
total_episodes = 10000 #Total number of episodes - Number of times we're going to run the game
max_steps = 100 

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
#used to generate Q values 
"""
A random number is generated :
If it’s smaller, then a random action is chosen 
using env.action_space.sample() and if it’s greater 
then we choose the action having the maximum value in the Q-table for state: state
"""
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start the game
for episode in range(total_episodes):
    state = env.reset()
    t = 0 #timestep 
    
    while t < max_steps:
        env.render() #load environment to GUI

        action = choose_action(state)  #action chosen using the epsilong - greedy effect
        
        state2, reward, done, info = env.step(action)  #reward is returned - the agent learns from the process

        learn(state, state2, reward, action) # 

        state = state2

        t += 1
       
        if done:
            break

        time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
def choose_action(state):
    action = np.argmax(Q[state, :])
    return action

# start
for episode in range(5):

    state = env.reset()
    print("*** Episode: ", episode)
    t = 0
    while t < 100:
        env.render()

    action = choose_action(state)  
    state2, reward, done, info = env.step(action)  
    state = state2
    if done:
        break

        time.sleep(0.5)
        os.system('clear')
    


# In[ ]:




