### Import Libraries
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
from numpy import array, load
from environment_wrapper import *
from numpy.random import randint
import time
from dgl.nn.pytorch import SAGEConv

warnings.filterwarnings("ignore")

# Traing on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = GraphWrapper()
env.reset(123)
G, node_inputs, leaf_nodes = env.observe()
replay_buffer_size = 0
last_few = 75
max_steps_reward = -50
max_steps = 300
max_reward = 250
n_episodes = 10
# Helper function to create episodes as namedtuple
make_episode = namedtuple('Episode', 
                          field_names=['states', 
                                       'actions', 
                                       'rewards', 
                                       'init_command', 
                                       'total_return', 
                                       'length', 
                                       ])

### Sample Exploratory Commands
def sample_command(buffer, last_few):
    if len(buffer) == 0: return [1, 1]
    
    # 1.
    commands = buffer.get(last_few)
    
    # 2.
    lengths = [command.length for command in commands]
    desired_horizon = round(np.mean(lengths))
    
    # 3.
    returns = [command.total_return for command in commands]
    mean_return, std_return = np.mean(returns), np.std(returns)
    desired_return = np.random.uniform(mean_return, mean_return+std_return)
    
    return [desired_return, desired_horizon]

### random actions to be taken 
def random_action(leaf_nodes:list):
    index = randint(len(leaf_nodes))
    print(index, "Index")
    return (leaf_nodes[index], 1)

### Generate Episode --> needs a lot of change, define max reward
def generate_episode(env:GraphWrapper, init_command=[1, 1], random_policy=True):
    command = init_command.copy()
    desired_return = command[0]
    desired_horizon = command[1]
    
    states = []; actions = []; rewards = []
    
    time_steps = 0; done = False; total_rewards = 0
    env.reset()
    G, node_inputs, leaf_nodes = env.observe()
    
    while not done:

        #Check the policy
        if random_policy:
            action = random_action(leaf_nodes)
        else :
            # replace the below line with learned policy
            action = random_action(leaf_nodes)

        next_state, reward, done, _ = env.step(action)
        
        G, node_inputs, leaf_nodes = next_state
        # Modifying a bit the reward function punishing the agent, -100, 
        # if it reaches hyperparam max_steps. The reason I'm doing this 
        # is because I noticed that the agent tends to gather points by 
        # landing the spaceshipt and getting out and back in the landing 
        # area over and over again, never switching off the engines. 
        # The longer it does that the more reward it gathers. Later on in 
        # the training it realizes that it can get more points by turning 
        # off the engines, but takes more epochs to get to that conclusion.
        if not done and time_steps > max_steps:
            done = True
            reward = max_steps_reward
        ## -------------------------------- You still need to make sure that you add these to state | in our case state -> G, node_inputs 
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state.tolist()
        # Clipped such that it's upper-bounded by the maximum return achievable in the env
        desired_return = min(desired_return - reward, max_reward)
        
        # Make sure it's always a valid horizon
        desired_horizon = max(desired_horizon - 1, 1)
    
        command = [desired_return, desired_horizon]
        time_steps += 1
        
    return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)


while True:
    # Take a random action 
    index = randint(len(leaf_nodes))
    print(index, "Index")
    action = (leaf_nodes[index], 1)
    # Initialize Replay Buffer
    print(type(leaf_nodes), "Leaf Nodes Type")
    print(type(action), "Action Type")
    for a in action:
        print(a, "1 Action Value")
  

    print(leaf_nodes, "Leaf Nodes")
    print(action, "Random Action among the leaf nodes")
    print(index, "Random index position in the leaf node array")
 
    # Agent performs action
    next_state, reward, done = env.step(action, False)
    print(next_state, "Next State")
    print(type(next_state))

    print(reward, "Reward")
    print(type(reward))
    # Update state
    G, node_inputs, leaf_nodes = next_state
    replay_buffer_size = 0
    replay_buffer = []
    for i in range(n_episodes):
        command = sample_command(replay_buffer, last_few)
        episode = generate_episode(env, command) # See Algorithm 2
        replay_buffer.add(episode)
    
    # replay_buffer.sort()
    # Calculate Actions
    # leaf_nodes need to be re constructed as (ending_index, leaf_nodes)
    next_state, node_inputs, leaf_nodes = next_state
    indices = []; 
    prev = 0
    # calculate the logits for online model
    logits = net(next_state.to(cuda), node_inputs.to(cuda), model="online")

    # seperate the actions per graph
    for i, leaves in leaf_nodes:
        req = logits[prev:i, :][leaves]
        index = torch.argmax(req).item()
        indices.append(leaves[index]+prev)
        prev = i

    # calculate the next Q values
    next_Q = net(next_state.to(cuda), node_inputs.to(cuda), model="target")[indices]
    rewards = (reward + (1 - done.float()) * self.gamma * next_Q.to("cpu")).float()

    input()

    if done:
        break