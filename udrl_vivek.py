### Import Libraries
from base64 import encode
from typing import List, Tuple
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from environment_wrapper import *
from dgl.nn.pytorch import SAGEConv
import numpy as np
import datetime
from matplotlib import pyplot as plt
import pickle
from spark_env.canvas import *

warnings.filterwarnings("ignore")

# actions
RANDOM = 1
BEHAVIOUR_ACT = 2
BEHAVIOUR_GREEDY = 3

# config -max scheduling decisions
MAX_STEPS_REWARD = -50
MAX_REWARD = 0

# ------------------------------------------------------------------------------------------------

### GCN function to get the embedding
class GCN(nn.Module):

    def __init__(self, aggregator, features, hidden_layer_size, embedding_size, device="cpu"):
        super(GCN, self).__init__()
        
        # Simple Graph Conv 
        # self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        # self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size)

        # Using Graph SAGE to get embedding using the neighbours
        self.conv1 = SAGEConv(in_feats=features, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv2 = SAGEConv(in_feats=hidden_layer_size, out_feats=embedding_size, aggregator_type=aggregator)
        self.to(device)
        
    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.tanh(h)
        h = self.conv2(g, h)
        return h # torch.tanh(h) # TODO: change to h for pre saved models
        # return torch.sigmoid(h)

# -------------------------------------------------------------------------------------------------

### Behavior Function
class Behaviour(nn.Module):
    
    def __init__(self, aggregator='pool', features=5, hidden_layer_size=5, 
                embedding_size=10, command_scale = [1, 1], device='cpu'):

        super().__init__()

        # Check which device to be used for training and test
        self.device = device
        
        self.command_scale = torch.FloatTensor(command_scale).to(self.device)
        
        # Using Graph SAGE to get embedding for the graph
        self.gcn = GCN(aggregator, features, hidden_layer_size, embedding_size, device)
        # Use a sequential layer to get ecoding of the reward and command
        self.command_fn = nn.Sequential(nn.Linear(2, embedding_size)).to(device)

        # Output function
        self.output_nn = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh()
        ).to(device)
    
    #  Neural Network -
    #
    #  [10]- node embedding [10] - graph summary  |  [20] - command
    #  [20] - graph input  |  [20] - command
    #  [20] - graph input 
    #  [input - 20] -- layer 1 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 2 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 3 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 4 (sigmoid) -- [output - 1] -- Probability of scheduling node

    # Compute the action from a state and command
    # Note: this is only for computing a single action and not for training
    def action(self, state, command):
        G, node_inputs, leaf_nodes = state
        predictions = self.forward(state, command)
        index = leaf_nodes[predictions.argmax().item()]
        return index

    # Overloaded forward function
    def forward(self, state, command):
        G, node_inputs, leaf_nodes = state
        gnn_embeddings = self.gcn(G.to(self.device), node_inputs.to(self.device))
        command_input = torch.FloatTensor(command).to(self.device) * 0.0001
        encoded_command = self.command_fn(command_input)
        product = torch.mul(gnn_embeddings[leaf_nodes], encoded_command)
        return self.output_nn(product)

    # Get the random action for state
    def random_action(self, env:GraphWrapper, assist_probability=50):
        G, node_inputs, leaf_nodes = env.observe()
        action = leaf_nodes[(random.randint(0, len(leaf_nodes)-1))]
        if random.randint(0, 100-1) < assist_probability :
            action = env.auto_step()
        return action

    # save the model parameters
    def save(self, filename:str):
        torch.save(self.gcn.state_dict(), "./models/behaviour_gcn_"+filename)
        torch.save(self.command_fn.state_dict(), "./models/behaviour_cfn_"+filename)
        torch.save(self.output_nn.state_dict(), "./models/behaviour_nn_"+filename)
    
    # load model parameters
    def load(self, filename:str):
        if not filename:
            print("Not loading file weights")
            return
        print("loading file weights")
        self.gcn.load_state_dict(torch.load("./models/behaviour_gcn_"+filename))
        self.command_fn.load_state_dict(torch.load("./models/behaviour_cfn_"+filename))
        self.output_nn.load_state_dict(torch.load("./models/behaviour_nn_"+filename))

# -----------------------------------------------------------------------------------------------------

class Episode :

    def __init__(self, init_command:List) :
        self.list = []
        self.total_reward = 0   
        self.time_steps = 0
        self.loss = 0.0
        self.init_command = init_command

    def add_iteration(self, step:Tuple, loss=0):
        # G, node_inputs, leaf_nodes, action, reward
        if len(step) != 5:
            print("Length of each episode has to be 5")
            return
        self.total_reward += step[4] 
        self.time_steps += 1
        self.loss += loss
        self.list.append(step)

    def sample(self):
        total_len = len(self.list)
        start = random.randint(0, total_len-2); end = random.randint(start+1, total_len-1)
        horizon = end-start
        G, node_inputs, leaf_nodes, action, reward = self.list[start]
        tot_rewards = sum([observation[4] for observation in self.list[start:end+1]])
        actions = []
        for leaf in leaf_nodes:
            if action == leaf:
                actions.append(1)
            else:
                actions.append(-1)
        actions = torch.FloatTensor(actions).reshape(1, len(leaf_nodes))
        return G, node_inputs, actions, leaf_nodes, [tot_rewards, horizon]

    def __lt__(self, other):
        return self.total_reward > other.total_reward

    def __len__(self):
        return len(self.list)

### Replay Buffer for episodes
class Memory:

    def __init__(self, load_file=None, load_version=0, should_load=False):
        
        self.buffer = []
        if should_load:
            print("Loading warmup data file..")
            self.load(load_file, load_version)
        
    def add_episode(self, episode:Episode):
        self.buffer.append(episode)
        self.sort()
    
    def sample(self, batch_size:int):
        batch_size = min(len(self.buffer), batch_size)
        return np.random.choice(self.buffer, 
            size=min(batch_size, len(self.buffer)-1), replace=False)

    def get_possible_command(self, count=10):
        rewards = []; steps = []
        for episode in self.buffer[:(min(count, len(self.buffer)))]:
            rewards.append(episode.total_reward)
            steps.append(episode.time_steps)
        possible_horizon = np.mean(steps)
        rewards_std = min(np.std(rewards), 1)
        rewards_mean = np.mean(rewards)
        possible_reward = np.random.uniform(rewards_mean, rewards_mean-rewards_std)

        return [possible_reward, possible_horizon]
         
    def sort(self):
        self.buffer.sort()
    
    def load(self, data_file:str, load_version:int):
        print("loading training data")
        with open("./warmup/warmup_"+data_file+"_"+str(load_version)+".dat", "rb") as f:
            self.buffer = pickle.load(f)
        self.sort()
        
    def save(self, name:str, version:int):
        if (not name) or (not version):
            print("Not saving data : filename not provided")
            return
        print("Saving Training Data")
        with open("./warmup/warmup_"+name+"_"+str(version)+".dat", "wb") as f:
            pickle.dump(self.buffer, f)

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------------------------------------------

class MetricLogger:

    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        self.epoch = 1

        # running rewards graphs
        self.running_rewards_plot = save_dir + "/reward_plot_running.png"
        self.running_lengths_plot = save_dir + "/length_plot_running.png"
        self.running_losses_plot = save_dir + "/loss_plot_running.png"
        self.moving_avg_running_rewards = []
        self.moving_avg_running_lengths = []
        self.moving_avg_running_losses = []
        self.running_rewards = []
        self.running_lengths = []
        self.running_losses = []
        self.reset_run()

        # episode reward graphs
        self.ep_rewards_plot = ""
        self.ep_lengths_plot = ""
        self.ep_losses_plot = ""
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_losses = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_losses = []

        # Timing
        self.record_time = time.time()

    def log_step(self, reward:float, loss:float):
        self.running_reward += reward
        self.running_length += 1
        self.running_loss += loss

    def log_timestep(self):
        self.running_rewards.append(self.running_reward)
        self.running_lengths.append(self.running_length)
        ep_loss = np.round(self.running_loss / self.running_length, 3)
        self.running_losses.append(ep_loss)
        self.epoch += 1
        self.reset_run()

    def log_episode(self, episode:Episode, mode, aggregator, version:int) :
        self.ep_rewards_plot = self.save_dir +"/"+mode+ "/reward_plot_"+aggregator+"_"+str(version)+".png"
        self.ep_lengths_plot = self.save_dir +"/"+mode+ "/length_plot_"+aggregator+"_"+str(version)+".png"
        self.ep_losses_plot = self.save_dir +"/"+mode+ "/loss_plot_"+aggregator+"_"+str(version)+".png"
        self.ep_rewards.append(episode.total_reward)
        self.ep_lengths.append(episode.time_steps)
        self.ep_losses.append(np.round(episode.total_reward / episode.time_steps, 3))
        self.moving_avg_ep_rewards.append(np.round(np.mean(self.ep_rewards[-30:]), 3))
        self.moving_avg_ep_lengths.append(np.round(np.mean(self.ep_lengths[-30:]), 3))
        self.moving_avg_ep_losses.append(np.round(np.mean(self.ep_losses[-30:]), 3))
        for metric in ["ep_rewards", "ep_lengths", "ep_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

    def reset_run(self):
        self.running_reward = 0.0
        self.running_length = 1
        self.running_loss = 0.0

    def clear_episodes(self):
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_losses = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_running_lengths = []
        self.moving_avg_running_losses = []
        self.moving_avg_running_rewards = []

    def add_running_record(self, running_log:int):
        if running_log != 0:
            return
        self.log_timestep()
        mean_reward = np.round(np.mean(self.running_rewards[-30:]), 3)
        mean_length = np.round(np.mean(self.running_lengths[-30:]), 3)
        mean_loss = np.round(np.mean(self.running_losses[-30:]), 3)
        self.moving_avg_running_rewards.append(mean_reward)
        self.moving_avg_running_lengths.append(mean_length)
        self.moving_avg_running_losses.append(mean_loss)
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)
        # print(f"Epoch :  {self.epoch} | "
        #     f"Mean Reward :  {mean_reward} | "
        #     f"Time Delta :  {time_since_last_record} |")
        with open(self.save_dir+"/running_log.log", "a") as f:
            f.write(f"{self.epoch:8d}{mean_reward:15.3f}{mean_length:15.3f}"
                f"{mean_loss:15.3f}{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")
        for metric in ["running_rewards", "running_lengths", "running_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
        self.ep_rewards = self.ep_rewards[-60:]
        self.ep_lengths = self.ep_lengths[-60:]
        self.ep_losses = self.ep_losses[-60:]

    def gen_cummulative_dist_fn(self, rewards:List, aggregator:str, version:int) :
        figure = plt.figure()
        subplot = figure.add_subplot(111)
        x, y = compute_CDF(rewards)
        print(x)
        print(y)
        subplot.plot(x, y, color="red", label="UDRL Sched")
        plt.xlabel('Total reward')
        plt.ylabel('CDF')
        figure.savefig(self.save_dir+'/cdf_'+aggregator+'_'+str(version)+'.png')
        

# ---------------------------------------------------------------------------------------------------------

class Runner:

    def __init__(self, env: GraphWrapper, behaviour: Behaviour, logger: MetricLogger, memory: Memory) -> None:
        self.env = env
        self.behaviour = behaviour
        self.logger = logger
        self.memory = memory

    def run_random_episode_with_assist(self, run_version:int, assist=50, seed=None):
        episode = Episode([1, 1])
        seed = seed; done = False
        if not seed :
            seed = random.randint(1, 100000)
        self.env.reset(seed=seed)

        while not done :
            state = self.env.observe() 
            index = self.behaviour.random_action(self.env, assist)
            ns, reward, done = self.env.step((index, 1), False)
            if done:
               reward = 50
            G, node_inputs, leaf_nodes = state
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward], 0)

        self.logger.log_episode(episode, mode="random", aggregator=str(assist), version=run_version)
        print("---------------------------------------------------------------------------------------------------------------")
        print("Ended episode | Done : ", done, " | Steps : ", episode.time_steps, " | Reward :", episode.total_reward, " |")
        print("---------------------------------------------------------------------------------------------------------------")
        return episode

    def run_episode_with_learned_action(self, version:int, aggregator:str, device:str, seed=None, command=[-1766.31, 4258.0]):
        episode = Episode(command)
        seed = seed; done = False
        if not seed:
            seed = random.randint(1, 100000)
        self.env.reset(seed=seed)
        state = self.env.observe()

        while not done:
            index = self.behaviour.action(state, command)
            next_state, reward, done = self.env.step((index, 1), False)
            G, node_inputs, leaf_nodes = state  
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])
            state = next_state
            command[0] = min(command[0]-reward, MAX_REWARD) # desired reward
            command[1] = max(command[1]-1, 1)               # desired time frame
        
        self.logger.log_episode(episode, mode="test", aggregator=aggregator, version=version)
        print("---------------------------------------------------------------------------------------------------------------")
        print("Ended episode | Done : ", done, " | Steps : ", episode.time_steps, " | Reward :", episode.total_reward, " |")
        print("---------------------------------------------------------------------------------------------------------------")
        return episode

    def generate_warmup_data(self, iterations:int, data_file_name:str, data_version:int, assist=100, seeds=None, save=True, gradient=0.6) :
        print("Generating warmup data")
        if seeds:
            for seed in seeds:
                episode = self.run_random_episode_with_assist(seed=seed, run_version=data_version, assist=assist)
                self.memory.add_episode(episode)
                assist *= gradient
        else :
            for i in range(iterations):
                episode = self.run_random_episode_with_assist(run_version=data_version, assist=assist)
                self.memory.add_episode(episode)
                assist *= gradient

        self.logger.clear_episodes()
        if save:
            self.memory.save(version=str(data_version), name=data_file_name) 

    def test_learned_behaviour(self, version:int, agrgegator:str, device:str, seeds=None, 
                            iterations=10, command=[-1747.9001939176676, 5529.0]):
        print("Testing Learned Behaviour")
        rewards = []
        if seeds:
            for seed in seeds:
                episode = self.run_episode_with_learned_action(version=version, 
                            aggregator=agrgegator, device=device, seed=seed, command=command)
                rewards.append(episode.total_reward)
        else :
            for i in range(iterations):
                episode = self.run_episode_with_learned_action(version=version, 
                            aggregator=agrgegator, device=device, command=command)
                rewards.append(episode.total_reward)

        logger.gen_cummulative_dist_fn(rewards, aggregator=agrgegator, version=version)


# ---------------------------------------------------------------------------------------------------------

class Trainer : 

    def __init__(self, device, version, memory: Memory, logger: MetricLogger, env: GraphWrapper,
                behaviour = Behaviour, optimizer=Adam, lr=0.00001, ) -> None:
        
        # initialize the objects required
        self.memory = memory
        self.logger = logger
        self.env = env
        self.version = version
        self.behaviour = behaviour
        self.device = device
            
        self.optimizer = optimizer(self.behaviour.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def run_episode(self, iteration:int, aggregator:str, version:int, command=[1, 1], seed=None, train_params={"steps":5, "batch": 10}, running_log=100):
        # init seed if not initialized already
        if not seed :
            seed = random.randint(1, 100000)

        # init the episode params
        episode = Episode(init_command=command)
        self.env.reset(seed)
        state = self.env.observe()
        done = False; loss = 0
        
        while not done :
            loss = self.train_batch(steps=train_params["steps"], batch_size=train_params["batch"])
            index = self.behaviour.action(state, command)
            next_state, reward, done = self.env.step((index, 1), False)
            if done:
                reward = 50
            
            G, node_inputs, leaf_nodes = state  
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])
            
            state = next_state                              # next state
            command[0] = min(command[0]-reward, MAX_REWARD) # desired reward
            command[1] = max(command[1]-1, 1)               # desired time frame

            self.logger.log_step(reward, loss)
            self.logger.add_running_record(episode.time_steps % running_log)

        self.logger.log_episode(episode=episode, mode="train", aggregator=aggregator, version=version)
        
        print("---------------------------------------------------------------------------------------------------------------")
        print("Ended episode | Done : ", done, " | Steps : ", episode.time_steps, " | Reward :", episode.total_reward, " |")
        self.behaviour.save("scheduling_"+str(self.version)+"_"+str(iteration)+"_"+aggregator+".pt")
        print("---------------------------------------------------------------------------------------------------------------")
        
        return episode

    def train_batch(self, steps=1, batch_size=5):
        loss_total = 0
        self.optimizer.zero_grad()
        for step in range(steps):
            episodes = self.memory.sample(batch_size)
            for episode in episodes:
                graphs, inputs, actions, leaves, command = episode.sample()
                predictions = self.behaviour((graphs, inputs, leaves), command)
                loss = self.loss_fn(predictions.reshape(1, len(leaves)), actions.to(self.device))
                loss.backward()
                loss_total += loss.item()
        self.optimizer.step()
        return loss_total

    def start_udrl(self, aggregator:str, version:int, iterations=100, command_batch=5, running_log=100):
        print("Starting upside down reinforcement learning...")
        for i in range(iterations):
            command = self.memory.get_possible_command(count=command_batch)
            print("Command: ", command)
            episode = self.run_episode(i, aggregator=aggregator, command=command.copy(), running_log=running_log, version=version)
            self.memory.add_episode(episode)

# ---------------------------------------------------------------------------------------------------------

# neural network params
nn_aggregator = "mean"
nn_version = 3
nn_device = "cuda"
nn_file = "scheduling_2_3_mean.pt"
train_episodes = 10

# warmup parameters
warmup_file_version = 3
warmup_gradient = 1
warmup_assist = 100
warmup_episodes = 5
warmup_file_name = str(warmup_episodes)+"_ep_"+str(warmup_assist)+"_"+str(warmup_gradient)+"_assist"
generate_warmup = False

# test params
test_command = [-1747.9001939176676, 5529.0]
seeds = [8738,  9029, 182, 9832, 9335, 3162, 10212, 10523, 12083, 1380, 887, 1304, 6905, 7318, 
        7634, 4422, 5597, 8190, 10023, 11435, 7639, 3308, 12014, 906, 6027]
iterations = 10


logger = MetricLogger()
env = GraphWrapper()

# init behaviour function
behaviour = Behaviour(aggregator=nn_aggregator, device=nn_device)
behaviour.load(nn_file)
memory = Memory(load_file=warmup_file_name, load_version=warmup_file_version, should_load=(not generate_warmup))
runner = Runner(env=env, behaviour=behaviour, logger=logger, memory=memory)
logger = MetricLogger()
if generate_warmup :
    runner.generate_warmup_data(iterations=warmup_episodes, data_file_name=warmup_file_name, 
        data_version=warmup_file_version, assist=warmup_assist, gradient=warmup_gradient)

# Uncomment the below lines to train
# trainer = Trainer(device=nn_device, version=nn_version, memory=memory, logger=logger, env=env, behaviour=behaviour)
# trainer.start_udrl(iterations=train_episodes, aggregator=nn_aggregator, command_batch=5, running_log=100, version=nn_version)

# Uncomment the below lines to test
runner.test_learned_behaviour(version=nn_version, agrgegator=nn_aggregator, device=nn_device, 
        seeds=seeds, iterations=iterations,command=test_command)

# Version 

# DO NOT DELETE
# scheduling_0_15_mean.pt command [-1766.31, 4258.0]
# scheduling_2_*_mean.pt command [-1747.9001939176676, 5529.0]