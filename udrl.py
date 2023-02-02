### Import Libraries
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, datetime
from matplotlib import pyplot as plt
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from numpy.random import randint
import dgl
from param import args
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple, deque

cuda = args.cuda


class GCN(nn.Module):

    def __init__(self, aggregator, features=5, hidden_layer_size=5, embedding_size=1):
        super(GCN, self).__init__()
        
        # Simple Graph Conv 
        # self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        # self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size)
        # self.conv3 = GraphConv(in_feats=hidden_layer_size, out_feats=embedding_size)

        # Using Graph SAGE to get embedding using the neighbours
        self.conv1 = SAGEConv(in_feats=features, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv2 = SAGEConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv3 = SAGEConv(in_feats=hidden_layer_size, out_feats=embedding_size, aggregator_type=aggregator)

    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.sigmoid(h)
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        h = self.conv3(g, h)
        h = torch.sigmoid(h)

        return h

class Net(nn.Module):

    def __init__(self, aggregator, features=5, hidden_layer_size=10, embedding_size=1):
        super().__init__()
        
        # The GNN for online training and the target which gets updated 
        # in a timely manner
        self.online = GCN(aggregator, features, hidden_layer_size, embedding_size)
        self.target = copy.deepcopy(self.online)
        #self.network = GCN()

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, graph, node_input, model="online"):
        # remove model argument 
        # logits = self.network(grpah, node_input)
        if model == "online":
            logits = self.online(graph, node_input)
        else :
            logits = self.target(graph, node_input)

        return logits 

class Agent():

    print("Entered Agent!")
    ### Hyperparameters ###

    # Number of iterations in the main loop
    n_main_iter = 700

    # Number of (input, target) pairs per batch used for training the behavior function
    batch_size = 768

    # Scaling factor for desired horizon input
    horizon_scale = 0.01

    # Number of episodes from the end of the replay buffer used for sampling exploratory
    # commands
    last_few = 75

    # Learning rate for the ADAM optimizer
    learning_rate = 0.0003

    # Number of exploratory episodes generated per step of UDRL training
    n_episodes_per_iter = 20

    # Number of gradient-based updates of the behavior function per step of UDRL training
    n_updates_per_iter = 100

    # Number of warm up episodes at the beginning of training
    n_warm_up_episodes = 10

    # Maximum size of the replay buffer (in episodes)
    replay_size = 500

    # Scaling factor for desired return input
    return_scale = 0.02

    # Evaluate the agent after `evaluate_every` iterations
    evaluate_every = 10

    # Target return before breaking out of the training loop
    target_return = 200

    # Maximun reward given by the environment
    max_reward = 250

    # Maximun steps allowed
    max_steps = 300

    # Reward after reaching `max_steps` (punishment, hence negative reward)
    max_steps_reward = -50

    # Hidden units
    hidden_size = 32

    # Times we evaluate the agent
    n_evals = 1

    # Will stop the training when the agent gets `target_return` `n_evals` times
    stop_on_solved = False
    #######################

    def __init__(self, save_dir="./models", assist=True, assist_p=(1, 7), aggregator="mean"):
        print("Initializing Agent... ")
            # Helper function to create episodes as namedtuple
        make_episode = namedtuple('Episode', 
                                field_names=['states', 
                                            'actions', 
                                            'rewards', 
                                            'init_command', 
                                            'total_return', 
                                            'length', 
                                            ])
        self.save_dir = save_dir

        # DNN to predict the most optimal action
        self.net = Net(aggregator).float()
        self.net = self.net.to(cuda)
        self.aggregator = aggregator

        self.exploration_rate = 1
        self.exploration_rate_decay = args.exploration_rate_decay # 0.9999975 # 0.999992 
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=100000)
        self.batch_size = args.batch_size

        self.save_every = 1e3  # no. of experiences

        self.gamma = args.gamma

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        #self.loss_fn = torch.nn.SmoothL1Loss()

        #self.burnin = args.burnin            # min. experiences before training
        self.learn_every = args.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = args.sync_every    # no. of experiences between Q_target & Q_online sync

        self.assist = assist
        self.assist_range = assist_p
        self.softmax = nn.Softmax()

    def load(self, file:str, new_rate=0.6):
        path = self.save_dir+"/"+file
        loaded_file = torch.load(path)
        self.net.load_state_dict(loaded_file["model"])
        # if exploration_rate:
        #     self.exploration_rate = loaded_file["exploration_rate"]
        # else:
        #     self.exploration_rate = new_rate
        self.net.eval()


    # move to Net class --> see what state_fn is doing and command_fn is doing 
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.command_scale)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)

    def action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- stochastic action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()
    
    ### On Assist
    def greedy_action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- greedy action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())


    ### Cache Replay Buffer
    def initialize_replay_buffer(replay_size, n_episodes, last_few):
        '''
        Initialize replay buffer with warm-up episodes using random actions.
        See section 2.3.1
        
        Params:
            replay_size (int)
            n_episodes (int)
            last_few (int)
        
        Returns:
            ReplayBuffer instance
            
        '''
        
        # This policy will generate random actions. Won't need state nor command
        random_policy = lambda state, command: np.random.randint(env.action_space.n)
        
        buffer = ReplayBuffer(replay_size)
        
        for i in range(n_episodes):
            command = sample_command(buffer, last_few)
            episode = generate_episode(env, random_policy, command) # See Algorithm 2
            buffer.add(episode)
        
        buffer.sort()
        return buffer

    def initialize_behavior_function(state_size, 
                                    action_size, 
                                    hidden_size, 
                                    learning_rate, 
                                    command_scale):
        '''
        Initialize the behaviour function. See section 2.3.2
        
        Params:
            state_size (int)
            action_size (int)
            hidden_size (int) -- NOTE: not used at the moment
            learning_rate (float)
            command_scale (List of float)
        
        Returns:
            Behavior instance
        
        '''
        
        behavior = Behavior(state_size, 
                            action_size, 
                            hidden_size, 
                            command_scale)
        
        behavior.init_optimizer(lr=learning_rate)
        
        return behavior

    ### Generate Episode
    def generate_episode(env, policy, init_command=[1, 1]):
        '''
        Generate an episode using the Behaviour function.
        
        Params:
            env (OpenAI Gym Environment)
            policy (func)
            init_command (List of float) -- default [1, 1]
        
        Returns:
            Namedtuple (states, actions, rewards, init_command, total_return, length)
        '''
        
        command = init_command.copy()
        desired_return = command[0]
        desired_horizon = command[1]
        
        states = []
        actions = []
        rewards = []
        
        time_steps = 0
        done = False
        total_rewards = 0
        state = env.reset().tolist()
        
        while not done:
            state_input = torch.FloatTensor(state).to(device)
            command_input = torch.FloatTensor(command).to(device)
            action = policy(state_input, command_input)
            next_state, reward, done, _ = env.step(action)
            
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
            
            # Sparse rewards. Cumulative reward is delayed until the end of each episode
    #         total_rewards += reward
    #         reward = total_rewards if done else 0.0
            
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

    def generate_episodes(env, behavior, buffer, n_episodes, last_few):
        '''
        1. Sample exploratory commands based on replay buffer
        2. Generate episodes using Algorithm 2 and add to replay buffer
        
        Params:
            env (OpenAI Gym Environment)
            behavior (Behavior)
            buffer (ReplayBuffer)
            n_episodes (int)
            last_few (int):
                how many episodes we use to calculate the desired return and horizon
        '''
        
        stochastic_policy = lambda state, command: behavior.action(state, command)
        
        for i in range(n_episodes_per_iter):
            command = sample_command(buffer, last_few)
            episode = generate_episode(env, stochastic_policy, command) # See Algorithm 2
            buffer.add(episode)
        
        # Let's keep this buffer sorted
        buffer.sort()


        ### Main Training Loop
        def learn_batch(selfenv, buffer=None, behavior=None, learning_history=[]):
            '''
            Upside-Down Reinforcement Learning main algrithm
            
            Params:
                env (OpenAI Gym Environment)
                buffer (ReplayBuffer):
                    if not passed in, new buffer is created
                behavior (Behavior):
                    if not passed in, new behavior is created
                learning_history (List of dict) -- default []
            '''
        
        if buffer is None:
            buffer = initialize_replay_buffer(replay_size, 
                                            n_warm_up_episodes, 
                                            last_few)
        
        if behavior is None:
            behavior = initialize_behavior_function(state_size, 
                                                    action_size, 
                                                    hidden_size, 
                                                    learning_rate, 
                                                    [return_scale, horizon_scale])
        
        for i in range(1, n_main_iter+1):
            mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
            
            print('Iter: {}, Loss: {:.4f}'.format(i, mean_loss), end='\r')
            
            # Sample exploratory commands and generate episodes
            generate_episodes(env, 
                            behavior, 
                            buffer, 
                            n_episodes_per_iter,
                            last_few)
            
            if i % evaluate_every == 0:
                command = sample_command(buffer, last_few)
                mean_return = evaluate_agent(env, behavior, command)
                
                learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })
                
                if stop_on_solved and mean_return >= target_return: 
                    break
        
        return behavior, buffer, learning_history

    def save(self, episode=0):
        save_path = (self.save_dir + f"/sched_net_{self.aggregator}_{episode}.pt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Sched_net saved to {save_path} at step {self.curr_step}")

### Replay Buffer
class ReplayBuffer():
    '''
    Replay buffer containing a fixed maximun number of trajectories with 
    the highest returns seen so far
    
    Params:
        size (int)
    
    Attrs:
        size (int)
        buffer (List of episodes)
    '''
    
    def __init__(self, size=0):
        self.size = size
        self.buffer = []
        
    def add(self, episode):
        '''
        Params:
            episode (namedtuple):
                (states, actions, rewards, init_command, total_return, length)
        '''
        
        self.buffer.append(episode)
    
    def get(self, num):
        '''
        Params:
            num (int):
                get the last `num` episodes from the buffer
        '''
        
        return self.buffer[-num:]
    
    def random_batch(self, batch_size):
        '''
        Params:
            batch_size (int)
        
        Returns:
            Random batch of episodes from the buffer
        '''
        
        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]
    
    def sort(self):
        '''Keep the buffer sorted in ascending order by total return'''
        
        key_sort = lambda episode: episode.total_return
        self.buffer = sorted(self.buffer, key=key_sort)[-self.size:]
    
    def save(self, filename):
        '''Save the buffer in numpy format
        
        Param:
            filename (str)
        '''
        
        np.save(filename, self.buffer)
    
    def load(self, filename):
        '''Load a numpy format file
        
        Params:
            filename (str)
        '''
        
        raw_buffer = np.load(filename)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]
    
    def __len__(self):
        '''
        Returns:
            Size of the buffer
        '''
        return len(self.buffer)

class MetricLogger():
    def __init__(self, save_dir="./results", mode="train", version="0", aggregator="mean"):
        save_dir = save_dir+"/"+mode
        self.save_log = save_dir + "/episodes_"+aggregator+"_"+version+".log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot_"+aggregator+"_"+version+".png"
        self.ep_lengths_plot = save_dir + "/length_plot_"+aggregator+"_"+version+".png"
        self.ep_avg_losses_plot = save_dir + "/loss_plot_"+aggregator+"_"+version+".png"
        self.ep_avg_qs_plot = save_dir + "/q_plot_"+aggregator+"_"+version+".png"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 6)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 6)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 6)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 6)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode :  {episode} \n"
            f"Step :  {step} \n"
            f"Epsilon :  {epsilon} \n"
            f"Mean Reward :  {mean_ep_reward} \n"
            f"Mean Length :  {mean_ep_length} \n"
            f"Mean Loss :  {mean_ep_loss} \n"
            f"Mean Q Value :  {mean_ep_q} \n"
            f"Time Delta :  {time_since_last_record} "
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('episodes')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()




# He has a class for behaviour -> behaviour is generally defined by a neural network.
# He has multiple steps to calculate state and command embeddings
# Which he passes through an output neural network
# In our case, we have an Net defining just the neural network, This will have the forward part 
# Here define what you want to do for state and command like what he has done
# In out case agent will have the action, greedy action, load, save, and udrl
# Please figure out what each line does --- Comment on every line if needed, google package name and see what it is used for
# if you don't understand
# Correlate what he has done with what You should. If you feel that his method is easier --> go with it ( and follow only one convention)