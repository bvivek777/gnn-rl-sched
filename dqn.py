import copy
from collections import deque
from environment_wrapper import GraphWrapper
import random
from dgl.convert import graph
import numpy as np
import torch
import torch.nn as nn
import time, datetime
from matplotlib import pyplot as plt
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from numpy.random import randint
import dgl
from param import args

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

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, graph, node_input, model="online"):
        if model == "online":
            logits = self.online(graph, node_input)
        else :
            logits = self.target(graph, node_input)

        return logits  

class Agent():

    def __init__(self, save_dir="./models", assist=True, assist_p=(1, 7), aggregator="mean"):
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
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = args.burnin            # min. experiences before training
        self.learn_every = args.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = args.sync_every    # no. of experiences between Q_target & Q_online sync

        self.assist = assist
        self.assist_range = assist_p
        self.softmax = nn.Softmax()

    def act(self, state, assist_index=None):
        
        G, node_inputs, leaf_nodes = state
        
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            if assist_index:
                action_idx = assist_index
            else:
                index = randint(len(leaf_nodes))
                action_idx = (leaf_nodes[index], 1)

        # EXPLOIT
        else:
            logits = self.net(G.to(cuda), node_inputs.to(cuda), model="target")
            req    = torch.argmax(logits[leaf_nodes]).item()
            action_idx = (leaf_nodes[req], 1)
            del logits


        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        action = torch.tensor(action)

        state, node_inputs, leaf_nodes = state
        next_state, next_node_inputs, next_leaf_nodes = next_state

        self.memory.append((state, node_inputs, next_state, next_node_inputs, next_leaf_nodes, action, reward, done))

    def recall(self):
        return random.sample(self.memory, self.batch_size)

    def td_estimate_batch(self, state, action):
        # actions need to be re indexed based on batching
        state, node_inputs = state
        current_Q = self.net(state.to(cuda), node_inputs.to(cuda), model="online")[action.to(cuda)]
        return current_Q

    @torch.no_grad()
    def td_target_batch(self, reward, next_state, done):

        # leaf_nodes need to be re constructed as (ending_index, leaf_nodes)
        next_state, node_inputs, leaf_nodes = next_state
        indices = []; 
        prev = 0
        # calculate the logits for online model
        logits = self.net(next_state.to(cuda), node_inputs.to(cuda), model="online")
        logits = logits.detach()

        # seperate the actions per graph
        for i, leaves in leaf_nodes:

            req = logits[prev:i, :][leaves]
            index = torch.argmax(req).item()
            indices.append(leaves[index]+prev)
            prev = i

        # calculate the next Q values
        next_Q = self.net(next_state.to(cuda), node_inputs.to(cuda), model="target")[indices]

        rewards = (reward + (1 - done.float()) * self.gamma * next_Q.to("cpu")).float()

        return  rewards

    def update_Q_batch(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target.to(cuda))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss = loss.item()
        return total_loss

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.target.state_dict())

    def save(self, episode=0):
        save_path = (self.save_dir + f"/sched_net_{self.aggregator}_{episode}.pt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Sched_net saved to {save_path} at step {self.curr_step}")

    def get_batch(self):
        # Sample from memory
        memory = self.recall()

        length_current = 0; length_next = 0 
        graphs = []; node_input_list = []; actions = []; rewards = []; done_list = []
        next_graphs = []; next_leaf_nodes = []; next_node_inputs = []
        indices = []

        for sample in memory:
            graph, node_inputs, next_graph, next_node_input, next_leaves, action, reward, done = sample

            # actions have to be appended before length is increased
            actions.append(action + length_current)
            length_current += graph.number_of_nodes()

            # append values to respective lists to create batches
            graphs.append(graph)
            node_input_list.append(node_inputs)
            rewards.append(reward)
            done_list.append(done)

            # set the length of the nodes to be used for next_state
            length_next += next_graph.number_of_nodes()

            # append values to lists to create next_state
            next_graphs.append(next_graph)
            next_node_inputs.append(next_node_input)
            next_leaf_nodes.append((length_next, next_leaves))
            indices.append(length_next)

        # create tensors to train as batches
        batch_state = (dgl.batch(graphs), torch.cat(node_input_list))
        batch_actions = torch.stack(actions)
        batch_rewards = torch.stack(rewards)
        batch_next_state = (dgl.batch(next_graphs), torch.cat(next_node_inputs), next_leaf_nodes)
        batch_done = torch.stack(done_list)

        return batch_state, batch_next_state, batch_actions, batch_rewards, batch_done

    def learn_batch(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, actions, rewards, done = self.get_batch()

        # Get TD Estimate       
        td_est = self.td_estimate_batch(state, actions)

        # Get TD Target
        td_tgt = self.td_target_batch(rewards, next_state, done)
        
        # get the loss function
        loss = self.update_Q_batch(td_est, td_tgt)

        return td_est.flatten().mean().item(), loss

    def load(self, file:str, exploration_rate=False, new_rate=0.6):
        path = self.save_dir+"/"+file
        loaded_file = torch.load(path)
        self.net.load_state_dict(loaded_file["model"])
        if exploration_rate:
            self.exploration_rate = loaded_file["exploration_rate"]
        else:
            self.exploration_rate = new_rate
        self.net.eval()

    def direct_train(self, env:GraphWrapper, episodes=10):

        for i in range(episodes):

            state = env.observe()
            done = False

            while True:

                action = env.step(state)
                exit()

class MetricLogger:
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