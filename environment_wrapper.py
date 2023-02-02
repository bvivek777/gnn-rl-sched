from operator import le
from spark_env.env import Environment, JobDAG, Node
from param import args
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import time
import torch
import numpy as np

cuda = args.cuda

class GraphWrapper:
    
    def __init__(self, reset_prob=5e-7) -> None:

        # Set pre built environment
        self.env = Environment()
        
        # environment parameters
        self.reset_prob = reset_prob
        self.max_exec = args.exec_cap

        self.frontier_nodes = []
        self.leaf_nodes = []
        self.source_exec = args.exec_cap
    
    # reset the environment to a new seed
    def reset(self, seed:int):
        self.env.seed(seed)
        self.env.reset(max_time=np.random.geometric(self.reset_prob))
        self.offset = 0
        self.logits = False
        self.observe()

    # observe and decode an observation into usable paramaters for the agent
    # this involves embedding the graph nodes into 10 dim vectors
    def observe(self):
        # get the new observation from the environement
        G, frontier_nodes, leaf_nodes, num_source_exec, node_inputs = self.env.observe()

        # reset the frontier nodes and the number of free executors
        self.frontier_nodes = frontier_nodes
        self.source_exec = num_source_exec
        self.leaf_nodes = leaf_nodes

        return G, node_inputs, leaf_nodes

    # perform an action and return the resultant state and reward
    def step(self, action, early_stop=False):

        # set the job index as per the leaf nodes
        index, limit = action

        # count the frontier nodes and 
        frontier_node_count = len(self.frontier_nodes)

        # if no more frontier nodes then clear the executors
        if frontier_node_count == 0 or index < 0:
            reward, done = self.env.step(None, self.max_exec)
            state = self.observe()
            return state, reward, early_stop or done
            
        # limits for the number of executors
        if limit > self.source_exec :
            limit = self.source_exec
        limit = max(1, limit)

        # take a step and observe the reward, completion and the state from the old environement
        reward, done = self.env.step(self.frontier_nodes[index], limit)
        state = self.observe()
        
        return state, reward, done
    
    # to get and display the graph
    def get_networkx(self):
        return self.env.G, self.env.pos

    # test the heuristic action reward
    def auto_step(self, step=False):
        node, limit = self.get_heuristic_action()
        
        if not step:
            if node :
                return self.env.action_map.inverse_map[node]
            else :
                return -1
        reward, done = self.env.step(node, limit)
        return reward, done

    # compute the heuristics to perform an action
    def get_heuristic_action(self):
        # Get frontier nodes from the graph
        frontier_nodes = self.env.get_frontier_nodes()

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([
            any(n.next_task_idx + self.env.exec_commit.node_commit[n] + self.env.moving_executors.count(n) < n.num_tasks for n in job_dag.nodes) \
            for job_dag in self.env.job_dags ])

        # compute the executor cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

        # sort out the exec_map
        exec_map = {}
        for job_dag in self.env.job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in self.env.moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in self.env.exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in self.env.exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += self.env.exec_commit.commit[s][n]

        scheduled = False
        # first assign executor to the same job
        if self.env.source_job is not None:
            # immediately scheduable nodes
            for node in self.env.source_job.frontier_nodes:
                if node in frontier_nodes:
                    return node, self.env.num_source_exec
            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == self.env.source_job:
                    return node, self.env.num_source_exec

        # the source job is finished or does not exist
        for job_dag in self.env.job_dags:
            if exec_map[job_dag] < exec_cap:
                next_node = None
                # immediately scheduable node first
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break
                # then schedulable node in the job
                if next_node is None:
                    for node in frontier_nodes:
                        if node in job_dag.nodes:
                            next_node = node
                            break
                # node is selected, compute limit
                if next_node is not None:
                    use_exec = min(
                        node.num_tasks - node.next_task_idx - \
                        self.env.exec_commit.node_commit[node] - \
                        self.env.moving_executors.count(node),
                        exec_cap - exec_map[job_dag],
                        self.env.num_source_exec)
                    return node, use_exec

        # there is more executors than tasks in the system
        return None, self.env.num_source_exec