from numpy import load
from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time

# ------------------------------------------------------------------------------------------------------------ #
#                                                  DQN
# ------------------------------------------------------------------------------------------------------------ #
def dqn_train(load_path=None, episodes=31, version=0, aggregator="mean", prob=100, exp_rate=True, new_rate=None):
    env    = GraphWrapper()
    agent  = Agent(aggregator=aggregator)
    logger = MetricLogger(version=str(version), mode="train", aggregator=aggregator)

    # load the presaved model
    if load_path:
        agent.load(load_path, exploration_rate=exp_rate, new_rate=new_rate)

    index = 0
    # seeds = [9968, 8726, 11869, 6750, 5684, 6008, 9276, 5, 3283, 203, 5808, 5596, 6043, 9571, 5579]

    for e in range(episodes):
        
        seed = randint(12345)
        env.reset(seed)
        state = env.observe()
        done = False
        start = time.time()
        total_reward = 0; total_actions = 0

        # Run the simulation
        while True:

            # Assist based on probability 
            if randint(0, 100) < prob:
                index = env.auto_step()
                action = agent.act(state, (index, 2))
            else:
                action = agent.act(state)
 
            # Agent performs action
            next_state, reward, done = env.step(action, False)
            total_reward += reward
            total_actions += 1

            # Remember
            agent.cache(state, next_state, action[0], reward, done)

            # Learn
            q, loss = agent.learn_batch()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            if done:
                break
        
        agent.save(version)
        torch.cuda.empty_cache()
        logger.log_episode()

        print('--------------------------------------------------------------------------------------------------------------------------------------------')
        print("Completed episode :", e, "Time Taken:", time.time()-start, "reward", total_reward, "seed :", seed, "actions :", total_actions)
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
        print('--------------------------------------------------------------------------------------------------------------------------------------------')

# assist seeds for mean : [9968, 8726, 11869, 6750, 5684, 6008, 9276, 5, 3283, 203, 5808, 5596, 6043, 9571, 5579]
# uncomment below 2 lines and set decay = 0.9998 in params to train for mean (also use assist seeds to give better training results)
# dqn_train(load_path=None, episodes=15, version=0, aggregator="mean", prob=100) # decay = 0.9998
# dqn_train(load_path="sched_net_pool_0.pt", episodes=50, version=1, aggregator="mean", prob=0, exp_rate=True)

# uncomment below 2 lines and set decay = 0.999992 and burnin to 1e4 in params to train for pool
# dqn_train(load_path=None, episodes=100, version=0, aggregator="pool", prob=85) # decay = 0.999992
# dqn_train(load_path="sched_net_pool_0.pt", episodes=100, version=1, aggregator="pool", prob=85)