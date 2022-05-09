print("Importing Libraries... ")
from numpy import load
from environment_wrapper import *
from udrl import Agent, ReplayBuffer, MetricLogger
from numpy.random import randint
import time
print("------------------------------------------------------------------------------------------------------------")
print("                                                  UDRL")
print("------------------------------------------------------------------------------------------------------------")


# ------------------------------------------------------------------------------------------------------------ #
#                                                  UDRL
# ------------------------------------------------------------------------------------------------------------ #
print("!!! Entered training file !!!")

def udrl_train(load_path=None, episodes=31, version=0, aggregator="mean", prob=100, exp_rate=True, new_rate=None):
    print("Entered the main UDRL training loop.")
    env    = GraphWrapper()
    agent  = Agent(aggregator=aggregator)
    logger = MetricLogger(version=str(version), mode="train", aggregator=aggregator)
    print("Environment, Agent, and Metric Logger instantiated successfully.\n")

    if buffer is None:
        buffer = initialize_replay_buffer(replay_size, 
                                          n_warm_up_episodes, 
                                          last_few)
        print("Buffer initialized successfully.")
    
    if behavior is None:
        behavior = initialize_behavior_function(state_size, 
                                                action_size, 
                                                hidden_size, 
                                                learning_rate, 
                                                [return_scale, horizon_scale])
    print("Behavior initialized successfully.")

    
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
                action = agent.action(state, (index, 2), )
            else:
                action = agent.greedy_action(state)
 
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