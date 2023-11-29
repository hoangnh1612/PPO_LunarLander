import gym
import numpy as np
from PPO import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/ppo.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

for i in range(n_games):
    print(i)
    observation, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        if n_steps % N == 0:
            agent.learn()
            print(score)
            learn_iters += 1

        # Check for truncation threshold and start a new epoch if needed
        if n_steps >= 200:
            observation = observation_
            break

        observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print("avg: ", avg_score)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iters)

    # Check if truncation threshold is reached, start a new epoch if needed
    if n_steps >= 200:
        n_steps = 0
        agent.save_models()  # Save models at the end of each epoch

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
