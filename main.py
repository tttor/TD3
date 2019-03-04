#!/usr/bin/env python3
import numpy as np
import torch
import gym
import argparse
import os
import datetime
import utils
import OurDDPG

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Reacher-v2")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	args = parser.parse_args()

	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	env.action_space.np_random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	# Initialize policy
	policy = OurDDPG.DDPG(state_dim, action_dim, None)
	replay_buffer = utils.ReplayBuffer()

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True

	while total_timesteps < args.max_timesteps:
		if done:
			if total_timesteps != 0:
				print("Total T: {} Episode Num: {} Episode T: {} Return: {} @ {}".format(
					total_timesteps, episode_num, episode_timesteps, episode_reward, datetime.datetime.now().strftime("%H:%M:%S")))
				# policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
				policy.train(replay_buffer, 50, args.batch_size, args.discount, args.tau)

			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Select action randomly or according to policy
		action = policy.select_action(np.array(obs))
		# action = env.action_space.sample()
		action_noise = np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])
		action = (action + action_noise).clip(env.action_space.low, env.action_space.high)
		# print(action)
		# print(action_noise)
		# exit()

		# Perform action
		new_obs, reward, done, _ = env.step(action)
		done_bool = float(done) #0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward
		# print(reward)

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1
