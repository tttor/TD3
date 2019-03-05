import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils, net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()
# 		self.l1 = nn.Linear(state_dim, 400)
# 		self.l2 = nn.Linear(400, 300)
# 		self.l3 = nn.Linear(300, action_dim)

# 	def forward(self, x):
# 		# x = F.relu(self.l1(x))
# 		# x = F.relu(self.l2(x))
# 		x = F.tanh(self.l1(x))
# 		x = F.tanh(self.l2(x))
# 		x = torch.tanh(self.l3(x))
# 		return x


# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim):
# 		super(Critic, self).__init__()
# 		self.l1 = nn.Linear(state_dim + action_dim, 400)
# 		self.l2 = nn.Linear(400, 300)
# 		self.l3 = nn.Linear(300, 1)

# 	def forward(self, x, u):
# 		x = F.relu(self.l1(torch.cat([x, u], 1)))
# 		x = F.relu(self.l2(x))
# 		x = self.l3(x)
# 		return x

class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		# self.actor = Actor(state_dim, action_dim, max_action).to(device)
		# self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor = net.DeterministicPolicyNetwork(state_dim, [400, 300], action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		# self.critic = Critic(state_dim, action_dim).to(device)
		# self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic = net.ActionValueNetwork(state_dim+action_dim, [400, 300]).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.actor_target = net.DeterministicPolicyNetwork(state_dim, [400, 300], action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target = net.ActionValueNetwork(state_dim+action_dim, [400, 300]).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.train_idx = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size, discount, tau):
		for it in range(iterations):
			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(d).to(device)
			reward = torch.FloatTensor(r).to(device)
			# print(reward)
			# print(next_state)

			# Compute the target Q value
			# Q_next = self.critic_target(next_state, self.actor_target(next_state))
			# target_Q = reward + (discount * Q_next * (1 - done)).detach()
			with torch.no_grad():
				Q_next = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (discount * Q_next * (1 - done))
			# print(target_Q)

			# Compute critic loss
			current_Q = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q, target_Q)
			print('cri=', critic_loss)

			# Optimize the critic
			# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			# for name, param in self.critic.named_parameters():
			# 	print(param.grad)

			self.critic_optimizer.step()
			# torch.save({'critic_state_dict': self.critic.state_dict()},
			# 	os.path.join('results', 'critic_net_'+str(it)+'.pt'))

			# Compute actor loss
			action = self.actor(state)
			actionvalue = self.critic(state, action)
			actor_loss = -actionvalue.mean()
			# print(state) # same
			# print(action) # same
			# print(actionvalue)
			print('act=', actor_loss)

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		torch.save({'actor_state_dict': self.actor.state_dict()},
				os.path.join('results', 'actor_net_train_idx_'+str(self.train_idx)+'.pt'))
		self.train_idx += 1
