"""
This code implements the paper: Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play
link for the paper: https://arxiv.org/pdf/1703.05407.pdf

author of the code: Nishanth
email: nishanth127127@gmail.com
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np

from torch.autograd import Variable
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='automatic curricula via self-play')
parser.add_argument('--gamma', type=float, default=0.01, metavar='Gamma',
	help="Reward scaling factor")
parser.add_argument('--tmax', type=int, default=500, metavar='Tmax',
	help="Max time steps in self-play")
parser.add_argument('--type', default="reset", metavar='Gamma',
	help="Type of the environment| reset or reverse")
parser.add_argument('--epochSelfPlay', type=int, default=50000, metavar='SelfPlayEpochs',
	help="Number of episodes for self-play")
parser.add_argument('--epochsTest', type=int, default=50000, metavar='Gamma',
	help="Reward scaling factor")
parser.add_argument('--GPU', type=bool, default=False, metavar='GPU',
	help="bool value | True will use GPU")
parser.add_argument('--lamb', type=float, default=0.003, metavar='lamb',
	help="balancing parameter between reward and baseline")
parser.add_argument('--p', type=float, default=0.003, metavar='entropy',
	help="entropy regularization parameter")
parser.add_argument('--lr', type=float, default=0.003, metavar='learningRate',
	help="Learning rate for the optimizer")
parser.add_argument('--alpha', type=float, default=0.97,
	help='alpha parameter for RMS Prop optimizer')
parser.add_argument('--eps', type=float, default=1e-6,
	help='epsilon parameter for RMS Prop')
parser.add_argument('--env', default='MountainCar-v0',
	help='gym environment to work on')

args = parser.parse_args()

class nnPolicy(nn.Module):
	
	"""
	This class implements the policy gradient method
	Arguments it takes are below
	state_size: size of the state
	action_size: number of actions in the environment
	"""
	
	def __init__(self, state_size, action_size):
		super(nnPolicy, self).__init__()
		self.first = nn.Linear(state_size, 50)
		self.second = nn.Linear(50, 50)
		self.action_layer = nn.Linear(50, action_size)
		self.baseline_layer = nn.Linear(50, 1)

		self.log_prob_list = []
		self.baseline_list = []
		self.prob_list = []
		self.rewards =[] 

	def forward(self,x):
		x = F.relu(self.first(x))
		x = F.relu(self.second(x))
		action = self.action_layer(x)
		baseline = self.baseline_layer(x)
		return F.softmax(action, dim=1), baseline

env = gym.make(args.env)
initial_state = env.reset()

policy_alice = nnPolicy(2 * env.observation_space.shape[0], env.action_space.n + 1)
policy_bob = nnPolicy(2 * env.observation_space.shape[0], env.action_space.n)
optimizer_alice = optim.RMSprop(policy_alice.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
optimizer_bob = optim.RMSprop(policy_bob.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
if args.GPU:
	policy_alice = policy_alice.cuda()
	policy_bob = policy_bob.cuda()

def select_action(policy, state):
	
	"""
	This function finds the action to be taken on a state as per the current policy
	Arguments:
	1. Policy - Policy gradient network
	2. State - state at which the action has to be found

	Return:
	1. action value - action to be taken at the state
	"""

	state = torch.from_numpy(state).float().unsqueeze(0)
	if args.GPU:
		state = state.cuda()
	probs, baseline_val = policy.forward(Variable(state))
	policy.prob_list.append(probs)
	policy.baseline_list.append(baseline_val)
	m = Categorical(probs)
	action = m.sample()
	policy.log_prob_list.append(m.log_prob(action))
	return action.data[0]

def baseline(policy, state):
	
	"""
	This function returns the baseline value for the state
	Arguments:
	1. Policy - Policy gradient network
	2. State - state at which the action has to be found

	Return:
	1. baseline value - baseline value of the queried state
	"""

	state = torch.from_numpy(state).float().unsqueeze(0)
	if args.GPU:
		state = state.cuda()
	value = policy.forward(Variable(state))[1]
	return value.data[0]

def update(policy, optimizer):
	
	"""
	This function updates the network passed
	Arguments:
	1. Network to be updated
	2. Optimizer used to update the policy
	"""

	policy_loss = []
	value_loss = []
	entropy_loss = []
	for prob, log_prob, base, reward in zip(policy.prob_list, policy.log_prob_list, policy.baseline_list, policy.rewards):
		rew = reward - base.data[0,0]
		policy_loss.append(-log_prob * rew)
		value_loss.append(args.lamb * (reward - base) ** 2)
		entropy_loss.append(-args.p * prob * log_prob)
	optimizer.zero_grad()
	total_loss = torch.cat(policy_loss).sum() + torch.cat(value_loss).sum() + torch.cat(entropy_loss).sum()
	total_loss.backward()
	optimizer.step()
	del policy.log_prob_list[:]
	del policy.baseline_list[:]
	del policy.prob_list[:]
	del policy.rewards[:]


def selfPlay():
	
	"""
	This function block implements the algorithm mentioned in the paper
	"""
	
	curr_state = initial_state

	for n_step in range(args.epochSelfPlay):
		time_alice = 0
		if args.type == "reverse":
			bob_final_state = initial_state

		while True:
			#env.render()
			time_alice += 1
			action = select_action(policy_alice, np.concatenate((initial_state, curr_state), axis = 0))
			if action == env.action_space.n or time_alice >= args.tmax:
				if args.type == "reset":
					bob_final_state = curr_state
					curr_state = env.reset()
				break
			curr_state, reward, done, info = env.step(action)

		time_bob = 0
		while True:
			#env.render()
			time_bob += 1
			action = select_action(policy_bob, np.concatenate((bob_final_state, curr_state), axis = 0))
			if np.abs(curr_state - bob_final_state).sum() < 0.05 or time_alice + time_bob >= args.tmax:
				env.reset()
				break
			curr_state, reward, done, info = env.step(action)

		policy_alice.rewards.append(args.gamma * max(0, time_bob - time_alice))
		policy_bob.rewards.append(-args.gamma * time_bob)

		if (n_step+1)%500 == 0:
			print(time_alice, time_bob)
			#torch.save(policy_alice.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/alice_policy_after_'+str(n_step)+'_episodes')
			#torch.save(policy_bob.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/bob_policy_after_'+str(n_step)+'_episodes')

		if (n_step+1) % 50 == 0:
			update(policy_alice, optimizer_alice)
			update(policy_bob, optimizer_bob)

def target():
	
	"""
	This function implements the algorithm for solving the test task.
	"""
	
	env = gym.make(args.env)
	for n_epoch in range(args.epochsTest):
		time_target = 0
		env_rew = 0
		curr_state = env.reset()
		
		while True:
			#env.render()
			time_target += 1
			action = select_action(policy_bob, np.concatenate((curr_state, np.zeros(env.observation_space.shape[0])), axis = 0))
			curr_state, reward, done, info = env.step(action)
			if done or time_target >= args.tmax:
				curr_state = env.reset()
				break
			env_rew += reward
		
		if (n_epoch+1)%500 == 0:
			print(n_epoch, env_rew)
			#torch.save(policy_bob.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/bob_policy_after_'+str(n_epoch)+'_target_episodes')
		
		policy_bob.rewards.append(env_rew)
		
		if (n_epoch+1) % 50 == 0:
			update(policy_bob, optimizer_bob)

if __name__ == "__main__":
	selfPlay()
	target()