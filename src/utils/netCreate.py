import torch
import torch.nn as nn
import torch.nn.functional as F

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