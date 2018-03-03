import torch
from torch.autograd import Variable
from torch.distributions import Categorical

def select_action(policy, state, args):
	
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

def baseline(policy, state, GPU = False):
	
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