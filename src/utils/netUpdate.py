import torch

def update(policy, optimizer, args):
	
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
		entropy_loss.append(args.p * prob * log_prob)
	optimizer.zero_grad()
	total_loss = torch.cat(policy_loss).sum() + torch.cat(value_loss).sum() + torch.cat(entropy_loss).sum()
	total_loss.backward()
	optimizer.step()
	del policy.log_prob_list[:]
	del policy.baseline_list[:]
	del policy.prob_list[:]
	del policy.rewards[:]