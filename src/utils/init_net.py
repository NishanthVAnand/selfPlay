import torch.nn as nn 

def initNet(policy):
	"""
	This function initializes the network weights

	Arguments:
	1. Policy network

	Return:
	1. Updated Policy network
	"""

	for m in policy.parameters():
	    nn.init.normal(m, mean=0, std=0.2)

	return policy