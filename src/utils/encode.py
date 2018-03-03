import numpy as np

def encode(observation_size, state):
	
	"""
	This function encodes the discrete state to a one-hot representation
	Arguments:
	1. state - state to be encoded (type: int)

	Return:
	1. state_one_hot - one-hot representation of the state
	"""
	state_one_hot = np.zeros(observation_size)
	state_one_hot[state] = 1
	return state_one_hot