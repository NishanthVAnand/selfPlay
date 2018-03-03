"""
This code implements the paper: Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play
link for the paper: https://arxiv.org/pdf/1703.05407.pdf
"""

import argparse

import torch.optim as optim
import gym
import numpy as np

from src.models.selfPlayDiscrete import selfPlay
from src.models.targetDiscrete import target
from src.utils.encode import encode
from src.utils.netCreate import nnPolicy
from src.utils.init_net import initNet

parser = argparse.ArgumentParser(description='automatic curricula via self-play')
parser.add_argument('--gamma', type=float, default=0.01, metavar='Gamma',
	help="Reward scaling factor")
parser.add_argument('--tmax', type=int, default=10, metavar='Tmax',
	help="Max time steps in self-play")
parser.add_argument('--type', default="reset",
	help="Type of the environment | reset or reverse")
parser.add_argument('--epochSelfPlay', type=int, default=500, metavar='SelfPlayEpochs',
	help="Number of episodes for self-play")
parser.add_argument('--epochTest', type=int, default=500,
	help="Reward scaling factor")
parser.add_argument('--optim', default='SGD',
	help="Optimizer to be used | SGD or RMSProp")
parser.add_argument('--GPU', default=False, action='store_true',
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
parser.add_argument('--env', default='NChain-v0',
	help='gym environment to work on')
parser.add_argument('--batch', type=int, default=64,
	help='batch size for updating the network')

args = parser.parse_args()

env = gym.make(args.env)
initial_state = env.reset()

initial_state = encode(env.observation_space.n, initial_state)
policy_alice = initNet(nnPolicy(2 * env.observation_space.n, env.action_space.n + 1))
policy_bob = initNet(nnPolicy(2 * env.observation_space.n, env.action_space.n))

if args.optim == 'SGD':
	optimizer_bob = optim.SGD(policy_bob.parameters(), lr=args.lr)
	optimizer_alice = optim.SGD(policy_alice.parameters(), lr=args.lr)

else:
	optimizer_bob = optim.RMSprop(policy_bob.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
	optimizer_alice = optim.RMSprop(policy_alice.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)

if args.GPU:
	policy_alice = policy_alice.cuda()
	policy_bob = policy_bob.cuda()

if __name__ == "__main__":
	policy_bob = selfPlay(env,\
						  policy_alice,\
						  policy_bob,\
						  optimizer_alice,\
	 					  optimizer_bob,\
						  initial_state,\
						  args)

	target(policy_bob, optimizer_bob, args)