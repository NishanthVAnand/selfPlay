"""
This code implements the paper: Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play
link for the paper: https://arxiv.org/pdf/1703.05407.pdf

author of the code: Nishanth
email: nishanth127127@gmail.com
"""

import argparse

parser = argparse.ArgumentParser(description='automatic curricula via self-play')
parser.add_argument('--gamma', type=float, default=0.01, metavar='Gamma',
	help="Reward scaling factor")
parser.add_argument('--tmax', type=int, default=500, metavar='Tmax',
	help="Max time steps in self-play")
parser.add_argument('--type', default="reset", metavar='Gamma',
	help="Type of the environment| reset or reverse")
parser.add_argument('--epochSelfPlay', type=int, default=50000, metavar='SelfPlayEpochs',
	help="Number of episodes for self-play")
parser.add_argument('--epochsTest', type=int, default=, metavar='Gamma',
	help="Reward scaling factor")
parser.add_argument('--GPU', type=bool, default=False, metavar='GPU',
	help="bool value | True will use GPU")
parser.add_argument('--lambda', type=float, default=0.003, metavar='lamb',
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
