import numpy as np
import gym

from ..utils.encode import encode
from ..utils.select import select_action
from ..utils.netUpdate import update

def target(policy_bob,\
		   optimizer_bob,\
		   args):
	
	"""
	This function implements the algorithm for solving the test task.
	"""
	
	env = gym.make(args.env)
	total_reward = 0.0

	for n_epoch in range(args.epochTest):
		time_target = 0
		env_rew = 0
		curr_state = env.reset()
		curr_state = encode(env.observation_space.n, curr_state)
		
		while True:
			time_target += 1
			action = select_action(policy_bob, np.concatenate((curr_state, np.zeros(env.observation_space.n)), axis = 0), args)
			curr_state, reward, done, info = env.step(action)
			curr_state = encode(env.observation_space.n, curr_state)
			env_rew += reward
			
			if done or time_target >= args.tmax:
				curr_state = env.reset()
				curr_state = encode(env.observation_space.n, curr_state)
				break
		total_reward += env_rew

		if (n_epoch+1)%1  == 0:
			print(n_epoch, env_rew)
			#torch.save(policy_bob.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/bob_policy_after_'+str(n_epoch)+'_target_episodes')
		
		policy_bob.rewards.append(env_rew)
		
		if len(policy_bob.rewards) == args.batch:
			update(policy_bob, optimizer_bob, args)