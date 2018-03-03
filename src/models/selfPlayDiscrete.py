import numpy as np

from ..utils.encode import encode
from ..utils.select import select_action
from ..utils.netUpdate import update

def selfPlay(env,\
	policy_alice,\
	policy_bob,\
	optimizer_alice,\
	optimizer_bob,\
	initial_state,\
	args):
	
	"""
	This function block implements the algorithm mentioned in the paper
	"""
	curr_state = initial_state

	for n_step in range(args.epochSelfPlay):
		time_alice = 0
		if args.type == "reverse":
			bob_final_state = initial_state

		while True:
			time_alice += 1
			action = select_action(policy_alice, np.concatenate((initial_state, curr_state), axis = 0), args)
			if action == env.action_space.n or time_alice >= args.tmax:
				if args.type == "reset":
					bob_final_state = curr_state
					curr_state = env.reset()
					curr_state = encode(env.observation_space.n, curr_state)
				break
			curr_state, reward, done, info = env.step(action)
			curr_state = encode(env.observation_space.n, curr_state)

		time_bob = 0

		while True:			
			if np.all(curr_state == bob_final_state) or time_alice + time_bob >= args.tmax:
				env.reset()
				break

			time_bob += 1
			action = select_action(policy_bob, np.concatenate((bob_final_state, curr_state), axis = 0), args)
			curr_state, reward, done, info = env.step(action)
			curr_state = encode(env.observation_space.n, curr_state)

		policy_alice.rewards.append(args.gamma * max(0, time_bob - time_alice))
		policy_bob.rewards.append(-args.gamma * time_bob)

		if (n_step+1)%1 == 0:
			print(time_alice, time_bob)
			#torch.save(policy_alice.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/alice_policy_after_'+str(n_step)+'_episodes')
			#torch.save(policy_bob.state_dict(), '/home/ml/nanand4/McGill/Thesis/self_play/models/bob_policy_after_'+str(n_step)+'_episodes')

		if len(policy_alice.rewards) == args.batch:
			update(policy_alice, optimizer_alice, args)
		if len(policy_bob.rewards) == args.batch:
			update(policy_bob, optimizer_bob, args)

	return policy_bob