import numpy as np
from PolicyEvaluation import policy_eval

def policy_improvement(env, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    Policy = np.ones([env.nS, env.nA]) / env.nA
    V = policy_eval(Policy, env, theta=0.01)

    while True:

        for StateIdx, StateName in enumerate(env.P.keys()): 
            StateInfo = env.P[StateName]
            ActionValues = np.zeros(env.nA)

            for ActionIdx, ActionName in enumerate(StateInfo.keys()):
                
                # For now assume that all probabilities are 1
                ActionInfo = StateInfo[ActionName][0]
                
                Reward = ActionInfo[2]
                NextState = ActionInfo[1]
                NextStateValue = V[NextState]
                ActionValues[ActionIdx] = Reward + discount_factor*NextStateValue
            
            MaxValueIdx = np.argmax(ActionValues)
            Policy[StateIdx,:] = 0
            Policy[StateIdx,MaxValueIdx] = 1
        
        VNew = policy_eval(Policy, env, theta=0.01)
        if np.all(VNew==V):
            V = VNew
            break
        else:
            V = VNew
    
    return Policy, V
