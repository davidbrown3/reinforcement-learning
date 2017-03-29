import numpy as np

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    
    while True:
        
        # Initialising a boolean flag as to whether the value function has converged
        ConvergeFlag = True
        
        for StateIdx, StateName in enumerate(env.P.keys()):
            
            ActionProbability = policy[StateIdx,:]
            ActionValues = np.zeros(env.nA)
            
            StateInfo = env.P[StateName]
            
            for ActionIdx, ActionName in enumerate(StateInfo.keys()):
                
                # For now assume that all probabilities are 1
                ActionInfo = StateInfo[ActionName][0]
                
                Reward = ActionInfo[2]
                NextState = ActionInfo[1]
                NextStateValue = V[NextState]
                ActionValues[ActionIdx] = Reward + discount_factor*NextStateValue
                
            # Update value @ k+1 for that state
            UpdatedValue = np.sum(ActionProbability*ActionValues)
            ConvergeFlag = ConvergeFlag & (np.abs(V[StateIdx]-UpdatedValue) < theta)
            V[StateIdx] = UpdatedValue

        if ConvergeFlag: break

    return np.array(V)
