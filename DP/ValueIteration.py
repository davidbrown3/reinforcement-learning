import numpy as np

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """

    V = np.zeros(env.nS)
    Policy = np.zeros([env.nS, env.nA])
    
    while True:

        ConvergeFlag = True

        for StateIdx, StateName in enumerate(env.P.keys()):

            ActionValues = np.zeros(env.nA)
            
            StateInfo = env.P[StateName]
            
            for ActionIdx, ActionName in enumerate(StateInfo.keys()):
                
                # For now assume that all probabilities are 1
                ActionInfo = StateInfo[ActionName][0]
                
                Reward = ActionInfo[2]
                NextState = ActionInfo[1]
                NextStateValue = V[NextState]
                ActionValues[ActionIdx] = Reward + discount_factor*NextStateValue
            
            UpdatedValue = np.max(ActionValues)
            ConvergeFlag = ConvergeFlag & (np.abs(V[StateIdx]-UpdatedValue) < theta)
            V[StateIdx] = UpdatedValue

        if ConvergeFlag: break

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
            
    return Policy, V