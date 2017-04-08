import numpy as np

def SampleEpsilonFunction(episode,decay,_):

    return 1.0/np.max([episode*0.1,1])

def DecayEpsilonFunction(episode,decay,epsilon_initial):

    return epsilon_initial*decay**episode

def ZeroEpsilonFunction(_1,_2,_3):
    return 0