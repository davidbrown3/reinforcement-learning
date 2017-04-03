import numpy as np

def SampleEpsilonFunction(episode):

    return 1.0/np.max([episode,1])
