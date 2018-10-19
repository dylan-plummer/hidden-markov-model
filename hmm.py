import numpy as np


class HiddenMarkovModel():

    def __init__(self, p_transmission, p_emission, observations=None):
        '''
        Create a Hidden Markov Model using:
            n hidden states
            m possible observations

        :param p_transmission: (n+2) x (n+2) numpy array of transmission probabilities
        :param p_emission: m x n numpy array of emission probabilities
        :param observations: 1 x m numpy array of possible emitted characters
        '''
        self.p_transmission = p_transmission
        self. p_emission = p_emission
        self.m = self.p_emission.shape[0]
        self.n = self.p_emission.shape[1]
        self.observations = observations
        