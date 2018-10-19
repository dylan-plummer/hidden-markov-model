import numpy as np


class HiddenMarkovModel():

    def __init__(self, p_transmission, p_emission, states, observations, p_start):
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
        self.state_names = states
        self.states = np.arange(0, len(states))
        self.characters = observations
        self.observations = np.arange(0, len(observations))
        self.p_start = p_start

    @staticmethod
    def dptable(V, sequence):
        # Print a table of steps from dictionary
        yield '\t  ' + '       '.join('%.7s' % ('%.7s' % sequence[i] + '') for i in range(len(V)))
        for state in V[0]:
            yield '%.7s: ' % state + ' '.join('%.7s' % ('%f' % v[state]['prob']) for v in V)

    def viterbi(self, sequence_obs):
        V = [{}]

        index_sequence = np.uint8(np.zeros(len(sequence_obs)))
        for i in range(len(sequence_obs)):
            index_sequence[i] = self.characters.index(sequence_obs[i])

        for s in self.states:
            V[0][s] = {'prob': self.p_start[s] * self.p_emission[s][index_sequence[0]],
                       'prev': None}

        for t in range(1, len(index_sequence)):
            V.append({})
            for s in self.states:
                p_trans_max = max(V[t-1][prev_s]['prob'] * self.p_transmission[prev_s][s] for prev_s in self.states)
                for prev_s in self.states:
                    if V[t-1][prev_s]['prob'] * self.p_transmission[prev_s][s] == p_trans_max:
                        p_max = p_trans_max * self.p_emission[s][index_sequence[t]]
                        V[t][s] = {'prob': p_max, 'prev': prev_s}
                        break

        for line in self.dptable(V, sequence_obs):
            print(line)

        opt = []
        # the highest probability
        p_max = max(value['prob'] for value in V[-1].values())
        prev = None
        # get most probable state
        for s, data in V[-1].items():
            if data['prob'] == p_max:
                opt.append(s)
                prev = s
                break
        # and backtrack until first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][prev]['prev'])
            prev = V[t + 1][prev]['prev']
        state_path = ' -> '.join(self.state_names[i] for i in opt)
        print('The steps of states are', state_path, 'with highest probability of', p_max)

if __name__ == '__main__':
    emission = np.array([[0.5, 0.5], [0.75, 0.25]])
    transmission = np.array([[0.9, 0.1], [0.1, 0.9]])
    start = np.array([0.5, 0.5])

    hmm = HiddenMarkovModel(transmission, emission, ['F', 'B'], ['H', 'T'], start)
    sequence = 'HHHHHTTTTT'
    hmm.viterbi(list(sequence))