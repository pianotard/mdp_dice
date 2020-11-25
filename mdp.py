DEBUG = False

class Transition:
    
    def __init__(self, states, actions):
        self._states = states
        self._actions = actions
        self._probs = {}
        
    def add_prob(self, state, action, r_state, prob):
        assert state in self._states, f'Invalid state provided: {state}'
        assert action in self._actions, f'Invalid action provided: {action}'
        assert r_state in self._states, f'Invalid state provided: {r_state}'
        assert prob >= 0 and prob <= 1, f'Invalid prob provided: {prob}'
        if state not in self._probs.keys():
            self._probs[state] = {action: {} for action in self._actions}
        if r_state not in self._probs[state][action].keys():
            self._probs[state][action] = {}
        self._probs[state][action][r_state] = prob
            
    def get_prob(self, state, action, r_state):
        return self._probs[state][action][r_state]
    
    def get_from_states(self):
        return self._probs.keys()
    
    def get_result_states(self, state, action):
        return self._probs[state][action].keys()
        
class MDP:
    
    def __init__(self, states, actions, trans_probs, reward_func, gamma):
        """
        :param states: list of possible states
        :param actions: list of possible actions
        :param trans_probs: list of list of [state, action, result_state, prob]
        :param reward_func: function that takes in (state, action, result_state) and returns a number
        """
        self._states = states
        self._actions = actions
        self._trans_probs = Transition(states, actions)
        for trans_prob in trans_probs:
            self._trans_probs.add_prob(trans_prob[0], trans_prob[1], trans_prob[2], trans_prob[3])
        self._reward_func = reward_func
        self._q_stars = self.compute_q_stars(gamma)
        self._v_stars = self.compute_v_stars()
        self._pi_stars = self.compute_pi_stars()
            
    def compute_q_stars(self, gamma, thres = 10 ** -4):
        q_stars = {state: {action: 0 for action in self._actions} for state in self._states}
        while True:
            q_stars_copy = {state: {action: q_stars[state][action] for action in self._actions} for state in self._states}
            q_diffs = []
            for state in self._trans_probs.get_from_states():
                if DEBUG: print('.', end = '')
                for action in self._actions:
                    q_star = 0
                    for r_state in self._trans_probs.get_result_states(state, action):
                        trans_prob = self._trans_probs.get_prob(state, action, r_state)
                        reward = self._reward_func(state, action, r_state)
                        max_q_star_next = q_stars_copy[r_state][max(q_stars_copy[r_state], key = q_stars_copy[r_state].get)]
                        q_star += trans_prob * (reward + gamma * max_q_star_next)
                    q_stars_copy[state][action] = q_star
                    q_diffs.append(abs(q_star - q_stars[state][action]))
            q_stars = q_stars_copy
            if all(diff < thres for diff in q_diffs):
                if DEBUG: print('Threshold attained for q stars, exiting..')
                return q_stars
    
    def compute_v_stars(self):
        q_stars = self._q_stars
        return {state: q_stars[state][max(q_stars[state], key = q_stars[state].get)] for state in self._states}
    
    def compute_pi_stars(self):
        q_stars = self._q_stars
        return {state: max(q_stars[state], key = q_stars[state].get) for state in self._states}
    
    def compute_v_stars_naive(self, gamma, thres = 10 ** -4):
        v_stars = {state: 0 for state in self._states}
        while True:
            v_stars_copy = {state: v_stars[state] for state in self._states}
            v_diffs = []
            for state in self._states:
                action_values = []
                for action in self._actions:
                    v_star = 0
                    for r_state in self._states:
                        trans_prob = self._trans_probs.get_prob(state, action, r_state)
                        reward = self._reward_func(state, action, r_state)
                        v_star_next = v_stars_copy[r_state]
                        v_star += trans_prob * (reward + gamma * v_star_next)
                    action_values.append(v_star)
                v_stars_copy[state] = max(action_values)
                v_diffs.append(abs(v_stars[state] - v_stars_copy[state]))
            v_stars = v_stars_copy
            if all(diff < thres for diff in v_diffs):
                print('Threshold attained for v stars, exiting..')
                return v_stars
        
    @property
    def q_stars(self):
        return self._q_stars
    
    @property
    def v_stars(self):
        return self._v_stars
    
    @property
    def pi_stars(self):
        return self._pi_stars