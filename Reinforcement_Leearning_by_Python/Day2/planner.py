
class Planner():
    def __init__(self, env):
        self.env = env
        self.log = []
    
    def initialize(self):
        self.env.reset()
        self.log = []
    
    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")
    
    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward
    
    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]
        
        return grid

class CalueIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)
    
    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            V[s] = 0
        
        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_actions_at(s):
                    continue
                expect_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expect_rewards.append(r)
                max_reward = max(expect_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
            
            if delta < threshold:
                break
        
        V_grid = self.dict_to_grid(V)
        
        return V_grid


