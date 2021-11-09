from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory





class S_E(Env):
    def __init__(self): 
        self.act_space = Discrete(3)
        self.obs_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 37 + random.randint(-3,3)
        self.length = 60 
    def step(self,act):
        self.state += act - 1  
        self.length -= 1 
        if self.state >= 37 and self.state <= 39: 
            reward = 1 
        else: 
            reward = -1 

        if self.length <= 0: 
            done = True 
        else: 
            done = False 

        self.state += random.randint(-1,1) 
        inf = {}
        return self.state,reward,done,inf 
    def render(self): 
        pass
    def reset(self): 
        self.state = 37 + random.randint(-3,3)
        self.length = 60 
        return self.state 

env = S_E() 
states = env.obs_space.shape
act = env.act_space.n 
ep = 10 


print("PRINTING ACT")
print(act)

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states,act)
print(model.summary())

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, act)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
scr = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scr.history['episode_reward']))
dqn.save_weights('dqn_weights.h5f', overwrite=True)


'''
for x in range(1,ep+1):
    state = env.reset()
    done = False
    scr = 0 
    while not done: 
        #env.render()
        act = env.act_space.sample()
        n_st,reward,done,inf = env.step(act)
        scr += reward 
    print('EP {} SCR {}'.format(ep,scr))


'''










