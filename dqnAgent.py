import random
import numpy as np
from collections import deque
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space, load_path=None, test=False):
        self.test = test
        self.action_space = action_space
        if load_path:
            self.model = load_model(load_path + "model/")
            with open(load_path + 'memory.pickle', 'rb') as handle:
                self.memory = pickle.load(handle)
            with open(load_path + 'modelMeta.pickle', 'rb') as handle:
                self.exploration_rate = pickle.load(handle)['exploration_rate']
        else:
            self.exploration_rate = EXPLORATION_MAX
            self.memory = deque(maxlen=MEMORY_SIZE)
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.test:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
        elif np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def trainCartpole(model=None):
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    if model:
        dqn_solver = DQNSolver(observation_space, action_space, load_path="./artifacts/iteration_%s/" % str(model))
        run = model
    else:
        dqn_solver = DQNSolver(observation_space, action_space)
        run = 0
    while True:
        if run % 10 == 0:
            dqn_solver.model.save("./artifacts/iteration_%s/model/" % run)
            with open("./artifacts/iteration_%s/memory.pickle" % run, 'wb') as handle:
                pickle.dump(dqn_solver.memory, handle)
            save_dict = {'exploration_rate': dqn_solver.exploration_rate}
            with open("./artifacts/iteration_%s/modelMeta.pickle" % run, 'wb') as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Save iteration %s completed" % run)

        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


def testCartpole(model, test=True, record=True):
    env = gym.make(ENV_NAME)
    if record:
        rec = VideoRecorder(env, "./video/iteration_%s.mp4" % str(model))
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, load_path="./artifacts/iteration_%s/" % str(model),
                           test=test)
    run = 0
    while True:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        run += 1
        while True:
            step += 1
            env.render()
            if record:
                rec.capture_frame()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", score: " + str(step))
                if record:
                    rec.close()
                    record = False
                break


if __name__ == "__main__":
    model = 50
    testCartpole(model=model)
    # trainCartpole(model=model)
