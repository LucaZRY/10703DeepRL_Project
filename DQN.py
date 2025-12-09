# DQN for CarRacing-v2 + Gymnasium
# - Uses TensorFlow/Keras DQN
# - Environment: gymnasium.make("CarRacing-v2", ...)
# - Collects ALL transitions and saves a single merged .npz dataset for diffusion

import random
import os
import time
import datetime
from collections import deque

import numpy as np
import cv2
from scipy import stats
from tqdm import tqdm

import gymnasium as gym  # <-- changed from gym to gymnasium

# import pyvirtualdisplay

# Tensorflow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Optional: if you still have this file
try:
    from plot_results import plotResults
    HAVE_PLOT = True
except ImportError:
    HAVE_PLOT = False

# Disable eager execution (as in your original script)
# tf.compat.v1.disable_eager_execution()

############################## SERVER CONFIGURATION ##################################
# Prevent tensorflow from allocating all GPU memory
GPUs = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUs:
    tf.config.experimental.set_memory_growth(gpu, True)

# Virtual display for headless servers (safe to keep; no effect on local)
# pyvirtualdisplay.Display(visible=0, size=(720, 480)).start()

############################## PATHS & PARAMS ##################################

USERNAME                = "jo642"
MODEL_TYPE              = "DQN2_v2"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"
REWARD_DIR              = f"./rewards/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# NEW: dataset output
DATASET_DIR             = "./data/dqn_for_diffusion"
DATASET_NAME            = "dqn_carracing_v2_dataset.npz"
os.makedirs(DATASET_DIR, exist_ok=True)

# Training params
RENDER                  = True                # show window (needs render_mode="human")
PLOT_RESULTS            = False
EPISODES                = 1000
SAVE_TRAINING_FREQUENCY = 50
SKIP_FRAMES             = 2
TARGET_UPDATE_STEPS     = 5
MAX_PENALTY             = -5
BATCH_SIZE              = 65
CONSECUTIVE_NEG_REWARD  = 30

# Testing params
PRETRAINED_PATH         = "model/jjt72/DQN2/20220423-121042/episode_60.h5"
TEST                    = False


############################## AGENT ##################################
class DQN_Agent:
    def __init__(
        self,
        action_space=None,
        memory_size=10000,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001
    ):
        # If no action_space passed, create symmetric discrete action grid (as in your original)
        if action_space is None:
            action_space = []
            for steering in [0, 0.33, 0.67, 1]:
                for accel in [0, 0.33, 0.67, 1]:
                    for braking in [0, 0.33, 0.67, 1]:
                        action_space.append((steering, accel, braking))
                        action_space.append((-steering, accel, braking))

        self.action_space = action_space
        self.D = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = self.build_model()
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        # Input: grayscale 96x96x1
        model = Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=3,
                activation="relu",
                input_shape=(96, 96, 1),
            )
        )
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=3, activation="relu"))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(len(self.action_space), activation=None))

        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7),
        )
        return model

    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        self.D.append((state, action, reward, new_state, done))

    def choose_action(self, state, best=False):
        # state: (96,96,1)
        state = np.expand_dims(state, axis=0)
        q_values = list(self.model.predict(state, verbose=0)[0])
        max_val = max(q_values)
        indices = [i for i, q in enumerate(q_values) if q == max_val]
        action_idx = random.choice(indices)

        if not best:
            if stats.bernoulli(self.epsilon).rvs():
                action_idx = random.randrange(len(self.action_space))

        return self.action_space[action_idx]

    def batch_priority(self):
        options = list(range(1, len(self.D) + 1))
        minibatch = []
        for _ in range(BATCH_SIZE):
            total = len(options) * (len(options) + 1) // 2
            prob_dist = [i / total for i in range(1, len(options) + 1)]

            choice = np.random.choice(options, 1, p=prob_dist)[0]
            del options[options.index(choice)]
            minibatch.append(self.D[choice - 1])
            minibatch.append(self.D[len(self.D) - 1])

        return minibatch

    def experience_replay(self):
        if len(self.D) < BATCH_SIZE:
            return

        minibatch = self.batch_priority()

        train_state = []
        train_target = []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            idx = self.action_space.index(action)

            if done:
                target[idx] = reward
            else:
                t = self.target_model.predict(
                    np.expand_dims(next_state, axis=0), verbose=0
                )[0]
                target[idx] = reward + self.gamma * np.amax(t)

            train_state.append(state)
            train_target.append(target)

        self.model.fit(
            np.array(train_state),
            np.array(train_target),
            epochs=1,
            verbose=0,
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name, rewards):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(REWARD_DIR, exist_ok=True)

        self.target_model.save_weights(os.path.join(MODEL_DIR, name + ".weights.h5"))
        np.savetxt(os.path.join(REWARD_DIR, name + ".csv"), rewards, delimiter=",")

        if PLOT_RESULTS:
            try:
                plotResults(os.path.join(REWARD_DIR, name + ".csv"))
            except Exception:
                pass

    def load(self, name):
        if not name.endswith(".weights.h5"):
            name = name + ".weights.h5"
        self.model.load_weights(name)
        self.target_model.set_weights(self.model.get_weights())


############################## IMAGE PROCESSING ##################################
def convert_greyscale(state):
    """
    Take input RGB state from CarRacing-v2 and convert to grayscale (96x96x1).
    Also returns a flag whether the grey road is visible (simple mask).
    """
    x, y, _ = state.shape  # 96, 96, 3
    cropped = state[0 : int(0.85 * y), 0:x]
    mask = cv2.inRange(
        cropped,
        np.array([100, 100, 100]),
        np.array([150, 150, 150]),
    )

    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    # cover scoreboard area
    gray[85:100, 0:12] = 0

    cv2.imshow("out", gray)
    cv2.waitKey(1)

    return np.expand_dims(gray, axis=2), bool(np.any(mask == 255))


############################## TRAINING + DATASET EXPORT ##################################
def train_agent(agent: DQN_Agent, env: gym.Env, episodes: int):
    """
    Train DQN on CarRacing-v2 and export a single merged .npz dataset:

    data/dqn_for_diffusion/dqn_carracing_v2_dataset.npz

    Keys:
        - states  : [N, 9216]  (flattened grayscale 96x96)
        - actions : [N, 3]
        - rewards : [N]
        - dones   : [N] (bool)
    """
    episode_rewards = []

    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []

    for episode in range(episodes):
        print("Replay buffer size:", len(agent.D))
        print(f"[INFO] Starting Episode {episode}")

        # Gymnasium reset: obs, info
        state_colour, info = env.reset()
        state_grey, can_see_road = convert_greyscale(state_colour)

        latest_rewards = []
        sum_reward = 0.0
        step = 0
        done = False
        reward_terminate = False
        repeat_neg_reward = 0

        while (
            not done
            and sum_reward > MAX_PENALTY
            and can_see_road
            and not reward_terminate
        ):
            # flatten grayscale for dataset
            flat_state = state_grey.astype(np.float32).reshape(-1)  # (9216,)

            action = agent.choose_action(state_grey)

            reward = 0.0
            for _ in range(SKIP_FRAMES + 1):
                # Gymnasium step: obs, reward, terminated, truncated, info
                new_state_colour, r, terminated, truncated, info = env.step(action)
                _done = terminated or truncated

                latest_rewards.append(r)
                reward += r
                done = _done

                if len(latest_rewards) > 500:
                    latest_rewards.pop(0)
                    if sum(latest_rewards) < 0:
                        reward_terminate = True

                if RENDER:
                    env.render()

                if done:
                    break

            # negative reward streak
            repeat_neg_reward = repeat_neg_reward + 1 if reward < 0 else 0
            if repeat_neg_reward >= CONSECUTIVE_NEG_REWARD:
                break

            new_state_grey, can_see_road = convert_greyscale(new_state_colour)
            if not can_see_road:
                reward -= 10.0

            reward = float(np.clip(reward, a_max=1.0, a_min=-10.0))

            # store for replay
            agent.store_transition(state_grey, action, reward, new_state_grey, done)

            # store for dataset
            all_states.append(flat_state)
            all_actions.append(np.array(action, dtype=np.float32))
            all_rewards.append(reward)
            all_dones.append(
                bool(done or reward_terminate or (sum_reward + reward <= MAX_PENALTY))
            )

            agent.experience_replay()

            state_grey = new_state_grey
            sum_reward += reward
            step += 1

        episode_rewards.append([sum_reward, agent.epsilon])
        print(
            f"[INFO] Episode {episode} finished | steps={step} | return={sum_reward:.2f}"
        )

        if episode % TARGET_UPDATE_STEPS == 0:
            agent.update_model()

        if episode % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episode}", rewards=episode_rewards)

    env.close()

    # merge & save dataset
    all_states_arr = np.asarray(all_states, dtype=np.float32)
    all_actions_arr = np.asarray(all_actions, dtype=np.float32)
    all_rewards_arr = np.asarray(all_rewards, dtype=np.float32)
    all_dones_arr = np.asarray(all_dones, dtype=bool)

    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    np.savez(
        dataset_path,
        states=all_states_arr,
        actions=all_actions_arr,
        rewards=all_rewards_arr,
        dones=all_dones_arr,
    )

    print(f"[DATASET] Saved merged DQN dataset to {dataset_path}")
    print(f"  states.shape  = {all_states_arr.shape}")
    print(f"  actions.shape = {all_actions_arr.shape}")
    print(f"  rewards.shape = {all_rewards_arr.shape}")
    print(f"  dones.shape   = {all_dones_arr.shape}")

    return all_states_arr, all_actions_arr, all_rewards_arr, all_dones_arr


############################## TESTING ##################################
def test_agent(agent: DQN_Agent, env: gym.Env, model: str, render=True):
    agent.load(model)
    while True:
        state_colour, info = env.reset()
        state_grey, _ = convert_greyscale(state_colour)

        sum_reward = 0.0
        t0 = time.time()
        while sum_reward > MAX_PENALTY:
            action = agent.choose_action(state_grey, best=True)
            new_state_colour, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if render:
                env.render()

            new_state_grey, _ = convert_greyscale(new_state_colour)
            state_grey = new_state_grey
            sum_reward += reward

            if done:
                break

        t1 = time.time() - t0
        print(f"[TEST] Reward={sum_reward:.2f} | Time={t1:.2f}s")


############################## MAIN ##################################
if __name__ == "__main__":
    # CarRacing-v2 with Gymnasium
    # For visible window:
    env = gym.make("CarRacing-v2", render_mode="human")

    # If you want to run headless but still get frames:
    # env = gym.make("CarRacing-v2", render_mode="rgb_array")

    if not TEST:
        agent = DQN_Agent()
        train_agent(agent, env, episodes=EPISODES)
    else:
        agent = DQN_Agent()
        test_agent(agent, env, model=PRETRAINED_PATH)
