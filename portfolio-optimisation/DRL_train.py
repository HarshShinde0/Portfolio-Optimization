import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured!")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU.")

class StockEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.state_size = data.shape[1] + 3  # stock data + balance + net_worth + stock_owned
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = 0
        self.stock_owned = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.hstack((self.data.iloc[self.current_step].values, [self.balance, self.net_worth, self.stock_owned]))
        return obs

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        prev_net_worth = self.net_worth

        if action == 0:  # Buy
            if current_price <= self.balance:
                self.stock_owned += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.balance += current_price
        
        self.net_worth = self.balance + self.stock_owned * current_price
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done

class ActorCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.learning_rate = 0.001

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(32, activation='relu')(state_input)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(self.action_size, activation='softmax')(dense2)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(32, activation='relu')(state_input)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(1, activation='linear')(dense2)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def train(self, state, action, reward, next_state, done):
        target = np.zeros((1, 1))
        advantages = np.zeros((1, self.action_size))
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        advantages[0][action] = reward + (0 if done else self.gamma * next_value) - value
        target[0][0] = reward + (0 if done else self.gamma * next_value)

        self.actor.fit(state, advantages, epochs=1, verbose=0, batch_size=1)
        self.critic.fit(state, target, epochs=1, verbose=0, batch_size=1)

    def act(self, state):
        probabilities = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=probabilities)


def train_model(env, actor_critic, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        state = np.reshape(env.reset(), [1, env.state_size]).astype(np.float32)
        done = False
        total_reward = 0

        while not done:
            action = actor_critic.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size]).astype(np.float32)
            actor_critic.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward:.4f}")

    plt.plot(total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def main():
    df = pd.read_csv('new_dataset.csv')
    if 'Symbol' not in df.columns:
        raise ValueError("Dataset is missing the 'Symbol' column.")
    
    groups = df['Symbol'].unique()
    selected_group = groups[0]  # Select the first stock symbol
    data = df[df['Symbol'] == selected_group]
    
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    data.loc[:, 'SMA_20'] = data['Close'].rolling(window=20).mean()
    data.loc[:, 'SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data[required_columns + ['SMA_20', 'SMA_50']]), columns=required_columns + ['SMA_20', 'SMA_50'])

    env = StockEnv(data)
    actor_critic = ActorCritic(state_size=env.state_size, action_size=3)
    train_model(env, actor_critic)


if __name__ == '__main__':
    main()
