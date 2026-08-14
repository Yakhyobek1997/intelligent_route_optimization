import random
import matplotlib.pyplot as plt
import time
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import numpy as np
from data_model import Data
import math
from collections import deque

# parameters
data = Data()
pr = 0.01            # exploit rate
d_rate = 0.90        # discount rate

# results
bestTour = 0
bestTourLength = 0
result_data = []

# keras_params
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 64
discount_factor = 0.99


class DQNAgent:
    def __init__(self):
        self.node_size = data.node_size
        self.visited = []
        self.path = []
        self.distance = 0
        self.c_city = -1
        self.n_city = -1

        # 행동 가치 예측 모델과 안정적인 학습을 위한 타깃 모델을 생성합니다.
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 최대 2,000개의 전이 샘플을 저장하는 리플레이 메모리를 구성합니다.
        self.memory = deque(maxlen=2000)

    # 현재 상태를 입력받아 각 행동의 Q 값을 출력하는 신경망을 생성합니다.
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.node_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    # 일정 주기마다 현재 모델의 가중치를 타깃 모델에 복사합니다.
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 예측된 Q 값 중 가장 큰 값을 가진 방문 가능 도시를 선택합니다.
    def get_action(self):
        # 현재 상태를 모델에 입력하여 도시별 행동 점수 또는 확률을 계산합니다.
        if np.random.rand() <= epsilon:
            return random.randrange(self.node_size)
        else:
            state = np.array([[self.c_city]])
            q_value = self.model.predict(state)
            q_value = q_value[0]

        # 이미 방문한 도시는 다음 행동 후보에서 제외합니다.
        for i in range(self.node_size):
            if self.visited[i]:
                q_value[i] = -9999

        # 선택한 도시를 방문 완료 상태로 표시합니다.
        self.visited[self.n_city] = True
        self.path.append(self.n_city)
        self.distance += data.get_distance(self.c_city, self.n_city)

        # 선택한 다음 도시를 새로운 현재 도시로 설정합니다.
        self.c_city = self.n_city

        # 방문 가능한 행동 중 가장 높은 Q 값을 반환합니다.
        return np.argmax(q_value)

    # 상태, 행동, 보상, 다음 상태 전이를 리플레이 메모리에 저장합니다.
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위 미니배치를 추출하여 모델을 학습합니다.
    def train_model(self):
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 메모리에서 지정한 배치 크기만큼 전이 샘플을 무작위 추출합니다.
        mini_batch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, 1))
        next_states = np.zeros((batch_size, 1))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에서 모델이 예측한 모든 행동의 Q 값을 계산합니다.
        # 다음 상태에서 타깃 모델이 예측한 모든 행동의 Q 값을 계산합니다.
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 보상과 다음 상태의 최대 Q 값을 이용해 벨만 학습 목표를 계산합니다.
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * (
                    np.amax(target_val[i]))

            self.model.fit(states, target, batch_size= batch_size,
                           epochs=1, verbose=0)

    def cal_total_distance(self):
        total_distance = 0
        for i in range(self.node_size):
            city1 = self.path[i - 1]
            city2 = self.path[i]
            total_distance += data.get_distance(city1, city2)
        return total_distance


####################
# Main function
####################
# init
def init():
    agent.c_city = random.randrange(agent.node_size)
    agent.n_city = -1
    agent.visited = [False] * agent.node_size
    agent.visited[agent.c_city] = True
    agent.path = [agent.c_city]
    agent.distance = 0


# move
def move():
    # 시작 도시는 이미 방문한 것으로 표시한 상태에서 에피소드를 시작합니다.
    for i in range(data.node_size - 1):
        next_node = agent.get_action()

        # 선택한 도시로 이동하여 환경 상태를 한 단계 전이합니다.
        next_state, reward, done, info = env.step(action)
        next_node = np.reshape(next_state, [1, 1])


# train
def train_model():
    agent.train_model()


# update
def update_best():
    global bestTourLength, bestTour
    distance = agent.cal_total_distance()
    result_data.append(distance)
    if bestTourLength == 0 or distance < bestTourLength:
        bestTourLength = distance
        bestTour = agent.path


###################
# start!
###################
if __name__ == "__main__":
    agent = DQNAgent()

    start_time = time.time()

    for e in range(500):
        init()            # 새로운 에피소드를 위해 상태, 방문 기록, 경로 정보를 초기화합니다.
        move()            # 모든 도시를 한 번씩 방문하는 하나의 완전한 경로를 생성합니다.
        train_model()     # 생성한 경로의 보상을 이용하여 모델을 한 단계 학습합니다.
        update_best()

    end_time = time.time()

    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()
