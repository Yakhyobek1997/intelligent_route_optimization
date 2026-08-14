import random
import matplotlib.pyplot as plt
import time
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import numpy as np
import math
from data_model import Data

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


class ReinforceAgent:
    def __init__(self):
        self.node_size = data.node_size
        self.visited = []
        self.path = []
        self.distance = 0
        self.c_city = -1
        self.n_city = -1
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.node_size, activation='softmax'))
        model.summary()
        return model

    # 정책 신경망을 학습하기 위한 손실 함수와 최적화 함수를 구성합니다.
    def build_optimizer(self):
        action = K.placeholder(shape=[None, self.node_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # 선택한 행동의 확률을 이용하여 크로스 엔트로피 손실을 계산합니다.
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # 계산된 손실을 최소화하도록 정책 신경망을 갱신하는 학습 함수를 생성합니다.
        optimizer = Adam(lr=learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)
        return train

    # 정책 신경망이 출력한 확률 분포를 이용해 다음 행동을 선택합니다.
    def get_action(self):
        # 현재 상태를 모델에 입력하여 도시별 행동 점수 또는 확률을 계산합니다.
        policy = self.model.predict(np.array([[self.c_city]]))[0]

        # 이미 방문한 도시는 다음 행동 후보에서 제외합니다.
        for i in range(self.node_size):
            if self.visited[i]:
                policy[i] = 0

        # 방문 가능한 도시만 남긴 뒤 행동 확률의 합이 1이 되도록 다시 정규화합니다.
        p_sum = np.sum(policy)
        policy /= p_sum
        self.n_city = np.random.choice(range(self.node_size), 1, p=policy)[0]

        # 선택한 도시를 방문 완료 상태로 표시합니다.
        self.visited[self.n_city] = True
        self.path.append(self.n_city)
        self.distance += data.get_distance(self.c_city, self.n_city)

        # 선택한 다음 도시를 새로운 현재 도시로 설정합니다.
        self.c_city = self.n_city

    # 수집한 경로와 보상 정보를 이용하여 정책 신경망의 가중치를 갱신합니다.
    def train_model(self):
        reward = 1 / self.distance
        discounted_rewards = [0] * self.node_size
        for i in range(self.node_size):
            discounted_rewards[self.node_size - 1 - i] = (reward * math.pow(d_rate, i))
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

        next_states = [np.zeros(self.node_size) for _ in range(self.node_size)]
        for index, node in enumerate(self.path):
            next_states[index-1][node] = 1

        states = [np.array([node]) for node in self.path]
        self.optimizer([states, next_states, discounted_rewards])

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
        agent.get_action()


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
    agent = ReinforceAgent()

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
