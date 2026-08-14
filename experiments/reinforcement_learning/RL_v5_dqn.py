import random
import matplotlib.pyplot as plt
import time
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
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

    # 현재 상태를 입력받아 각 행동의 Q 값을 출력하는 신경망을 생성합니다.
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.node_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    # 예측된 Q 값 중 가장 큰 값을 가진 방문 가능 도시를 선택합니다.
    def get_action(self):
        # 아직 방문하지 않아 다음 행동으로 선택 가능한 도시 목록입니다.
        cities = np.array(range(self.node_size))
        not_visited= np.array([not flag for flag in self.visited])
        left_cities = cities[not_visited]

        # 현재 상태를 모델에 입력하여 도시별 행동 점수 또는 확률을 계산합니다.
        if np.random.rand() <= epsilon:
            self.n_city = np.random.choice(left_cities)
        else:
            state = np.array([[self.c_city]])
            q_value = self.model.predict(state)
            q_value = q_value[0]

            # 이미 방문한 도시는 다음 행동 후보에서 제외합니다.
            for i in range(self.node_size):
                if self.visited[i]:
                    q_value[i] = -999999

                self.n_city = np.argmax(q_value)

        # 선택한 도시를 방문 완료 상태로 표시합니다.
        self.visited[self.n_city] = True
        self.path.append(self.n_city)
        self.distance += data.get_distance(self.c_city, self.n_city)

        # 정책에 따라 선택된 다음 도시의 인덱스를 반환합니다.
        return self.n_city

    # 계산된 학습 목표를 이용하여 정책 신경망의 가중치를 갱신합니다.
    def train_model(self):
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 현재 상태에서 모델이 예측한 모든 행동의 Q 값을 계산합니다.
        # 다음 상태에서 타깃 모델이 예측한 모든 행동의 Q 값을 계산합니다.
        for idx in range(self.node_size):

            state = np.array([[self.path[idx-1]]])
            next_state = np.array([[self.path[idx]]])
            reward = 10000 / data.get_distance(state[0][0], next_state[0][0])

            target = self.model.predict(state)

            # 보상과 다음 상태의 최대 Q 값을 이용해 벨만 학습 목표를 계산합니다.
            target[0][next_state] = reward + discount_factor * (np.amax(target[0]))
            self.model.fit(state, target, batch_size=self.node_size, epochs=1, verbose=0)

    def cal_total_distance(self):
        total_distance = 0
        for i in range(self.node_size):
            city1 = self.path[i - 1]
            city2 = self.path[i]
            total_distance += data.get_distance(city1, city2)

        return total_distance


# init
def init():
    agent.c_city = random.randrange(agent.node_size)
    agent.n_city = -1
    agent.visited = [False] * agent.node_size
    agent.visited[agent.c_city] = True
    agent.path = [agent.c_city]
    agent.distance = 0


# update
def update_best():
    global bestTourLength, bestTour
    distance = agent.cal_total_distance()
    result_data.append(distance)
    if bestTourLength == 0 or distance < bestTourLength:
        bestTourLength = distance
        bestTour = agent.path


###################
# Main Function
###################
if __name__ == "__main__":
    agent = DQNAgent()
    start_time = time.time()

    for e in range(5000):
        init()            # 새로운 에피소드를 위해 상태, 방문 기록, 경로 정보를 초기화합니다.
        # 시작 도시는 이미 방문한 것으로 표시한 상태에서 에피소드를 시작합니다.
        done = 0
        for i in range(data.node_size - 1):
            agent.n_city = agent.get_action()
            # 선택한 다음 도시를 새로운 현재 도시로 설정합니다.
            agent.c_city = agent.n_city

        agent.train_model()
        if e and e % 100 == 0:
            print(e, " done")
            print(bestTourLength)
        update_best()

    end_time = time.time()

    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()
