import random as rand
import matplotlib.pyplot as plt
import math
import time
import tensorflow as tf

c = 1.           # original amount of trail
alpha = 1          # trail preference
beta = 1           # greedy preference

pr = 0.01           # probability of pure random selection of the next town

n = 0               # 입력 데이터에 포함된 전체 도시(노드)의 개수입니다.
m = 0               # 한 반복에서 경로를 탐색하는 개미 에이전트의 수입니다.
numAntFactor = 0.8  # 도시 수에 비례하여 개미 에이전트 수를 결정하는 계수입니다.
graph = []
trails = []         # 도시 간 경로의 페로몬 강도를 저장하는 행렬입니다.
ants = []
currentIndex = 0

bestTour = 0
bestTourLength = 0

X = 0
W = 0
sess = 0
hypothesis = 0
step = 0
RMSE = 0


class Ant:
    def __init__(self):
        self.tour = []        # 현재 에이전트가 순서대로 방문한 도시 인덱스를 저장합니다.
        self.visited = [False] * n     # 각 도시의 방문 여부를 기록하여 중복 방문을 방지합니다.

    def visit_town(self, town):
        self.tour.append(town)
        self.visited[town] = True

    def tour_length(self):
        length = 0
        for i in range(-1, n - 1):
            length += get_distance(self.tour[i], self.tour[i + 1])

        return length


def data_init():
    global n, m, trails, ants, c, X, W, sess, step, RMSE, hypothesis
    with open("_data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    m = int(n * numAntFactor)
    c = get_distance(0, 1)
    trails = [[c] * n for _ in range(n)]
    ants = [Ant() for _ in range(m)]


    ###############
    # tensorflow
    ###############
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[1, n])
    W = tf.Variable(tf.random_normal([n, 1]), name='weight')

    # Forward prop
    hypothesis = tf.matmul(X, W)

    # diff
    diff = hypothesis

    # Back prop (chain rule)
    d_l1 = diff
    d_w = tf.matmul(tf.transpose(X), d_l1)

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis))

    # Updating network using gradients
    learning_rate = 1e-6
    step = [
        tf.assign(W, W - learning_rate * d_w),
    ]

    # 7. Running and testing the training process
    RMSE = tf.reduce_mean(tf.square((hypothesis)))

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)


def get_distance(node1, node2):
    global graph
    x1 = graph[node1][0]
    y1 = graph[node1][1]
    x2 = graph[node2][0]
    y2 = graph[node2][1]
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def solve():
    global n, trails, c, bestTourLength, bestTour

    result_data = []
    start_time = time.time()
    for i in range(50):
        setup_ants()
        move_ants()
        update_trails()
        update_best()
        result_data.append(bestTourLength)

    sess.close()
    end_time = time.time()
    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    plt.plot(result_data)
    plt.show()


# m ants with random start city
def setup_ants():
    global currentIndex, m, ants
    for i in range(m):
        for j in range(len(ants[i].visited)):
            ants[i].visited[j] = False
        ants[i].tour = []
        ants[i].visit_town(rand.randrange(n))


# m ants with random start city
def move_ants():
    global n, ants
    for _ in range(n - 1):
        for ant in ants:
            selected_town = select_next_town(ant)
            try:
                ant.visit_town(selected_town)
            except:
                print("selected_town has none type value")


def select_next_town(ant):
    global n, pr
    if rand.random() < pr:
        # 일정 확률로 후보 도시를 무작위 선택하여 새로운 경로를 탐색합니다.
        while True:
            t = rand.randrange(n)  # random town
            if not ant.visited[t]:
                return t

    else:
        # 방문 가능한 각 경로의 선택 확률을 계산합니다.
        probs = prob_to(ant)

        r = rand.random()
        tot = 0
        for i in range(n):
            tot += probs[i]
            if tot >= r:
                return i


def prob_to(ant):
    global n, alpha, beta
    town = ant.tour[-1]
    total = 0.0
    probs = []
    # 아직 방문하지 않은 모든 후보 경로의 가중치 합계를 계산합니다.
    for i in range(n):
        if not ant.visited[i]:
            total += pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)

    for i in range(n):
        if ant.visited[i]:
            probs.append(0.0)
        else:
            # 각 후보 경로의 상대적 선호도를 확률로 정규화합니다.
            prob = pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)
            try:
                probs.append(prob / total)
            except:
                print("방문할 수 있는 곳이 없음")

    return probs


def update_trails():
    global trails

    for ant in ants:
        # 대안으로 방문 순서 대신 각 이동 구간의 거리 목록을 모델 입력으로 사용할 수 있습니다.
        x_data = [ant.tour]                                                             # 모델 입력으로 에이전트가 방문한 도시 순서를 사용합니다.
        w_data = [[trails[ant.tour[i - 1]][ant.tour[i]]] for i in range(n)]
        W.assign(w_data)

        _, _, w_val = sess.run([step, RMSE, W], feed_dict={X: x_data})

        # 학습 결과로 계산된 값을 해당 경로의 페로몬에 반영합니다.
        for i in range(n):
            if not math.isnan(float(w_val[i][0])):
                trails[ant.tour[i - 1]][ant.tour[i]] = float(w_val[i][0])


def update_best():
    global bestTour, bestTourLength
    for ant in ants:
        if bestTourLength == 0 or ant.tour_length() < bestTourLength:
            bestTourLength = ant.tour_length()

if __name__ == "__main__":
    data_init()
    solve()