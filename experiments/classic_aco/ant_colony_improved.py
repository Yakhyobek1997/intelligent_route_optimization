import random as rand
import matplotlib.pyplot as plt
import math
import time

graph = []
c = 1.0             # original amount of trail
alpha = 1           # trail preference
beta = 2            # greedy preference
global_evap = 0.05  # 전체 경로에 적용되는 전역 페로몬 증발 비율입니다.
local_evap = 0.02   # 개별 이동 직후 해당 경로에 적용되는 지역 페로몬 증발 비율입니다.
Q = 500             # 좋은 경로에 새로 추가할 페로몬의 기준 강도입니다.

pr = 0.01           # 다음 도시를 무작위로 선택하여 탐색 다양성을 확보하는 확률입니다.

n = 0               # 입력 데이터에 포함된 전체 도시(노드)의 개수입니다.
m = 0               # 한 반복에서 경로를 탐색하는 개미 에이전트의 수입니다.
numAntFactor = 0.8  # 도시 수에 비례하여 개미 에이전트 수를 결정하는 계수입니다.
trails = []         # 도시 간 경로의 페로몬 강도를 저장하는 행렬입니다.
ants = []
currentIndex = 0

bestTour = 0
bestTourLength = 0
d_rate = 0.9        # discount rate


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
    global n, m, trails, ants, local_evap
    with open("_data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(",")
            f_arr = [float(val) for val in arr]
            graph.append(f_arr)
        f.close()

    n = len(graph)
    m = int(n * numAntFactor)
    trails = [[c] * n for _ in range(n)]
    ants = [Ant() for _ in range(m)]
    local_evap = math.pow(global_evap, 1.0 / n)


def get_distance(node1, node2):
    global graph
    x1 = graph[node1][0]
    y1 = graph[node1][1]
    x2 = graph[node2][0]
    y2 = graph[node2][1]
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def nearest_neighbor(node):
    town = -1
    min_value = get_distance(node, 0)
    for i in range(1, n):
        value = get_distance(node, i)
        if i != node and value < min_value:
            town = i
    return town


# m ants with random start city
def solve():
    global n, trails, c, bestTourLength, bestTour
    result_data = []
    start_time = time.time()
    for i in range(200):
        setup_ants()
        move_ants()
        update_global_trails()
        update_best()
        result_data.append(bestTourLength)

    end_time = time.time()
    print("Time : " + str(end_time - start_time))
    print("Best tour length : " + str(bestTourLength))
    print(bestTour)
    plt.plot(result_data)
    plt.show()


############################
### main_method
############################
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
            selected_town = select_next_town(ant)                   # 방문하지 않은 후보 중에서 다음에 이동할 도시를 선택합니다.
            update_local_trails(ant.tour[-1], selected_town)        # 선택한 경로를 이동한 직후 지역 페로몬 값을 갱신합니다.
            ant.visit_town(selected_town)                           # 선택한 도시를 방문 처리하고 현재 경로에 추가합니다.


def update_best():
    global bestTour, bestTourLength
    for ant in ants:
        if bestTourLength == 0 or ant.tour_length() < bestTourLength:
            bestTour = ant.tour
            bestTourLength = ant.tour_length()


#######################
### sub_method
#######################

# 현재 도시에서 이동 가능한 다음 도시를 탐색합니다.
def select_next_town(ant):
    if rand.random() < pr:
        # exploit
        while True:
            t = rand.randrange(n)  # random town
            if not ant.visited[t]:
                return t

    # 방문 가능한 각 경로의 선택 확률을 계산합니다.
    probs = prob_to(ant)
    r = rand.random()
    tot = 0
    for i in range(n):
        tot += probs[i]
        if tot >= r:
            return i


# 페로몬과 거리 정보를 결합하여 후보 경로별 확률을 계산합니다.
def prob_to(ant):
    town = ant.tour[-1]
    total = 0.0
    probs = []

    # 아직 방문하지 않은 모든 후보 경로의 가중치 합계를 계산합니다.
    for i in range(n):
        if not ant.visited[i]:
            total += pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)

    # 각 후보 경로의 상대적 선호도를 확률로 정규화합니다.
    for i in range(n):
        if ant.visited[i]:
            probs.append(0.0)
        else:
            prob = pow(trails[town][i], alpha) * pow(1.0 / get_distance(town, i), beta)
            try:
                probs.append(prob / total)
            except():
                print("방문할 수 있는 곳이 없음")

    return probs


# 개별 이동 결과를 반영하여 지역 페로몬을 강화하는 단계입니다.
def update_local_trails(town1, town2):
    # 현재 이동에서 얻은 보상 값을 이용해 지역 페로몬을 갱신합니다.
    # best = trails[town2][0]
    # for i in range(1, n):
    #     if best < trails[town2][i]:
    #         best = trails[town2][i]
    #
    # trails[town1][town2] += d_rate * best

    # 기본 Ant Colony 방식으로 선택된 경로의 지역 페로몬을 조정합니다.
    nearest_town = nearest_neighbor(town1)
    trails[town1][town2] = (1 - local_evap) * trails[town1][town2] + local_evap * (1 / (n * get_distance(town1, nearest_town)))


# 한 반복이 끝난 뒤 전체 경로의 전역 페로몬을 갱신합니다.
def update_global_trails():
    global n, global_evap, ants, trails

    # 기존 페로몬을 감소시켜 오래된 경로 정보의 영향력을 줄입니다.
    for i in range(n):
        for j in range(n):
            trails[i][j] *= global_evap

    # 가장 우수한 경로를 기준으로 전역 페로몬을 추가합니다.
    best_length = 0
    best_ant = -1
    for ant in ants:
        if best_length == 0 or ant.tour_length() < best_length:  # 현재 반복에서 가장 짧은 경로를 찾은 에이전트인지 확인합니다.
            best_length = ant.tour_length()
            best_ant = ant

    for i in range(-1, n - 1):
        trails[best_ant.tour[i]][best_ant.tour[i + 1]] += 1 / best_length


if __name__ == "__main__":
    data_init()
    solve()