import random
import math
import time
import matplotlib.pyplot as plt
plt.rc("font",family='YouYuan')

# 城市坐标
cities = {
    0: (0, 0), 
    1: (1, 3), 
    2: (4, 2), 
    3: (3, 0), 
    4: (2, 4)
}
num_cities = len(cities)

# 计算两城市间的欧氏距离
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 计算路径总距离（构成闭环）
def total_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += distance(route[i], route[(i + 1) % len(route)])
    return dist

# 可视化TSP路径
def plot_route(route, title="最优路径"):
    route_coords = [cities[i] for i in route] + [cities[route[0]]]
    xs, ys = zip(*route_coords)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker='o')
    for i, (x, y) in enumerate(route_coords[:-1]):
        plt.text(x, y, f'{route[i]+1}', color='red', fontsize=12)
    plt.title(title)
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.grid(True)
    plt.show()

# ----------------------- 遗传算法实现 ---------------------------
def init_population(pop_size):
    population = []
    base_route = list(range(num_cities))
    for _ in range(pop_size):
        route = base_route.copy()
        random.shuffle(route)
        population.append(route)
    return population

def fitness(route):
    return 1 / total_distance(route)

def selection(population, fitnesses):
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    current = 0
    for route, fit in zip(population, fitnesses):
        current += fit
        if current > pick:
            return route.copy()
    return population[-1].copy()

def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end+1] = parent1[start:end+1]
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(route, mutation_rate):
    route = route.copy()
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(pop_size=100, generations=300, crossover_rate=0.9, mutation_rate=0.1):
    population = init_population(pop_size)
    best_route = None
    best_distance = float('inf')
    history_best = [] 
    history_avg = [] 

    start_time = time.time()
    
    for gen in range(generations):
        fitnesses = [fitness(route) for route in population]
        distances = [total_distance(route) for route in population]
        
        min_distance = min(distances)
        idx_best = distances.index(min_distance)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = population[idx_best].copy()
        
        avg_fitness = sum(fitnesses) / len(fitnesses)
        history_best.append(min_distance)
        history_avg.append(1 / avg_fitness)
        
        print(f"GA 第{gen+1}代: 最优路径长度 = {min_distance:.4f}, 平均路径长度 = {1/avg_fitness:.4f}")
        
        new_population = []
        while len(new_population) < pop_size:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    end_time = time.time()
    total_time = end_time - start_time
    return best_route, best_distance, total_time, history_best, history_avg

# ----------------------- 模拟退火算法实现 ---------------------------
def simulated_annealing(initial_temp=1000, cooling_rate=0.995, iterations=1000):
    current_route = list(range(num_cities))
    random.shuffle(current_route)
    current_distance = total_distance(current_route)
    best_route = current_route.copy()
    best_distance = current_distance

    temp = initial_temp
    start_time = time.time()

    for i in range(iterations):
        new_route = current_route.copy()
        a, b = random.sample(range(num_cities), 2)
        new_route[a], new_route[b] = new_route[b], new_route[a]
        new_distance = total_distance(new_route)

        delta = new_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_route = new_route
            current_distance = new_distance
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance

        temp *= cooling_rate

        if (i+1) % (iterations // 10) == 0:
            print(f"SA 迭代 {i+1}/{iterations}: 当前最优路径长度 = {best_distance:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    return best_route, best_distance, total_time

# ----------------------- 蚁群算法实现 ---------------------------
def ant_colony_optimization(num_ants=20, iterations=100, alpha=1, beta=5, evaporation_rate=0.5, Q=100):
    pheromone = [[1 for j in range(num_cities)] for i in range(num_cities)]
    heuristic = [[0 if i == j else 1/distance(i, j) for j in range(num_cities)] for i in range(num_cities)]
    best_route = None
    best_distance = float('inf')
    start_time = time.time()
    
    for it in range(iterations):
        all_routes = []
        all_distances = []
        for ant in range(num_ants):
            start = random.randint(0, num_cities-1)
            route = [start]
            unvisited = set(range(num_cities))
            unvisited.remove(start)
            while unvisited:
                current = route[-1]
                probabilities = []
                denominator = 0
                for j in unvisited:
                    denominator += (pheromone[current][j] ** alpha) * (heuristic[current][j] ** beta)
                for j in unvisited:
                    prob = (pheromone[current][j] ** alpha) * (heuristic[current][j] ** beta) / denominator
                    probabilities.append((j, prob))
                r = random.random()
                cumulative = 0
                for city, prob in probabilities:
                    cumulative += prob
                    if r <= cumulative:
                        next_city = city
                        break
                route.append(next_city)
                unvisited.remove(next_city)
            all_routes.append(route)
            all_distances.append(total_distance(route))
        
        min_dist = min(all_distances)
        idx = all_distances.index(min_dist)
        if min_dist < best_distance:
            best_distance = min_dist
            best_route = all_routes[idx].copy()
        
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1 - evaporation_rate)

        for route, dist in zip(all_routes, all_distances):
            deposit = Q / dist
            for i in range(num_cities):
                a = route[i]
                b = route[(i + 1) % num_cities]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit  
        
        if (it+1) % (iterations // 10) == 0:
            print(f"ACO 迭代 {it+1}/{iterations}: 当前最优路径长度 = {best_distance:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    return best_route, best_distance, total_time

# ----------------------- 主程序与结果对比 ---------------------------
if __name__ == '__main__':
    print("======== 遗传算法求解 TSP ========")
    ga_route, ga_distance, ga_time, ga_history_best, ga_history_avg = genetic_algorithm(
        pop_size=100,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    print("\n遗传算法最终最优路径（城市编号）：", [city+1 for city in ga_route])
    print(f"遗传算法最终路径长度：{ga_distance:.4f}, 耗时：{ga_time:.4f}秒")
    plot_route(ga_route, title="遗传算法最优路径")
    
    plt.figure(figsize=(8, 4))
    plt.plot(ga_history_best, label='最优路径长度')
    plt.plot(ga_history_avg, label='平均路径长度', linestyle='--')
    plt.xlabel("代数")
    plt.ylabel("路径长度")
    plt.title("遗传算法收敛曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n======== 模拟退火算法求解 TSP ========")
    sa_route, sa_distance, sa_time = simulated_annealing(
        initial_temp=1000,
        cooling_rate=0.995,
        iterations=1000
    )
    print("\n模拟退火算法最终最优路径（城市编号）：", [city+1 for city in sa_route])
    print(f"模拟退火算法最终路径长度：{sa_distance:.4f}, 耗时：{sa_time:.4f}秒")
    plot_route(sa_route, title="模拟退火算法最优路径")

    print("\n======== 蚁群算法求解 TSP ========")
    aco_route, aco_distance, aco_time = ant_colony_optimization(
        num_ants=20,
        iterations=100,
        alpha=1,
        beta=5,
        evaporation_rate=0.5,
        Q=100
    )
    print("\n蚁群算法最终最优路径（城市编号）：", [city+1 for city in aco_route])
    print(f"蚁群算法最终路径长度：{aco_distance:.4f}, 耗时：{aco_time:.4f}秒")
    plot_route(aco_route, title="蚁群算法最优路径")

    print("\n======== 算法对比 ========")
    print(f"遗传算法: 路径长度 = {ga_distance:.4f}, 耗时 = {ga_time:.4f}秒")
    print(f"模拟退火算法: 路径长度 = {sa_distance:.4f}, 耗时 = {sa_time:.4f}秒")
    print(f"蚁群算法: 路径长度 = {aco_distance:.4f}, 耗时 = {aco_time:.4f}秒")
