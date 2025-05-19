import heapq
import time

# 辅助函数：找到状态中某个数字的位置（行和列）
def find_position(state, num):
    for i in range(9):
        if state[i] == str(num):
            row = i // 3
            col = i % 3
            return row, col
    return None

# 辅助函数：生成所有可能的后继状态（空白块移动）
def get_successors(state):
    successors = []
    row, col = find_position(state, 0)  # 找到空白块（0）的位置
    i = row * 3 + col  # 计算空白块的索引
    # 上移
    if row > 0:
        new_state = list(state)
        new_state[i], new_state[i-3] = new_state[i-3], new_state[i]
        successors.append(''.join(new_state))
    # 下移
    if row < 2:
        new_state = list(state)
        new_state[i], new_state[i+3] = new_state[i+3], new_state[i]
        successors.append(''.join(new_state))
    # 左移
    if col > 0:
        new_state = list(state)
        new_state[i], new_state[i-1] = new_state[i-1], new_state[i]
        successors.append(''.join(new_state))
    # 右移
    if col < 2:
        new_state = list(state)
        new_state[i], new_state[i+1] = new_state[i+1], new_state[i]
        successors.append(''.join(new_state))
    return successors

# 启发函数：计算曼哈顿距离之和
def manhattan_distance(state, goal):
    distance = 0
    goal_pos = {}
    # 构建目标状态中每个数字的位置字典
    for i in range(9):
        num = goal[i]
        if num != '0':
            row = i // 3
            col = i % 3
            goal_pos[num] = (row, col)
    # 计算每个数字的曼哈顿距离
    for i in range(9):
        num = state[i]
        if num != '0':
            current_row = i // 3
            current_col = i % 3
            goal_row, goal_col = goal_pos[num]
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

# 辅助函数：以 3x3 网格形式打印状态
def print_state(state):
    for i in range(3):
        print(state[i*3:i*3+3])
    print()

# 节点类：用于 A* 算法的优先队列
class Node:
    def __init__(self, state, g, h, parent=None):
        self.state = state  # 当前状态
        self.g = g  # 从起点到当前状态的实际代价（深度）
        self.h = h  # 从当前状态到目标状态的估计代价（曼哈顿距离）
        self.f = g + h  # 总代价估计
        self.parent = parent  # 父节点，用于回溯路径

    # 定义小于比较，用于堆排序
    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h  # 如果 f 相等，h 小的优先
        return self.f < other.f

# A* 算法实现
def astar(start, goal):
    open_list = []  # 优先队列，存储待扩展节点
    open_set = set()  # 记录 open 列表中的状态
    close_set = set()  # 记录已扩展的状态
    start_node = Node(start, 0, manhattan_distance(start, goal))
    heapq.heappush(open_list, start_node)
    open_set.add(start)

    while open_list:
        # 打印当前的 open 和 close 列表
        print("\n扩展节点前的 Open 列表:")
        for node in open_list:
            print(f"状态: {node.state}, f: {node.f}, g: {node.g}, h: {node.h}")
        print("Close 列表:", sorted(list(close_set)))

        current = heapq.heappop(open_list)  # 取出 f 最小的节点
        open_set.remove(current.state)

        if current.state == goal:  # 达到目标状态
            path = []
            while current:
                path.append(current.state)
                current = current.parent
            path.reverse()
            return path, len(close_set)  # 返回路径和扩展节点数

        close_set.add(current.state)
        successors = get_successors(current.state)

        for successor_state in successors:
            if successor_state in close_set:  # 已扩展过，跳过
                continue
            if successor_state not in open_set:  # 未在 open 中，加入
                h = misplaced_tiles(successor_state, goal)
                successor_node = Node(successor_state, current.g + 1, h, current)
                heapq.heappush(open_list, successor_node)
                open_set.add(successor_state)

    return None, len(close_set)  # 无解，返回扩展节点数

# 启发函数：计算不在位棋子的数量
def misplaced_tiles(state, goal):
    return sum(1 for i in range(9) if state[i] != goal[i] and state[i] != '0')

# 主函数
if __name__ == "__main__":
    start = "283104765"  # 初始状态
    goal = "123804765"  # 目标状态

    print("初始状态:")
    print_state(start)
    print("目标状态:")
    print_state(goal)

    start_time = time.time()
    path, nodes_expanded = astar(start, goal)
    end_time = time.time()

    # 输出结果
    if path:
        print("找到解决方案:")
        for state in path:
            print_state(state)
        print("步数:", len(path) - 1)
    else:
        print("无解")
    print("搜索时间:", end_time - start_time, "秒")
    print("扩展节点数:", nodes_expanded)