import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from itertools import product

class ProductionSystem:
    def __init__(self, rules):
        self.rules = rules
        self.working_memory = set()
        self.derived = []

    def add_facts(self, facts):
        self.working_memory.update(facts)

    def infer(self):
        applied = True
        while applied:
            applied = False
            for conds, concl in self.rules:
                if conds.issubset(self.working_memory) and concl not in self.working_memory:
                    self.working_memory.add(concl)
                    self.derived.append((conds, concl))
                    applied = True
        return self.derived

    def get_diagnosis(self):
        dept_rules = [c for _, c in self.derived if isinstance(c, str) and '建议就诊科室' in c]
        if dept_rules:
            return dept_rules[-1]
        return "建议就诊科室为 全科医学"

rule_defs = [
    ({'咳嗽', '发烧'}, '疑似呼吸道感染'),
    ({'喉咙痛'}, '咽喉刺激'),
    ({'咽喉刺激', '疑似呼吸道感染'}, '疑似上呼吸道感染'),
    ({'疑似上呼吸道感染'}, '建议就诊科室为 耳鼻喉科'),
    ({'胸闷', '气短'}, '疑似肺部问题'),
    ({'疑似肺部问题', '疑似呼吸道感染'}, '疑似肺炎'),
    ({'疑似肺炎'}, '建议就诊科室为 呼吸科'),
    ({'腹痛'}, '疑似胃肠疾病'),
    ({'疑似胃肠疾病', '腹泻'}, '疑似肠炎'),
    ({'疑似肠炎'}, '建议就诊科室为 消化内科'),
    ({'心悸', '头晕'}, '疑似心律不齐'),
    ({'疑似心律不齐'}, '建议就诊科室为 心内科'),
    ({'胸痛'}, '建议就诊科室为 心内科'),
    ({'皮疹', '瘙痒'}, '疑似皮肤过敏'),
    ({'疑似皮肤过敏'}, '建议就诊科室为 皮肤科')
]

if __name__ == '__main__':
    ps = ProductionSystem(rule_defs)
    ps.add_facts({'咳嗽', '发烧', '喉咙痛', '胸闷'})
    derivations = ps.infer()
    print("推理过程:")
    for conds, concl in derivations:
        print(f"规则触发: {conds} -> {concl}")
    print("诊断结果:", ps.get_diagnosis())

    G = nx.DiGraph()
    for fact in ps.working_memory:
        G.add_node(fact)
    for conds, concl in derivations:
        for c in conds:
            G.add_edge(c, concl)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10, arrowsize=20)
    plt.title("语义网络")
    plt.show()

class BayesianNetwork:
    def __init__(self, nodes, edges, cpt):
        self.nodes = nodes
        self.edges = list(set(edges))
        self.parents = {v: [] for v in nodes}
        for p, c in self.edges:
            self.parents[c].append(p)
        self.cpt = cpt

    def prior(self, var):
        return self.cpt[var][()]

    def likelihood(self, var, evidence):
        try:
            parent_vals = tuple(evidence[p] for p in self.parents[var])
            return self.cpt[var][parent_vals]
        except KeyError:
            return 0.5

    def enumerate_all(self, vars_list, evidence):
        if not vars_list:
            return 1.0
        Y = vars_list[0]
        rest = vars_list[1:]
        if Y in evidence:
            prob = self.likelihood(Y, evidence) if evidence[Y] else 1 - self.likelihood(Y, evidence)
            return prob * self.enumerate_all(rest, evidence)
        else:
            ev_true = evidence.copy()
            ev_true[Y] = True
            ev_false = evidence.copy()
            ev_false[Y] = False
            return (self.enumerate_all(rest, ev_true) * self.likelihood(Y, ev_true)
                    + self.enumerate_all(rest, ev_false) * (1 - self.likelihood(Y, ev_false)))

    def query(self, X, evidence):
        dist_true = self.enumerate_all(self.nodes, {**evidence, X: True})
        dist_false = self.enumerate_all(self.nodes, {**evidence, X: False})
        alpha = 1.0 / (dist_true + dist_false)
        return alpha * dist_true

nodes = ['R', 'T', 'U', 'L', 'P']
edges = [('R', 'U'), ('T', 'U'), ('R', 'P'), ('L', 'P')]
cpt = {
    'R': {(): 0.1},
    'T': {(): 0.2},
    'U': {(True, True): 0.9, (True, False): 0.7, (False, True): 0.8, (False, False): 0.1},
    'L': {(): 0.05},
    'P': {(True, True): 0.85, (True, False): 0.6, (False, True): 0.7, (False, False): 0.02}
}
bn = BayesianNetwork(nodes, edges, cpt)
prob = bn.query('P', {'R': True, 'L': True})
print(f"给定呼吸道感染和肺部问题时，患肺炎的概率: {prob:.2f}")
