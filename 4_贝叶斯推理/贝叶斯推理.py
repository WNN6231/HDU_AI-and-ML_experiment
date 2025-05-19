from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Smoking', 'Bronchitis'),
    ('Cold',    'Bronchitis'),
    ('Bronchitis', 'Cough'),
    ('Bronchitis', 'Asthma')
])

cpd_S = TabularCPD(
    variable='Smoking', 
    variable_card=2, 
    values=[[0.4], [0.6]], 
    state_names={'Smoking': ['No','Yes']}
)
cpd_C = TabularCPD(
    variable='Cold', 
    variable_card=2, 
    values=[[0.2], [0.8]],
    state_names={'Cold': ['No','Yes']}
)

cpd_B = TabularCPD(
    variable='Bronchitis', 
    variable_card=2,
    values=[
        [1-0.002, 1-0.011, 1-0.25, 1-0.35],
        [   0.002,    0.011,    0.25,    0.35]
    ],
    evidence=['Smoking','Cold'],
    evidence_card=[2,2],
    state_names={
        'Bronchitis': ['No','Yes'],
        'Smoking':     ['No','Yes'],
        'Cold':        ['No','Yes']
    }
)

cpd_G = TabularCPD(
    variable='Cough', 
    variable_card=2,
    values=[[1-0.15, 1-0.85], [0.15, 0.85]],
    evidence=['Bronchitis'],
    evidence_card=[2],
    state_names={'Cough': ['No','Yes'], 'Bronchitis': ['No','Yes']}
)

cpd_A = TabularCPD(
    variable='Asthma', 
    variable_card=2,
    values=[[1-0.10, 1-0.50], [0.10, 0.50]],
    evidence=['Bronchitis'],
    evidence_card=[2],
    state_names={'Asthma': ['No','Yes'], 'Bronchitis': ['No','Yes']}
)

model.add_cpds(cpd_S, cpd_C, cpd_B, cpd_G, cpd_A)
model.check_model()

infer = VariableElimination(model)
result = infer.query(
    variables=['Bronchitis'],
    evidence={'Smoking':'Yes','Cold':'Yes'}
)
print(result)
