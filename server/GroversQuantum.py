from classiq import (
    synthesize,
    execute,
    show,
    construct_grover_model,
    RegisterUserInput,
)
from classiq.execution import ExecutionDetails
from classiq import set_constraints, Constraints

grover_model = construct_grover_model(
    definitions=[
        ("a", RegisterUserInput(size=4)),
        ("b", RegisterUserInput(size=4, is_signed=True)),
    ],
    expression="a + b == 7 and a & b == 8",
    num_reps=4,
)
grover_model = set_constraints(grover_model, Constraints(max_width=25))
qprog = synthesize(grover_model)
show(qprog)
res = execute(qprog).result()
results = res[0].value
print(results.counts_of_multiple_outputs(["a", "b"]))

def parse_result(a_str, b_str):
    a = int(a_str[::-1], 2)
    b = int(b_str[::-1], 2) - 16
    print(f"a = {a}, b = {b}, expression = {a + b == 7 and a & b == 8}")


for key in results.counts_of_multiple_outputs(["a", "b"]).keys():
    parse_result(key[0], key[1])