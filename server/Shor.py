import sys
import cirq
import fractions
import math
import random
import numpy as np
import sympy
from typing import Callable, Iterable, List, Optional, Sequence, Union

def multiplicative_group(n: int) -> List[int]:
    
    assert n > 1
    group = [1]
    for x in range(2, n):
        if math.gcd(x, n) == 1:
            group.append(x)
    return group

# n = 15
# print(f"The multiplicative group modulo n = {n} is:")
# print(multiplicative_group(n))


def classical_order_finder(x: int, n: int) -> Optional[int]:
    
    # Make sure x is both valid and in Z_n.
    if x < 2 or x >= n or math.gcd(x, n) > 1:
        raise ValueError(f"Invalid x={x} for modulus n={n}.")

    # Determine the order.
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r

n = 15  # The multiplicative group is [1, 2, 4, 7, 8, 11, 13, 14].
x = 8
r = classical_order_finder(x, n)

# Check that the order is indeed correct.
# print(f"x^r mod n = {x}^{r} mod {n} = {x**r % n}")

"""Example of defining an arithmetic (quantum) gate in Cirq."""
class Adder(cirq.ArithmeticGate):
    """Quantum addition."""
    def __init__(
        self,
        target_register: [int, Sequence[int]],
        input_register: Union[int, Sequence[int]],
    ):
        self.target_register = target_register
        self.input_register = input_register

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self.target_register, self.input_register

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> 'Adder':
        return Adder(*new_registers)

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        return sum(register_values)
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        wire_symbols = [' + ' for _ in range(len(self.input_register)+len(self.target_register))]
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))

# Two qubit registers.
qreg1 = cirq.LineQubit.range(2)
qreg2 = cirq.LineQubit.range(2, 4)

# Define an adder gate for two 2D input and target qubits.
adder = Adder(input_register=[2, 2], target_register=[2, 2])

# Define the circuit.
circ = cirq.Circuit(
    cirq.X.on(qreg1[0]),
    cirq.X.on(qreg2[1]),
    adder.on(*qreg1, *qreg2),
    cirq.measure_each(*qreg1),
    cirq.measure_each(*qreg2)
)

# Display it.
# print("Circuit:\n")
# print(circ)

# Print the measurement outcomes.
# print("\n\nMeasurement outcomes:\n")
# print(cirq.sample(circ, repetitions=5).data)

"""Example of the unitary of an Adder gate."""
cirq.unitary(
    Adder(target_register=[2, 2],
          input_register=1)
).real

"""Defines the modular exponential gate used in Shor's algorithm."""
class ModularExp(cirq.ArithmeticGate):
    """Quantum modular exponentiation.

    This class represents the unitary which multiplies base raised to exponent
    into the target modulo the given modulus. More precisely, it represents the
    unitary V which computes modular exponentiation x**e mod n:

        V|y⟩|e⟩ = |y * x**e mod n⟩ |e⟩     0 <= y < n
        V|y⟩|e⟩ = |y⟩ |e⟩                  n <= y

    where y is the target register, e is the exponent register, x is the base
    and n is the modulus. Consequently,

        V|y⟩|e⟩ = (U**e|y)|e⟩

    where U is the unitary defined as

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y
    """
    def __init__(
        self,
        target: Sequence[int],
        exponent: Union[int, Sequence[int]],
        base: int,
        modulus: int
    ) -> None:
        if len(target) < modulus.bit_length():
            raise ValueError(
                f'Register with {len(target)} qubits is too small for modulus'
                f' {modulus}'
            )
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self.target, self.exponent, self.base, self.modulus

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> 'ModularExp':
        """Returns a new ModularExp object with new registers."""
        if len(new_registers) != 4:
            raise ValueError(
                f'Expected 4 registers (target, exponent, base, '
                f'modulus), but got {len(new_registers)}'
            )
        target, exponent, base, modulus = new_registers
        if not isinstance(target, Sequence):
            raise ValueError(
                f'Target must be a qubit register, got {type(target)}'
            )
        if not isinstance(base, int):
            raise ValueError(
                f'Base must be a classical constant, got {type(base)}'
            )
        if not isinstance(modulus, int):
            raise ValueError(
              f'Modulus must be a classical constant, got {type(modulus)}'
            )
        return ModularExp(target, exponent, base, modulus)

    def apply(self, *register_values: int) -> int:
        assert len(register_values) == 4
        target, exponent, base, modulus = register_values
        if target >= modulus:
            return target
        return (target * base**exponent) % modulus

    def _circuit_diagram_info_(
      self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.CircuitDiagramInfo:
        assert args.known_qubits is not None
        wire_symbols = [f't{i}' for i in range(len(self.target))]
        e_str = str(self.exponent)
        if isinstance(self.exponent, Sequence):
            e_str = 'e'
            wire_symbols += [f'e{i}' for i in range(len(self.exponent))]
        wire_symbols[0] = f'ModularExp(t*{self.base}**{e_str} % {self.modulus})'
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))

n = 15
L = n.bit_length()

# The target register has L qubits.
target = cirq.LineQubit.range(L)

# The exponent register has 2L + 3 qubits.
exponent = cirq.LineQubit.range(L, 3 * L + 3)

# Display the total number of qubits to factor this n.
# print(f"To factor n = {n} which has L = {L} bits, we need 3L + 3 = {3 * L + 3} qubits.")

# Pick some element of the multiplicative group modulo n.
x = 5

# Display (part of) the unitary. Uncomment if n is small enough.
# cirq.unitary(ModularExp(target, exponent, x, n))


"""Function to make the quantum circuit for order finding."""
def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:
    """Returns quantum circuit which computes the order of x modulo n.

    The circuit uses Quantum Phase Estimation to compute an eigenvalue of
    the following unitary:

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    Args:
        x: positive integer whose order modulo n is to be found
        n: modulus relative to which the order of x is to be found

    Returns:
        Quantum circuit for finding the order of x modulo n
    """
    L = n.bit_length()
    target = cirq.LineQubit.range(L)
    exponent = cirq.LineQubit.range(L, 3 * L + 3)

    # Create a ModularExp gate sized for these registers.
    mod_exp = ModularExp([2] * L, [2] * (2 * L + 3), x, n)

    return cirq.Circuit(
        cirq.X(target[L - 1]),
        cirq.H.on_each(*exponent),
        mod_exp.on(*target, *exponent),
        cirq.qft(*exponent, inverse=True),
        cirq.measure(*exponent, key='exponent'),
    )


"""Example of the quantum circuit for period finding."""
n = 15
x = 7
circuit = make_order_finding_circuit(x, n)
# print(circuit)

"""Measuring Shor's period finding circuit."""
circuit = make_order_finding_circuit(x=5, n=6)
res = cirq.sample(circuit, repetitions=8)

# print("Raw measurements:")
# print(res)

# print("\nInteger in exponent register:")
# print(res.data)


def process_measurement(result: cirq.Result, x: int, n: int) -> Optional[int]:
    """Interprets the output of the order finding circuit.

    Specifically, it determines s/r such that exp(2πis/r) is an eigenvalue
    of the unitary

        U|y⟩ = |xy mod n⟩  0 <= y < n
        U|y⟩ = |y⟩         n <= y

    then computes r (by continued fractions) if possible, and returns it.

    Args:
        result: result obtained by sampling the output of the
            circuit built by make_order_finding_circuit

    Returns:
        r, the order of x modulo n or None.
    """
    # Read the output integer of the exponent register.
    exponent_as_integer = result.data["exponent"][0]
    exponent_num_bits = result.measurements["exponent"].shape[1]
    eigenphase = float(exponent_as_integer / 2**exponent_num_bits)

    # Run the continued fractions algorithm to determine f = s / r.
    f = fractions.Fraction.from_float(eigenphase).limit_denominator(n)

    # If the numerator is zero, the order finder failed.
    if f.numerator == 0:
        return None

    # Else, return the denominator if it is valid.
    r = f.denominator
    if x**r % n != 1:
        return None
    return r

# Set n and x here
n = 6
x = 5

# print(f"Finding the order of x = {x} modulo n = {n}\n")
measurement = cirq.sample(circuit, repetitions=1)
# print("Raw measurements:")
# print(measurement)

# print("\nInteger in exponent register:")
# print(measurement.data)

r = process_measurement(measurement, x, n)
# print("\nOrder r =", r)
# if r is not None:
    # print(f"x^r mod n = {x}^{r} mod {n} = {x**r % n}")

def quantum_order_finder(x: int, n: int) -> Optional[int]:
    """Computes smallest positive r such that x**r mod n == 1.

    Args:
        x: integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: modulus of the multiplicative group.
    """
    # Check that the integer x is a valid element of the multiplicative group
    # modulo n.
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    # Create the order finding circuit.
    circuit = make_order_finding_circuit(x, n)

    # Sample from the order finding circuit.
    measurement = cirq.sample(circuit)

    # Return the processed measurement result.
    return process_measurement(measurement, x, n)

"""This completes our quantum implementation of an order finder, and the quantum part of Shor's algorithm.

# The complete factoring algorithm

We can use this quantum order finder (or the classical order finder) to complete Shor's algorithm. In the following code block, we add a few pre-processing steps which:

(1) Check if $n$ is even,

(2) Check if $n$ is prime,

(3) Check if $n$ is a prime power,

all of which can be done efficiently with a classical computer. Additionally, we add the last necessary post-processing step which uses the order $r$ to compute a non-trivial factor $p$ of $n$. This is achieved by computing $y = x^{r / 2} \text{ mod } n$ (assuming $r$ is even), then computing $p = \text{gcd}(y - 1, n)$.
"""

"""Functions for factoring from start to finish."""
def find_factor_of_prime_power(n: int) -> Optional[int]:
    """Returns non-trivial factor of n if n is a prime power, else None."""
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1**k == n:
            return c1
        c2 = math.ceil(c)
        if c2**k == n:
            return c2
    return None


def find_factor(
    n: int,
    order_finder: Callable[[int, int], Optional[int]] = quantum_order_finder,
    max_attempts: int = 30
) -> Optional[int]:
    """Returns a non-trivial factor of composite integer n.

    Args:
        n: Integer to factor.
        order_finder: Function for finding the order of elements of the
            multiplicative group of integers modulo n.
        max_attempts: number of random x's to try, also an upper limit
            on the number of order_finder invocations.

    Returns:
        Non-trivial factor of n or None if no such factor was found.
        Factor k of n is trivial if it is 1 or n.
    """
    # If the number is prime, there are no non-trivial factors.
    if sympy.isprime(n):
        # print("n is prime!")
        return None

    # If the number is even, two is a non-trivial factor.
    if n % 2 == 0:
        return 2

    # If n is a prime power, we can find a non-trivial factor efficiently.
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c

    for _ in range(max_attempts):
        # Choose a random number between 2 and n - 1.
        x = random.randint(2, n - 1)

        # Most likely x and n will be relatively prime.
        c = math.gcd(x, n)

        # If x and n are not relatively prime, we got lucky and found
        # a non-trivial factor.
        if 1 < c < n:
            return c

        # Compute the order r of x modulo n using the order finder.
        r = order_finder(x, n)

        # If the order finder failed, try again.
        if r is None:
            continue

        # If the order r is even, try again.
        if r % 2 != 0:
            continue

        # Compute the non-trivial factor.
        y = x**(r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c

    # print(f"Failed to find a non-trivial factor in {max_attempts} attempts.")
    return None

"""The function `find_factor` uses the `quantum_order_finder` by default, in which case it is executing Shor's algorithm. As previously mentioned, due to the large memory requirements for classically simulating this circuit, we cannot run Shor's algorithm for $n \ge 15$. However, we can use the classical order finder as a substitute."""

import time

"""Example of factoring via Shor's algorithm (order finding)."""
# Number to factor

# start = time.time()

# n = 12

# # Attempt to find a factor
# p = find_factor(n, order_finder=classical_order_finder)
# q = n // p

# print("Factoring n = pq =", n)
# print("p =", p)
# print("q =", q)

# end = time.time()

# print("Total time taken:", end - start)

# """Check the answer is correct."""
# p * q == n

"""Example of factoring via Shor's algorithm (order finding)."""
# Number to factor

input_data = sys.stdin.readline().strip().split(',')
input_data = [int(x) for x in input_data]

start = time.time_ns()

n = input_data[0]

# Attempt to find a factor
p = find_factor(n, order_finder=quantum_order_finder)
q = n // p

print("Factoring n = pq =", n)
print("p =", p)
print("q =", q)

end = time.time_ns()
print("Total time spent: {} nanoseconds".format(end - start))
# print("Total time taken: {0:.16f}".format(total_time), "nanoseconds")
# print("Total time taken: ")
# print(total_time)
# print('{0:.16f}'.format(1.6))
"""Check the answer is correct."""
# p * q == n