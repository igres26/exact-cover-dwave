import sympy
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name, instance):
    """Collect data from .txt file that characterizes the problem instance.
    Args:
        file_name (str): name of the file that contains the instance information.
        instance (str): number of intance to use.

    Returns:
        control (list): important parameters of the instance.
            [number of qubits, number of clauses, number of ones in the solution]
        solution (list): list of the correct outputs of the instance for testing.
        clauses (list): list of all clauses, with the qubits each clause acts upon.
    """
    file = open('n{q}i{i}.txt'.format(q=file_name, i=instance), 'r')
    control = list(map(int, file.readline().split()))
    solution = list(map(str, file.readline().split()))
    clauses = [list(map(int, file.readline().split())) for _ in range(control[1])]
    return control, solution, clauses


def times(qubits, clauses):
    """Count the times each qubit appears in a clause to normalize H0.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses of the Exact Cover instance.

    Returns:
        times (list): number of times a qubit apears in all clauses.
    """
    times = np.zeros(qubits)
    for clause in clauses:
        for num in clause:
            times[num-1] += 1
    return times


def h_problem(qubits, clauses):
    """Hamiltonian that satisfies all Exact Cover clauses.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses for an Exact Cover instance.

    Returns:
        sham (sympy.Expr): Symbolic form of the problem Hamiltonian.
        smap (dict): Dictionary that maps the symbols that appear in the
            Hamiltonian to the corresponding matrices and target qubits.
    """
    z = sympy.symbols(" ".join((f"z{i}" for i in range(qubits))))
    s = []
    for x in z:
        s.append((x**2, x))
    smap = {s: i for i, s in enumerate(z)}
    sham = sympy.expand(sum((sum(z[i - 1] for i in clause) - 1) ** 2 for clause in clauses))
    sham = sham.subs(s)
    return sham, smap


def h_weights(qubits, clauses, times):
    """Hamiltonian that satisfies all Exact Cover clauses.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses for an Exact Cover instance.

    Returns:
        sham (sympy.Expr): Symbolic form of the problem Hamiltonian.
        smap (dict): Dictionary that maps the symbols that appear in the
            Hamiltonian to the corresponding matrices and target qubits.
    """
    z = sympy.symbols(" ".join((f"z{i}" for i in range(qubits))))
    s = []
    for x in z:
        s.append((x**2, x))
    smap = {s: i for i, s in enumerate(z)}
    sham = sympy.expand(sum((sum(z[i - 1] for i in clause) - 1) ** 2 for clause in clauses))
    sham = sham.subs(s)
    return sham, smap


def symbolic_to_dwave(symbolic_hamiltonian, symbol_num):
    """Transforms a symbolic Hamiltonian to a dictionary of targets and matrices.
    
    Works for Hamiltonians with one and two qubit terms only.
    
    Args:
        symbolic_hamiltonian: The full Hamiltonian written with symbols.
        symbol_num: Dictionary that maps each symbol that appears in the 
            Hamiltonian to its target.
    
    Returns:
       Q (dict): Dictionary with the interactions to send to the DWAVE machine.
       overall_constant (int): Constant that cannot be given to DWAVE machine.
    """ 
    Q = {}
    overall_constant = 0
    for term in symbolic_hamiltonian.args:
        if not term.args:
            expression = (term,)
        else:
            expression = term.args

        symbols = [x for x in expression if x.is_symbol]
        numbers = [x for x in expression if not x.is_symbol]

        if len(numbers) > 1:
            raise ValueError("Hamiltonian must be expanded before using this method.")
        elif numbers:
            constant = float(numbers[0])
        else:
            constant = 1

        if not symbols:
            overall_constant += constant

        elif len(symbols) == 1: 
            target = symbol_num[symbols[0]]
            Q[(target, target)] = constant

        elif len(symbols) == 2:
            target1 = symbol_num[symbols[0]]
            target2 = symbol_num[symbols[1]]
            Q[(target1, target2)] = constant

        else:
            raise ValueError("Only one and two qubit terms are allowed.")
    
    return Q, overall_constant