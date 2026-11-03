import numpy as np
import pandas as pd
import scipy as sp
from pyscf import gto, scf
import perceval as pcvl
from qiskit.quantum_info import Pauli, PauliList, Statevector, SparsePauliOp, Operator
from qlass.vqe.ansatz import list_of_ones
from qiskit.circuit.library import n_local
from qiskit import QuantumCircuit, transpile
from perceval.converters import QiskitConverter
import warnings

warnings.filterwarnings('ignore')


def input_pyscf(R: float = 1.5, XC_functional: str = 'SVWN',
                ):
    geometry = []

    geometry = [
        ("Li", (0.0, 0.0, -R / 2.0)),
        ("H", (0.0, 0.0, +R / 2.0)),
    ]

    molecule = gto.M(atom=geometry, basis="sto-3g")
    mf = scf.RKS(molecule)
    mf.xc = 'SVWN'
    mf.verbose = 4
    mf.scf()
    H_core = mf.get_hcore()
    S_AO = mf.get_ovlp()
    D_AO = mf.get_init_guess()
    E_nuc = mf.energy_nuc()
    F_AO = mf.get_fock(dm=D_AO)  # H_core + V_HF
    mo_energy = mf.mo_energy
    AO_kinetic = molecule.intor("int1e_kin")
    AO_potential = molecule.intor("int1e_nuc")  # V_nuc

    # Kinetic + nuclear
    E_T = np.einsum('pq,pq->', D_AO, AO_kinetic)
    E_V = np.einsum('pq,pq->', D_AO, AO_potential)

    # XC energy from grid
    n_full = molecule.nao  # This is 6

    Exc = mf.energy_elec(dm=D_AO)[1]  # E_Hartree + E_XC

    # Total electronic energy
    E_elec = E_T + E_V + Exc

    # Total energy
    E_tot = E_elec + E_nuc

    print(mf.mo_energy)

    print(f"Energy from guess density: {E_tot:.8f}")
    return D_AO, S_AO, AO_kinetic, AO_potential, E_nuc, mf


def reduce_matrix(matrix):
    keep_indices = [0, 1, 4, 5]
    return matrix[np.ix_(keep_indices, keep_indices)]


def restructure_matrix(matrix):
    Full_matrix = np.zeros((6, 6))
    keep_indices = [0, 1, 4, 5]
    Full_matrix[np.ix_(keep_indices, keep_indices)] = matrix
    return Full_matrix


def circuit(nqubit, n_states, lp, layers=1):
    initial_circuits = []

    if nqubit is None:
        num_qubits = 2
    else:
        num_qubits = nqubit
    for _i in range(n_states):
        initial_circuits += [QuantumCircuit(num_qubits)]
    """Initial states"""
    for state in range(n_states):  # binarystring representation of the integer
        for i in list_of_ones(state, num_qubits):
            initial_circuits[state].x(i)

    circuits = []
    for state in range(n_states):
        ansatz = n_local(
            num_qubits=num_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cx",
            entanglement="linear",
            reps=layers,
            insert_barriers=False,
        )
        # Prepend the initial state
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(initial_circuits[state], inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuits.append(circuit)

    """Assign parameters"""
    intial_states = []
    for c in range(len(initial_circuits)):
        sv = Statevector.from_instruction(initial_circuits[c])
        index = list(sv.data).index(1 + 0j)  # position of the "1"
        bitstring = format(index, f"0{sv.num_qubits}b")
        bits_list = [int(b) for b in bitstring]
        intial_states.append(pcvl.LogicalState(bits_list))
    circuits = [c.assign_parameters(lp) for c in circuits]
    ansatz_transpiled = [
        transpile(c, basis_gates=["u3", "cx"], optimization_level=3) for c in circuits
    ]
    q_converter = QiskitConverter(backend_name="SLOS")
    processors = [q_converter.convert(at, use_postselection=True) for at in ansatz_transpiled]

    p_processors = [p.with_input(input_state=ist) for p, ist in zip(processors, intial_states, strict=False)]

    return processors


'''decompose operator to pauli'''
pauli_matrices = {
    'I': np.array([[1, 0], [0, 1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
}


# Generate all tensor products of Pauli matrices for n qubits
def generate_pauli_strings(n):
    from itertools import product
    labels = ['I', 'X', 'Y', 'Z']
    pauli_strings = [''.join(p) for p in product(labels, repeat=n)]
    return pauli_strings


# Create a dictionary to map Pauli labels to their ndarray representations
def create_pauli_to_matrix(n):
    pauli_strings = generate_pauli_strings(n)
    pauli_to_matrix = {}
    for label in pauli_strings:
        matrix = pauli_matrices[label[0]]
        for char in label[1:]:
            matrix = np.kron(matrix, pauli_matrices[char])
        pauli_to_matrix[label] = matrix
    return pauli_to_matrix


# Function to decompose an operator into Pauli strings
def decompose_operator_to_pauli_list(operator):
    nqubits = int(np.log2(operator.dim[0]))  # Number of qubits
    pauli_to_matrix = create_pauli_to_matrix(nqubits)
    pauli_list = {}
    dim = operator.dim[0]

    # Iterate over all possible Pauli matrices
    for pauli_str, pauli_matrix in pauli_to_matrix.items():
        coeff = np.trace(np.dot(operator.data, pauli_matrix)) / dim

        # Check if coefficient is non-zero
        if np.abs(coeff) > 1e-10:
            pauli_list[pauli_str] = pauli_list.get(pauli_str, 0) + coeff

    # Convert to a list of tuples (Pauli object, coefficient)
    pauli_list = [(Pauli(pauli_str), coeff) for pauli_str, coeff in pauli_list.items()]
    return pauli_list


P = [Pauli('X'), Pauli('Y'), Pauli('Z'), Pauli('I')]
X, Y, Z, I = Operator(P[0]).data, Operator(P[1]).data, Operator(P[2]).data, Operator(P[3]).data


def list_Pauli_to_SparsePauliOp(list_pauli: list):
    pauli_list: list[Pauli] = []
    pauli_coeff: list[float] = []
    for lines in list_pauli:
        pauli_list.append(lines[0])
        pauli_coeff.append(lines[1])
    return SparsePauliOp(pauli_list, coeffs=pauli_coeff)


def transformation_Hmatrix_Hqubit(Hmatrix, nqubits):
    """
    This function transforms the Hamiltonian matrix into the qubit Hamiltonian.
    Each element of the Hamiltonian is defined in the basis of bitstring configurations,
    i.e. | 0000 ... 0 >, | 0000 ... 1 >, etc. Each value "1" corresponds to a qubit excitation
    that is given by the operation S^+.
    """
    H_qubit = 0.
    IJ_op_list = []  # list of operators for ( |phi_I><phi_J| + h.c. )
    full_pauliop_list = []  # list of pauli strings operator to measure the Hamiltonian and c_i c_j product for the density.
    full_pauli_list = []  # list of pauli strings to measure the Hamiltonian and c_i c_j product for the density.

    for i in range(len(Hmatrix[0])):
        for j in range(len(Hmatrix[0])):

            # < J |
            # i = | I >
            # j = < J |
            IJ_op = np.eye(2 ** nqubits)

            # First convert the integers into bitstrings:
            bitstring_i = bin(i)[2:].zfill(nqubits)
            bitstring_j = bin(j)[2:].zfill(nqubits)

            for qubit in range(nqubits):
                if int(bitstring_i[qubit]) == 0 and int(bitstring_j[qubit]) == 0:
                    op_1 = (((X + 1j * Y) @ (X - 1j * Y)) / 4.)
                    product_op = Operator(op_1)
                    # Construct the full tensor product
                    left_identity = Operator(np.eye(2 ** qubit))
                    right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                    full_op = left_identity.tensor(product_op).tensor(right_identity)
                    # Perform the matrix multiplication
                    IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 0 and int(bitstring_j[qubit]) == 1:
                    op_1 = (X + 1j * Y) / 2.
                    product_op = Operator(op_1)
                    # Construct the full tensor product
                    left_identity = Operator(np.eye(2 ** qubit))
                    right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                    full_op = left_identity.tensor(product_op).tensor(right_identity)
                    # Perform the matrix multiplication
                    IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 1 and int(bitstring_j[qubit]) == 0:
                    op_1 = (X - 1j * Y) / 2.
                    product_op = Operator(op_1)
                    # Construct the full tensor product
                    left_identity = Operator(np.eye(2 ** qubit))
                    right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                    full_op = left_identity.tensor(product_op).tensor(right_identity)
                    # Perform the matrix multiplication
                    IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 1 and int(bitstring_j[qubit]) == 1:
                    op_1 = (((X - 1j * Y) @ (X + 1j * Y)) / 4.)
                    product_op = Operator(op_1)
                    # Construct the full tensor product
                    left_identity = Operator(np.eye(2 ** qubit))
                    right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                    full_op = left_identity.tensor(product_op).tensor(right_identity)
                    # Perform the matrix multiplication
                    IJ_op = Operator(IJ_op) @ full_op

                # Add the element of the Hamiltonian matrix to H_qubit

            H_qubit += Hmatrix[i, j] * IJ_op

            # Calculate the adjoint of IJ_op and add it
            IJ_op_adjoint = IJ_op.transpose()

            IJ_op = IJ_op + IJ_op_adjoint

            IJ_op = decompose_operator_to_pauli_list(IJ_op)
            if j < i + 1:
                IJ_op_list.append(IJ_op)
                for num_pauli in range(len(IJ_op)):
                    pauli_string = IJ_op[num_pauli]
                    if pauli_string[0] not in full_pauli_list:
                        full_pauliop_list.append(SparsePauliOp(pauli_string[0]))
                        full_pauli_list.append(pauli_string[0])

    H_qubit = decompose_operator_to_pauli_list(H_qubit)
    H_qubit = list_Pauli_to_SparsePauliOp(H_qubit)
    # IJ_op_list = list_Pauli_to_SparsePauliOp(IJ_op_list)
    IJ_op_list_sp = []
    for list_values in IJ_op_list:
        IJ_op_list_sp.append(list_Pauli_to_SparsePauliOp(list_values))

    IJ_op_list = IJ_op_list_sp

    return H_qubit, IJ_op_list, full_pauliop_list, full_pauli_list


def mesurements(H, proc, nshots=1024):
    e = 0.
    for pauli_str, coeff in zip(H.paulis, H.coeffs):
        # Convert pauli_str to a standard string if it's a Qiskit object
        p_str = str(pauli_str)

        # Call the expectation function for ONE string at a time
        term_expectation = get_pauli_expectation(proc, p_str, nshots)
        e += coeff.real * term_expectation

    return e.real


def get_pauli_expectation(p, pauli_str, nshots=1024):
    """
    Computes the expectation value of a single Pauli string (e.g., 'ZX')
    by rotating the basis and simulating shots.
    """
    backend = p.backend
    cir = p.linear_circuit()
    int_st = p.input_state
    # Create a copy of the circuit to append measurement rotations
    c = cir.copy()

    # Apply rotations to the modes to measure in the X or Y basis
    for i, char in enumerate(pauli_str):
        mode_0 = i * 2
        mode_1 = i * 2 + 1
        if char == 'X':
            # Beam Splitter acts as a Hadamard for dual-rail
            c.add((mode_0, mode_1), pcvl.BS.H())
        elif char == 'Y':
            c.add(mode_0, pcvl.PS(np.pi / 2))
            c.add((mode_0, mode_1), pcvl.BS.H())

    # Simulate the distribution

    backend.set_circuit(c)
    backend.set_input_state(input_state=int_st)

    full_dist = backend.prob_distribution()
    states, probs_list, ev_list = [], [], []

    for state, prob in full_dist.items():
        # 2. POST-SELECTION:
        # Check if physical modes (0-3) have exactly 1 photon per pair
        # AND check if auxiliary modes (4-5) match the expected ancilla state
        # (Usually ancillas must end in a specific state for the gate to succeed)

        condition_q0 = (state[0] + state[1] == 1)
        condition_q1 = (state[2] + state[3] == 1)

        # If CX gate requires the ancilla to be in a specific state, add it here:
        # condition_ancilla = (state[4] == 0 and state[5] == 1)

        if condition_q0 and condition_q1:
            states.append(state)
            probs_list.append(prob)

            # 3. EIGENVALUE CALCULATION (Physical Modes Only)
            current_ev = 1
            for i, char in enumerate(pauli_str):

                if str(char) != "I":
                    # Only look at modes 0,1 or 2,3
                    val = 1 if state[i * 2] == 1 else -1
                    current_ev *= val
            ev_list.append(current_ev)
    probs = np.array(probs_list) / sum(probs_list)
    counts = np.random.multinomial(nshots, probs)

    return np.sum(counts * np.array(ev_list)) / nshots


def reconstruct_density_matrix(p, n_qubits=2, nshots=1024):
    import itertools
    # Define the Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    # Generate all possible Pauli strings (e.g., 'II', 'IX', ...)
    bases = ['I', 'X', 'Y', 'Z']
    pauli_strings = [''.join(p) for p in itertools.product(bases, repeat=n_qubits)]

    for s in pauli_strings:
        # Get <P>
        exp_val = get_pauli_expectation(p, s, nshots=nshots)

        # Compute the Kronecker product for the Pauli matrix P
        p_matrix = pauli_map[s[0]]
        for char in s[1:]:
            p_matrix = np.kron(p_matrix, pauli_map[char])

        # Add to the sum: rho = 1/2^n * sum(<P> * P)
        rho += exp_val * p_matrix

    return rho / dim


def _callback(param, H):
    c = cost_function(param, H)
    processors = circuit(2, 2, layers=1, lp=param)
    orb_1 = mesurements(H, processors[0])
    orb_2 = mesurements(H, processors[1])
    vqe_cost.append(c)
    EA.append(orb_1)
    EB.append(orb_2)


def _callback_fast(param, H):
    # IMPORTANT: do NOT recompute cost_function here (already evaluated by optimizer)

    processors = circuit(2, 2, layers=1, lp=param)

    # reuse same cost logic (or ideally store it from optimizer if available)
    c = cost_function(param, H)

    orb_1 = measurements_fast(H, processors[0])
    orb_2 = measurements_fast(H, processors[1])

    vqe_cost.append(c)
    EA.append(orb_1)
    EB.append(orb_2)


def cost_function(param, H, nshots=1024):
    processors = circuit(2, 2, layers=1, lp=param)
    w = []
    for i in range(len(processors)):
        w.append(1 / (len(processors)))
    cost = 0.
    tr = 0.
    for i, p in enumerate(processors):
        e = 0.
        for pauli_str, coeff in zip(H.paulis, H.coeffs):
            # Convert pauli_str to a standard string if it's a Qiskit object
            p_str = str(pauli_str)

            # Call the expectation function for ONE string at a time
            term_expectation = get_pauli_expectation(p, p_str, nshots)
            e += coeff.real * term_expectation
        cost += w[i] * e.real
        tr += e.real
    return cost


def cost_function_batch(params_batch, H, nshots=1024):
    results = []

    for param in np.atleast_2d(params_batch):

        # 1. build circuit / processor
        processors = circuit(2, 2, layers=1, lp=param)

        # 2. GET DISTRIBUTION
        backend = processors[0].backend
        cir = processors[0].linear_circuit()
        int_st = processors[0].input_state

        backend.set_circuit(cir)
        backend.set_input_state(input_state=int_st)

        full_dist = backend.prob_distribution()

        # 3. compute cost using same distribution
        w = [
            1 / (len(processors) ** 2) * (2 * len(processors) - 1 - 2 * i)
            for i in range(len(processors))
        ]

        cost = 0.0

        for i, p in enumerate(processors):
            e = 0.0

            for pauli_str, coeff in zip(H.paulis, H.coeffs):
                term = pauli_expectation_from_dist(full_dist, str(pauli_str), nshots)
                e += coeff.real * term

            cost += w[i] * e.real

        results.append(cost)

    return np.array(results)[0] if len(results) == 1 else np.array(results)


def parametershift_grad(vqefunction, param: np.ndarray, H, shots):
    shift = np.pi / 2
    grad = np.zeros_like(param)

    for i in range(len(param)):
        theta_plus = param.copy()
        theta_plus[i] += shift
        theta_minus = param.copy()
        theta_minus[i] -= shift
        grad[i] = 0.5 * (vqefunction(theta_plus, H, shots) - vqefunction(theta_minus, H, shots))
    return grad


def parametershift_grad_batch(vqefunction, param, H, shots):
    shift = np.pi / 2
    n = len(param)

    eye = np.eye(n)
    theta_plus = param + shift * eye
    theta_minus = param - shift * eye

    all_thetas = np.vstack([theta_plus, theta_minus])  # (2n, n)

    values = vqefunction(all_thetas, H, shots)

    f_plus = values[:n]
    f_minus = values[n:]

    return 0.5 * (f_plus - f_minus)


def get_distribution(proc):
    backend = proc.backend
    cir = proc.linear_circuit()
    int_st = proc.input_state

    backend.set_circuit(cir)
    backend.set_input_state(input_state=int_st)

    return backend.prob_distribution()


def pauli_expectation_from_dist(full_dist, pauli_str, nshots=1024):
    states = []
    probs = []
    ev = []

    for state, prob in full_dist.items():

        # post-selection
        if (state[0] + state[1] != 1) or (state[2] + state[3] != 1):
            continue

        states.append(state)
        probs.append(prob)

        val = 1
        for i, char in enumerate(pauli_str):
            if char != "I":
                val *= (1 if state[i * 2] == 1 else -1)

        ev.append(val)

    probs = np.array(probs)
    ev = np.array(ev)

    if probs.sum() == 0:
        return 0.0

    probs = probs / probs.sum()

    counts = np.random.multinomial(nshots, probs)

    return np.sum(counts * ev) / nshots


def measurements_fast(H, proc, nshots=1024):
    e = 0.0

    full_dist = get_distribution(proc)

    for pauli_str, coeff in zip(H.paulis, H.coeffs):
        p_str = str(pauli_str)

        term = pauli_expectation_from_dist(full_dist, p_str, nshots)

        e += coeff.real * term

    return e.real


# QDFT inputs
shots = 1000000000
SCF_maxiter = 12
E_conv = 1e-5
D_conv = 1e-4
XC_function = "SVWN"


n_qubits = 2
n_orbs = 2 ** n_qubits
n_occ = 2


vqe_cost = []
EA = []
EB = []

trail = 10
print(f"Number of shots: {shots:.1e}")
for tr in range(trail):
    # Compute the inverse square root of the overlap matrix S
    D_AO, S_AO, AO_kinetic, AO_potential, E_nuc, mf = input_pyscf(1.595, XC_function)
    S_AO = reduce_matrix(S_AO)
    AO_potential = reduce_matrix(AO_potential)
    AO_kinetic = reduce_matrix(AO_kinetic)
    S_eigval, S_eigvec = np.linalg.eigh(S_AO)
    S_sqrt_inv = S_eigvec @ np.diag((S_eigval) ** (-1.0 / 2.0)) @ S_eigvec.T

    Etot = 0.0
    F_list = []
    DIIS_error = []
    c_names = ["Iteration", "Total_energy", "E_T", "E_V", "E_Xc", "dE", "dRMS"]
    rows = []

    # params = np.random.rand(4)
    print("-" * 50)
    params = np.random.uniform(0, 2 * np.pi, 4)
    print("Initial random parameters: \n", params)
    print("-"*50)
    for SCF_ITER in range(1, SCF_maxiter + 1):
        F_AO = mf.get_fock(dm=D_AO)
        F_AO = reduce_matrix(F_AO)
        D_AO = reduce_matrix(D_AO)
        # DIIS error matrix
        diis_e = np.einsum('ij,jk,kl->il', F_AO, D_AO, S_AO) - np.einsum('ij,jk,kl->il', S_AO, D_AO, F_AO)
        diis_e = S_sqrt_inv @ diis_e @ S_sqrt_inv
        if SCF_ITER > 1:
            F_list.append(F_AO)
            DIIS_error.append(diis_e)
        dRMS = np.mean(diis_e ** 2) ** 0.5

        # DIIS extrapolation
        if SCF_ITER > 1:
            diis_count = len(F_list)
            B = np.empty((diis_count + 1, diis_count + 1))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0

            for num1, e1 in enumerate(DIIS_error):
                for num2, e2 in enumerate(DIIS_error):
                    if num2 > num1: continue
                    val = np.einsum('ij,ij->', e1, e2)
                    B[num1, num2] = val
                    B[num2, num1] = val

            # normalize
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
            resid = np.zeros(diis_count + 1)
            resid[-1] = -1

            # Solve Pulay equations, [Pulay:1980:393], Eqn. 6

            ci = np.linalg.solve(B, resid)

            # Calculate new fock matrix as linear
            # combination of previous fock matrices
            F_AO = np.zeros_like(F_AO)
            for num, c in enumerate(ci[:-1]):
                F_AO += c * F_list[num]

        # Transform to orthogonal AO basis
        F_OAO = S_sqrt_inv @ F_AO @ S_sqrt_inv
        eps, C_OAO = np.linalg.eigh(F_OAO)

        H_qubit, IJ_op_list, full_pauliop_list, full_pauli_list = transformation_Hmatrix_Hqubit(F_OAO, n_qubits)

        res = sp.optimize.minimize(cost_function, x0=params, args=(H_qubit, shots), method='COBYLA',
                                   options={"maxiter": 10000})
        optimized_parameters = res.x
        processors = circuit(2, 2, layers=1, lp=optimized_parameters)
        trH = 0.
        dim = n_qubits ** 2
        rho = np.zeros((dim, dim), dtype=complex)
        for state in range(n_occ):
            trH += mesurements(H_qubit, processors[state], shots)
            rho += reconstruct_density_matrix(processors[state], n_qubits, shots)
        density_csv = pd.DataFrame(rho)
        density_csv.to_csv(f"OAO_Density_Trail_{tr + 1}_SCF_iteration_{SCF_ITER}_VQE.csv")
        D_AO[:] = S_sqrt_inv @ rho @ S_sqrt_inv
        D_AO_new = 2 * D_AO
        # Compute total energy
        E_T = np.einsum('pq,pq->', D_AO_new, AO_kinetic)
        E_V = np.einsum('pq,pq->', D_AO_new, AO_potential)
        D_AO_new = restructure_matrix(D_AO_new)
        Exc = mf.energy_elec(dm=D_AO_new)[1]  # E_Hartree + E_XC
        Etot_new = E_T + E_V + Exc + E_nuc

        Delta_E = abs(Etot_new - Etot)

        print(f"Iter {SCF_ITER}: E = {Etot_new:.8f} dE = {Delta_E:.2e} dRMS = {dRMS:.2e}")
        row = {"Iteration": SCF_ITER, "Total_energy": Etot_new, "E_T": E_T, "E_V": E_V, "E_Xc": Exc, "dE": Delta_E,
               "dRMS": dRMS}
        rows.append(row)
        Etot = Etot_new
        D_AO = D_AO_new
        # plot_density(D_AO)
        print("printing density: \n", D_AO)
        AO_density_csv = pd.DataFrame(D_AO)
        AO_density_csv.to_csv(f"AO_Density_Trail_{tr + 1}_SCF_iteration_{SCF_ITER}_VQE.csv")
        tcost = pd.DataFrame({"cost": vqe_cost, "EA": EA, "EB": EB})
        vqe_cost.clear()
        EA.clear()
        EB.clear()
        tcost.to_csv(f"Trail_{tr + 1}_SCF_iteration_{SCF_ITER}_VQE.csv")
        if Delta_E < E_conv and dRMS < D_conv:
            print(f"SCF Converged at iteration {SCF_ITER}, Energy = {Etot:.8f} Ha")
            break
        if SCF_ITER == SCF_maxiter:
            print(f"SCF is not converged, Energy = {Etot:.8f} Ha")
    tenergy = pd.DataFrame(rows, columns=c_names)
    tenergy.to_csv(f"TotalEnergy_Trail_{tr + 1}.csv")

