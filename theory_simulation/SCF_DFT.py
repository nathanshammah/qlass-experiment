import numpy as np
from pyscf import gto, scf
from qiskit.quantum_info import Pauli, PauliList
from qlass.quantum_chemistry.hamiltonians import transformation_Hmatrix_Hqubit, sparsepauliop_dictionary
from qlass.vqe.ansatz import Bitstring_initial_states
from qlass.vqe import VQE
from perceval.algorithm import Sampler
import QDFT
import warnings
warnings.filterwarnings('ignore')

'''This code uses QLASS to perform QDFT calculations'''

def input_pyscf(n_hydrogens: int = 2, R: float = 1.2, XC_functional: str = 'SVWN',
):
    geometry = []
    numberof_qubits = int(np.log2(n_hydrogens))

    for d in range(n_hydrogens // 2):
        geometry.append(("H", (0.0, 0.0, -(R / 2.0 + d * R))))
        geometry.append(("H", (0.0, 0.0, +(R / 2.0 + d * R))))

    molecule = gto.M(atom=geometry, basis="sto-3g")
    mf = scf.RKS(molecule)
    mf.xc = 'SVWN'
    # mf.max_cycle = 1

    mf.scf()
    H_core = mf.get_hcore()
    S_AO = mf.get_ovlp()
    D_AO = mf.get_init_guess()
    # D_AO = mf.make_rdm1()
    E_nuc = mf.energy_nuc()
    F_AO = mf.get_fock(dm=D_AO)  # H_core + V_HF
    # F_AO = mf.get_fock()  # H_core + V_HF

    mo_energy = mf.mo_energy

    AO_kinetic = molecule.intor("int1e_kin")
    AO_potential = molecule.intor("int1e_nuc")  # V_nuc

    # Kinetic + nuclear
    E_T = np.einsum('pq,pq->', D_AO, AO_kinetic)
    E_V = np.einsum('pq,pq->', D_AO, AO_potential)

    # XC energy from grid
    Exc = mf.energy_elec(dm=D_AO)[1]  # E_Hartree + E_XC

    # Total electronic energy
    E_elec = E_T + E_V + Exc

    # Total energy
    E_tot = E_elec + E_nuc

    print(f"Energy from guess density: {E_tot:.8f}")
    return D_AO, S_AO, AO_kinetic, AO_potential, E_nuc, mf, numberof_qubits

def executor(params, pauli_string):

    if not hasattr(executor, "proc"):
        executor.proc = []
        executor.samples_cache = []
    processors = Bitstring_initial_states(1, 2, params, pauli_string, cost="e-VQE")
    samplers = [Sampler(p) for p in processors]
    samples = [sampler.samples(10_000) for sampler in samplers]
    if not executor.proc:
        executor.proc.extend(processors)
        executor.samples_cache.extend(samples)

    return samples

def mesurements(H, lp, executor):
    from qlass.quantum_chemistry.hamiltonians import group_commuting_pauli_terms
    from qlass.utils.utils import _extract_samples_from_executor_result, normalize_samples, get_probabilities, pauli_string_bin,qubit_state_marginal, compute_energy

    loss = 0.0
    lst_energies: list[float] | None = None
    all_pauli_strings = []
    all_pauli_coefficients = []
    all_expectation_values = []

    # Use automatic grouping for optimized measurements
    grouped_hamiltonians = group_commuting_pauli_terms(H)

    for group in grouped_hamiltonians:
        # Each group contains mutually commuting terms
        # In the future, this could be optimized to measure entire groups simultaneously
        # For now, we process each term individually but with the grouping organization
        for pauli_string, coefficient in group.items():
            samples = executor(lp, pauli_string)

            # Handle different executor return formats
            sample_lists = [_extract_samples_from_executor_result(s) for s in samples]

            # Normalize samples to consistent format
            normalized_samples = [normalize_samples(sample_list) for sample_list in sample_lists]

            prob_dist = [
                get_probabilities(normalized_sample) for normalized_sample in normalized_samples
            ]
            pauli_bin = pauli_string_bin(pauli_string)

            qubit_state_marg = [qubit_state_marginal(pd) for pd in prob_dist]
            expectation = [compute_energy(pauli_bin, qsm) for qsm in qubit_state_marg]

            all_expectation_values.append(expectation)
            all_pauli_strings.append(pauli_string)
            all_pauli_coefficients.append(coefficient)

            energies = [coefficient * expect for expect in expectation]
            # Initialize accumulator on first iteration
            if lst_energies is None:
                lst_energies = [0.0] * len(energies)

            # Accumulate energies dynamically
            for i, energy in enumerate(energies):
                lst_energies[i] += energy

    return lst_energies, all_pauli_strings, all_pauli_coefficients, all_expectation_values

def sparsepauliop_to_dict(sp_op) -> dict:
    """
    Convert a SparsePauliOp into a dictionary mapping Pauli strings to coefficients.
    """
    # sp_op.paulis is a list of Pauli objects, sp_op.coeffs is a numpy array
    pauli_strings = [p.to_label() for p in sp_op.paulis]
    coeffs = []
    for c in sp_op.coeffs:
        if abs(c.imag) < 1e-12:   # negligible imaginary part
            coeffs.append(float(c.real))  # cast to plain Python float
        else:
            coeffs.append(complex(c))  # keep complex if significant
    return dict(zip(pauli_strings, coeffs))



#Input preparation: PySCF
Number_of_Hydrogen = 4
bond_length = 0.784
XC_function = "SVWN"

# QDFT inputs
SCF_maxiter = 20
E_conv = 1e-5
D_conv = 1e-3

D_AO, S_AO, AO_kinetic, AO_potential, E_nuc, mf, n_qubits = input_pyscf(Number_of_Hydrogen, bond_length, XC_function)
n_orbs = 2 ** n_qubits
# Compute the inverse square root of the overlap matrix S
S_eigval, S_eigvec = np.linalg.eigh(S_AO)
S_sqrt_inv = S_eigvec @ np.diag((S_eigval) ** (-1.0 / 2.0)) @ S_eigvec.T

Etot = 0.0
F_list = []
DIIS_error = []
for SCF_ITER in range(1, SCF_maxiter + 1):
    F_AO = mf.get_fock(dm=D_AO)
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
    print(f"Energy from classical diagonalazation: {np.sum(eps[:2])}")


    H_qubit, IJ_op_list, full_pauliop_list, full_pauli_list = QDFT.transformation_Hmatrix_Hqubit(F_OAO, n_qubits)
    H_qubit = sparsepauliop_to_dict(H_qubit)
    print(f"Printing Hamiltonian: \n {H_qubit}")
    full_paul_dic = {}
    for size in range(len(full_pauliop_list)):
        full_paul_dic[f"{full_pauliop_list[size].paulis[0]}"] = float(full_pauliop_list[size].coeffs.real[0])


    # Initialize the VQE solver
    vqe = VQE(
        hamiltonian=H_qubit,
        executor=executor,
        executor_type='sampling',
        num_params=4,  # Number of parameters in the linear entangled ansatz
        # optimizer="L-BFGS-B"

    )

    # Run the VQE optimization
    vqe_energy = vqe.run(
        max_iterations=80,
        verbose=True,
        weight_option="equi",
        cost="e-VQE",
        # jacobian="parameter_shift"
    )

    optimized_parameters = vqe.parameter_history[-1]
    print("vqe_parameters:", vqe.parameter_history)
    print(f"optimized paramters: {optimized_parameters}")

    energies, pauli_strings, pauli_coeffs, exp_vals = mesurements(full_paul_dic, optimized_parameters, executor)

    n_pauli = len(pauli_strings)
    n_occ = len(exp_vals[0])

    all_exp_values = np.zeros((n_occ, n_pauli))

    for p in range(n_pauli):
        for occ in range(n_occ):
            all_exp_values[occ, p] = exp_vals[p][occ]


    product_coeff = np.zeros((n_occ, n_orbs * (n_orbs + 1) // 2))

    for occ in range(n_occ):
        for ij in range(len(IJ_op_list)):
            for pauli in IJ_op_list[ij]:
                product_coeff[occ, ij] += pauli.coeffs[0] * all_exp_values[
                    occ, full_pauli_list.index(Pauli(pauli.paulis))] / 2.

    # Compute the density matrix
    D_OAO = np.zeros((n_orbs, n_orbs))
    for i in range(n_orbs):
        for j in range(i + 1):
            for occ in range(n_occ):
                ij = i * (i + 1) // 2 + j
                D_OAO[i, j] += product_coeff[occ, ij]
            D_OAO[j, i] = D_OAO[i, j]
    D_AO[:] = S_sqrt_inv @ D_OAO @ S_sqrt_inv
    D_AO_new = 2 * D_AO

    # Compute total energy
    E_T = np.einsum('pq,pq->', D_AO_new, AO_kinetic)
    E_V = np.einsum('pq,pq->', D_AO_new, AO_potential)
    Exc = mf.energy_elec(dm=D_AO_new)[1]  # E_Hartree + E_XC
    Etot_new = E_T + E_V + Exc + E_nuc

    Delta_E = abs(Etot_new - Etot)


    print(f"Iter {SCF_ITER}: E = {Etot_new:.8f} dE = {Delta_E:.2e} dRMS = {dRMS:.2e}")

    Etot = Etot_new
    D_AO = D_AO_new
    print("printing density: \n", D_AO)


    if Delta_E < E_conv and dRMS < D_conv:
        print(f"SCF Converged at iteration {SCF_ITER}, Energy = {Etot:.8f} Ha")
        break
    if SCF_ITER == SCF_maxiter:
        print(f"SCF is not converged, Energy = {Etot:.8f} Ha")
