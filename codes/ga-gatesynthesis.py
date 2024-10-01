import sys, qiskit, typing
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.core import ansatz, state, measure
from qoop.backend import constant, utilities
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
from qoop.evolution.utilities import create_params

def grad_func(u, vdagger, thetas):
    S = 0.001
    grad = []

    for i in range(len(thetas)):
        thetas_new_plus = thetas.copy()
        thetas_new_minus = thetas.copy()
        
        thetas_new_plus[i] += S
        C1 = loss_new(u, vdagger, thetas_new_plus)

        thetas_new_minus[i] -= S
        C2 = loss_new(u, vdagger, thetas_new_minus)

        grad.append((-1)*(C1 - C2) / (2 * S))

    return np.array(grad)
    
def loss_new(
    u: qiskit.QuantumCircuit,
    vdagger: qiskit.QuantumCircuit,
    thetass: np.ndarray,  # Changed to np.ndarray for Qiskit compatibility
):
    Uf = vdagger
    
    # Assign parameters to the quantum circuit
    U = qiskit.quantum_info.Operator(u.assign_parameters(thetass)).data
    n = u.num_qubits
    
    f = np.abs(np.trace(Uf @ U) / 2**n )**2
    
    return 1-f



def gate_synthesis_fitness(qc: qiskit.QuantumCircuit):
    v_dagger = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0.]]) + 0j
    fidelities = []
    alpha = 0.5
    thetass = np.zeros(qc.num_parameters) + np.pi/3
    fidelities.append(loss_new(qc, v_dagger, thetass))
    m_grad = np.zeros(qc.num_parameters)
    v_grad = np.zeros(qc.num_parameters)
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    for iter in range(1000):
        internal_derivative_loss = grad_func(qc, v_dagger, thetass)
        # --- Adam optimizer --- #
        m_grad = beta1 * m_grad + (1 - beta1) * internal_derivative_loss
        v_grad = beta2 * v_grad + (1 - beta2) * internal_derivative_loss ** 2
        mhat = m_grad / (1 - beta1 ** (iter + 1))
        vhat = v_grad / (1 - beta2 ** (iter + 1))
        thetass += alpha * mhat / (np.sqrt(vhat) + epsilon)
        # --- Adam optimizer --- #
        fidelities.append(loss_new(qc, v_dagger, thetass))
        print(fidelities[-1])
        if np.abs(fidelities[-1] - fidelities[-2]) < 1e-12:
            break

    return fidelities[-1], thetass

def compilation_fitness_toffoli(qc: qiskit.QuantumCircuit):
    betas0 = []
    Uf = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0.]]) + 0j
    
    qsp = QuantumStatePreparation(
        u=qc,
        target_state= np.conjugate(Uf).T
        ).fit(num_steps=100, metrics_func=['loss_basic'])

    betas0.append(1-qsp.compiler.metrics['loss_basic'][-1])
    
    
    return betas0

num_qubits = 3
def super_evol(_depth, _num_circuit, _num_generation):
    env_metadata = EEnvironmentMetadata(
        num_qubits = num_qubits,
        depth = _depth,
        num_circuit = _num_circuit,
        num_generation = _num_generation,
        prob_mutate=3/(_depth * _num_circuit)
    )
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func=compilation_fitness_toffoli,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.layerflip_mutate,
        threshold_func=threshold.compilation_threshold
    )
    env.set_filename(f'n={num_qubits},d={_depth},n_circuit={_num_circuit},n_gen={_num_generation}')
    env.evol(verbose=1)
# def multiple_compile(params):
#     import concurrent.futures
#     executor = concurrent.futures.ProcessPoolExecutor()
#     results = executor.map(bypass_compile, params)
#     return results

# super_evol(_depth, _num_circuit, _num_generation) => super_evol(5, 16, 20)
# _num_circuit must %4 = 0
# main
if __name__ == '__main__':
    # depths = [2,3,4] # 5 qubits case
    # num_circuits = [4,8,16,32]
    # num_generations = [10,20,30,40, 50]
    # params = create_params(depths, num_circuits, num_generations)
    # multiple_compile(params)
    super_evol(30, 32, 4)