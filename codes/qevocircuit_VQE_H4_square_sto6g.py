import sys, qiskit, typing
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.vqe import vqe, utilities
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
<<<<<<< HEAD
=======
import time
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
#%load_ext autoreload
#%autoreload 2

R = 1.8

<<<<<<< HEAD
h2 = lambda distances:  f"H 0 0 {distances[0]}; H 0 0 {distances[1]}"
h4_square = lambda distances: f"H 0 {R*np.sin(distances[0]/180*np.pi/2)} {R*np.cos(distances[0]/180*np.pi/2)}; H 0 {-R*np.sin(distances[0]/180*np.pi/2)} {-R*np.cos(distances[0]/180*np.pi/2)}; H 0 {-R*np.sin(distances[0]/180*np.pi/2)} {R*np.cos(distances[0]/180*np.pi/2)}; H 0 {R*np.sin(distances[0]/180*np.pi/2)} {-R*np.cos(distances[0]/180*np.pi/2)} "
lih = lambda distances: f"Li 0 0 {distances[0]}; H 0 0 {distances[1]}"

#file = open('Li-H molecules-sto-6g-200.txt', 'r').readlines()
file = open('H4 square molecules-sto6g-200.txt', 'r').readlines()
basis_set = 'sto6g'


=======
#h2 = lambda distances:  f"H 0 0 {distances[0]}; H 0 0 {distances[1]}"
h4_square = lambda distances: f"H 0 {R*np.sin(distances[0]/180*np.pi/2)} {R*np.cos(distances[0]/180*np.pi/2)}; H 0 {-R*np.sin(distances[0]/180*np.pi/2)} {-R*np.cos(distances[0]/180*np.pi/2)}; H 0 {-R*np.sin(distances[0]/180*np.pi/2)} {R*np.cos(distances[0]/180*np.pi/2)}; H 0 {R*np.sin(distances[0]/180*np.pi/2)} {-R*np.cos(distances[0]/180*np.pi/2)} "
#lih = lambda distances: f"Li 0 0 {distances[0]}; H 0 0 {distances[1]}"

#file = open('Li-H molecules-sto-6g-200.txt', 'r').readlines()
file = open('H4-square-molecules-sto6g-2000.txt', 'r').readlines()
#file =  open('Hydrogen molecules-631g-200.txt', 'r').readlines()
#file =  open('Hydrogen-molecules-sto6g-200.txt', 'r').readlines()

basis_set = 'sto6g'
#basis_set = '631g'

num_qubits=8
depth=11
num_circuit=8
num_generation=20
prob_mutate=3/(depth * num_circuit)  # Mutation rate / (depth * num_circuit)
num_points = 10	
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180

def general_VQE_atom(distances: []):
    def VQE_atom(qc: qiskit.QuantumCircuit):
        return VQE_fitness(qc, 
                           # Replace atom here, below function returns text such as "Li 0 0 0; H 0 0 0.25"
                           atom =h4_square(distances), 
                           # Replace basis here
                           basis = basis_set) #631g
    return VQE_atom

def VQE_fitness(qc: qiskit.QuantumCircuit, atom: str, basis: str) -> float:
    """General VQE fitness

    Args:
        qc (qiskit.QuantumCircuit): ansatz
        atom (str): describe for atom
        basis (str): VQE basis

    Returns:
        float: similarity between experiment results and theory results
    """
    computation_value = vqe.general_VQE(qc, atom, basis)
    # I need to modify this
    #exact_value = eval(file[round((eval(atom.split(' ')[-1])-0.25)/((2.5-0.25)/199))].split(" ")[-2])

    # for h4 square
<<<<<<< HEAD
    exact_value = eval(file[round(((np.arccos(np.abs(eval(atom.split(' ')[-2]))/R)/np.pi*180*2)-70)/(110-70)*199)].split(' ')[-2])
    
    return utilities.similarity(computation_value, exact_value)

def VQE_H4_sto6g_fitness(qc: qiskit.QuantumCircuit) -> float:
=======
    exact_value = eval(file[round(((np.arccos(np.abs(eval(atom.split(' ')[-2]))/R)/np.pi*180*2)-70)/(110-70)*1999)].split(' ')[-2])
    
    return utilities.similarity(computation_value, exact_value)

def VQE_H4_square_sto6g_fitness(qc: qiskit.QuantumCircuit) -> float:
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
    """Fitness function for atom_basis_fitness

    Args:
        qc (qiskit.QuantumCircuit): ansatz

    Returns:
        float: fitness value
    """
<<<<<<< HEAD
    num_points = 10
    # Create pairs of distanc
    # list_distances = list(zip([0]*num_points, np.linspace(70, 110, num_points)))
=======
    #num_points = num_points
    # Create pairs of distanc
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
    list_distances = [[_] for _ in np.linspace(70,110, num_points)]
    fitnesss = []
    # Run for num_points
    for distances in list_distances:
        # Below is fitness function at special point, this function has only qc as parameter
        specific_VQE_molecules: typing.FunctionType = general_VQE_atom(distances)
        fitnesss.append(specific_VQE_molecules(qc))
    return np.mean(fitnesss)



env_metadata = EEnvironmentMetadata(
<<<<<<< HEAD
    num_qubits=8,
    depth=3,
    num_circuit=5,
    num_generation=10,
    prob_mutate=3/(5 * 5)  # Mutation rate / (depth * num_circuit)
=======
    num_qubits = num_qubits,
    depth= depth,
    num_circuit = num_circuit,
    num_generation = num_generation,
    prob_mutate= prob_mutate  # Mutation rate / (depth * num_circuit)
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
)
env = EEnvironment(
    metadata=env_metadata,
    # Fitness function alway has the function type: qiskit.QuantumCircuit -> float
<<<<<<< HEAD
    fitness_func=VQE_H4_sto6g_fitness,
=======
    fitness_func=VQE_H4_square_sto6g_fitness,
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
    selection_func=selection.elitist_selection,
    crossover_func=crossover.onepoint_crossover,
    mutate_func=mutate.layerflip_mutate,
    threshold_func=threshold.compilation_threshold
)

<<<<<<< HEAD
=======
localtime = time.localtime(time.time())

env.set_filename(f"{num_qubits}qubits_{num_points}points_{num_circuit}circuits_{depth}depth_{num_generation}generations_VQE_H4_square_{basis_set}_fitness_{localtime.tm_year}-{localtime.tm_mon}-{localtime.tm_mday}_1000")

>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
# Automatically save the results in the same level folder
env.evol(1)

computation_value = vqe.general_VQE(env.best_circuit, h4_square([70]), basis=basis_set)
print(computation_value)

# Load the result from folder
env2 = EEnvironment.load(
<<<<<<< HEAD
    './8qubits_VQE_H4_sto6g_fitness_2024-1-05', 
   VQE_H4_sto6g_fitness
=======
    './4qubits_VQE_H4_sto6g_fitness_2024-1-05', 
   VQE_H4_square_sto6g_fitness
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
)

env2.plot()