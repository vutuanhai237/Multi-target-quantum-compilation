import typing
import qiskit
import random
import numpy as np
import qtm.random_circuit


def divide_circuit(qc: qiskit.QuantumCircuit, percent) -> qiskit.QuantumCircuit:
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qc1.copy()
    stop = 0
    for x in qc:
        qc1.append(x[0], x[1])
        stop += 1
        if qc1.depth() / qc.depth() >= percent:
            for x in qc[stop:]:
                qc2.append(x[0], x[1])
            return qc1, qc2
    return qc1, qc2


def divide_circuit_by_depth(qc: qiskit.QuantumCircuit, depth) -> qiskit.QuantumCircuit:
    def look_forward(qc, x):
        return qc.append(x[0],x[1])
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qc1.copy()
    stop = 0
    for i in range(len(qc)):
        qc1.append(qc[i][0], qc[i][1])
        stop += 1
        if qc1.depth() == depth and i + 1 < len(qc) and look_forward(qc1, qc[i+1]) == depth + 1 :
            for x in qc[stop:]:
                qc2.append(x[0], x[1])
            return qc1, qc2
    return qc1, qc2


def fight(population):
    individuals = random.sample(population, 2)
    return individuals[0] if individuals[0].fitness > individuals[1].fitness else individuals[1]


def random_mutate(population, prob, mutate_func):
    random_individual_index = np.random.randint(0, len(population))
    random_value = random.random()
    if random_value < prob:
        print(random_value)
        print(f'Mutate {random_individual_index}')
        population[random_individual_index].mutate(mutate_func)
    return population


def compose_circuit(qcs: typing.List[qiskit.QuantumCircuit]) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        qcs (typing.List[qiskit.QuantumCircuit]): set of quantum circuit

    Returns:
        qiskit.QuantumCircuit: composed quantum circuit
    """
    qc = qiskit.QuantumCircuit(qcs[0].num_qubits)
    i = 0
    num_params = 0
    for sub_qc in qcs:
        num_params += len(sub_qc.parameters)
    thetas = qiskit.circuit.ParameterVector('theta', num_params)
    for sub_qc in qcs:
        for instruction in sub_qc:
            if len(instruction[0].params) == 1:
                instruction[0].params[0] = thetas[i]
                i += 1
            if len(instruction[0].params) == 3:
                instruction[0].params[0] = thetas[i:i+1]
                i += 2
            qc.append(instruction[0], instruction[1])
    qc.draw()
    return qc
