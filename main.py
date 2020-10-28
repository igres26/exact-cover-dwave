#!/usr/bin/env python
import numpy as np
import functions
import argparse
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from greedy import SteepestDescentSolver
import dimod
import dwave.inspector


def main(nqubits, instance, T, chainstrength, numruns, greedy, inspect):
    """

    Args:
        nqubits (int): number of qubits for the file that contains the
            information of an Exact Cover instance.
        instance (int): intance used for the desired number of qubits.
        T (float): 
        

    Returns:
        
    """
    control, solution, clauses = functions.read_file(nqubits, instance)
    nqubits = int(control[0])
    times = functions.times(nqubits, clauses)
    sh, smap = functions.h_problem(nqubits, clauses)
    Q, constant = functions.symbolic_to_dwave(sh, smap)
    

    model = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
    if not chainstrength:
        chainstrength = dwave.embedding.chain_strength.uniform_torque_compensation(model)
        print(f'Automatic chain strength: {chainstrength}\n')
    else:
        print(f'Chosen chain strength: {chainstrength}\n')

    if solution:
        print(f'Target solution that solves the problem: {" ".join(solution)}\n')

    sampler = EmbeddingComposite(DWaveSampler())
    if greedy:
        solver_greedy = SteepestDescentSolver()
        sampleset = sampler.sample(model, chain_strength=chainstrength, num_reads=numruns, annealing_time=T, answer_mode='raw')
        response = solver_greedy.sample(model, initial_states=sampleset)
    else:
        response = sampler.sample(model, chain_strength=chainstrength, num_reads=numruns, annealing_time=T, answer_mode='histogram')
    
    
    best_sample = response.record.sample[0]
    best_energy = response.record.energy[0]
    print(f'Best result found: {best_sample}\n')
    print(f'With energy: {best_energy+constant}\n')
    good_samples = response.record.sample[:min(len(response.record.sample), nqubits)]
    good_energies = response.record.energy[:min(len(response.record.energy), nqubits)]
    print(f'The best {len(good_samples)} samples found in the evolution are:\n')
    for i in range(len(good_samples)):
        print(f'Sample: {good_samples[i]}    with energy: {good_energies[i]+constant}\n')

    if inspect:
        dwave.inspector.show(response)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=8, type=int)
    parser.add_argument("--instance", default=1, type=int)
    parser.add_argument("--T", default=20, type=float)
    parser.add_argument("--chainstrength", default=None, type=float)
    parser.add_argument("--numruns", default=100, type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = vars(parser.parse_args())
    main(**args)