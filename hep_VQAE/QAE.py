import numpy as np
import cirq
import tensorflow_quantum as tfq
import sympy
from itertools import product


class QAE_layer(tfq.layers.PQC):
    def __init__(self, data_qbits, latent_qbits, layers):

        self.parameters = 0

        network_qbits = data_qbits + (data_qbits - latent_qbits)

        qbits = [cirq.GridQubit(0, i) for i in range(network_qbits + 1 + data_qbits)]

        model_circuit = self._build_circuit(qbits[:network_qbits], data_qbits, latent_qbits, qbits[network_qbits], qbits[network_qbits + 1:], layers)
        readout_operator = [cirq.Z(qbits[network_qbits])]
        super().__init__(model_circuit, readout_operator)


    def _layer(self, qbits):
        circ = cirq.Circuit()
        for i in reversed(range(len(qbits)-1)):
            circ += cirq.CNOT(qbits[i], qbits[i+1])
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
            circ += cirq.rz(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        return circ


    def _build_circuit(self, qbits, data_qbits, latent_qbits, swap_qbit, reference_qbits, layers):
        c = cirq.Circuit()
        for i in range(layers):
            c += self._layer(qbits[:data_qbits])
        for i in range(layers):
            c += self._layer(qbits[data_qbits - latent_qbits:])
        # SWAP Test
        c += cirq.H(swap_qbit)
        for i, j in product(range(data_qbits), range(data_qbits - latent_qbits, len(qbits))):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, reference_qbits[i], qbits[j])
        c += cirq.H(swap_qbit)
        return c
