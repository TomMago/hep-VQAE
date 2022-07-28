import numpy as np
import cirq
import tensorflow_quantum as tfq
import sympy
from itertools import product
from tensorflow.keras.models import Model
import tensorflow as tf


class QAE_layer(tfq.layers.PQC):

    def __init__(self, data_qbits, latent_qbits, layers):

        self.parameters = 0

        network_qbits = data_qbits + (data_qbits - latent_qbits)

        qbits = [cirq.GridQubit(0, i) for i in range(network_qbits + 1 + data_qbits)]

        model_circuit = self._build_circuit(qbits[:network_qbits], qbits[network_qbits:-1], data_qbits, latent_qbits, qbits[-1], layers)
        readout_operator = [cirq.Z(qbits[-1])]
        super().__init__(model_circuit, readout_operator)

    def _layer(self, qbits):
        circ = cirq.Circuit()
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        for i in range(len(qbits)):
            for j in range(i+1, len(qbits)):
                circ += cirq.CNOT(qbits[i], qbits[j])
        return circ


    def _build_circuit(self, network_qbits, reference_qbits, num_data_qbits, num_latent_qbits, swap_qbit, layers):
        c = cirq.Circuit()
        for i in range(layers):
            c += self._layer(network_qbits[:num_data_qbits])
        for i in range(layers):
            c += self._layer(network_qbits[num_data_qbits - num_latent_qbits:])
        # swap test
        c += cirq.H(swap_qbit)
        for i in range(num_data_qbits):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, reference_qbits[i], network_qbits[num_data_qbits - num_latent_qbits:][i])
        c += cirq.H(swap_qbit)
        return c


class QAE_model(Model):

    def __init__(self, data_qbits, latent_qbits, layers):
        super(QAE_model, self).__init__()
        self.latent_qbits = latent_qbits
        self.data_qbits = data_qbits
        self.num_layers = layers

        self.parameters = 0
        network_qbits = data_qbits + (data_qbits - latent_qbits)
        qbits = [cirq.GridQubit(0, i) for i in range(network_qbits + 1 + data_qbits)]

        self.model_circuit = self._build_circuit(qbits[:network_qbits], qbits[network_qbits:-1], data_qbits, latent_qbits, qbits[-1], self.num_layers)
        readout_operator = [cirq.Z(qbits[-1])]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(model_circuit, readout_operator),
        ])

    def _layer(self, qbits):
        circ = cirq.Circuit()
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        for i in range(len(qbits)):
            for j in range(i+1, len(qbits)):
                circ += cirq.CNOT(qbits[i], qbits[j])
        return circ


    def _build_circuit(self, network_qbits, reference_qbits, num_data_qbits, num_latent_qbits, swap_qbit, layers):
        c = cirq.Circuit()
        for i in range(layers):
            c += self._layer(network_qbits[:num_data_qbits])
        for i in range(layers):
            c += self._layer(network_qbits[num_data_qbits - num_latent_qbits:])
        c += cirq.H(swap_qbit)
        for i in range(num_data_qbits):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, reference_qbits[i], network_qbits[num_data_qbits - num_latent_qbits:][i])
        c += cirq.H(swap_qbit)
        return c

    def call(self, x):
        return self.model(x)



class SQAE_model(Model):

    def __init__(self, data_qbits, latent_qbits, layers):
        super(SQAE_model, self).__init__()
        self.latent_qbits = latent_qbits
        self.data_qbits = data_qbits
        self.num_layers = layers

        self.parameters = 0
        non_latent = data_qbits - latent_qbits
        qbits = [cirq.GridQubit(0, i) for i in range(data_qbits + non_latent + 1)]

        self.model_circuit = self._build_circuit(qbits[:data_qbits], qbits[data_qbits:-1], latent_qbits, qbits[-1], self.num_layers)
        readout_operator = [cirq.Z(qbits[-1])]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(self.model_circuit, readout_operator),
        ])

    def _layer(self, qbits):
        circ = cirq.Circuit()
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        for i in range(len(qbits)):
            for j in range(i+1, len(qbits)):
                circ += cirq.CNOT(qbits[i], qbits[j])
        return circ


    def _build_circuit(self, data_qbits, trash_qbits, num_latent_qbits, swap_qbit, layers):
        c = cirq.Circuit()
        #encoder
        for i in range(layers):
            c += self._layer(data_qbits)
        c += cirq.H(swap_qbit)
        for i in range(len(trash_qbits)):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, trash_qbits[i], data_qbits[num_latent_qbits:][i])
        c += cirq.H(swap_qbit)
        return c

    def call(self, x):
        return self.model(x)
