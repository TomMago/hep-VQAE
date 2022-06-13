import numpy as np
import cirq
from sklearn.decomposition import PCA

def PCA_reduce(data, pca_components):
    data_flat = np.array(data).reshape(data.shape[0],-1)

    pca_dims = PCA(n_components=pca_components)
    data_reduced = pca_dims.fit_transform(data_flat)
    return data_reduced

def data_to_circuit(data):
     qubits = cirq.GridQubit.rect(1, len(data))
     values = np.ndarray.flatten(datac).astype(np.float32)
     values = values * (2 * np.pi) - np.pi
     # circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(data, qubits) if not v==0])
     circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(data, qubits)])
     return circuit
