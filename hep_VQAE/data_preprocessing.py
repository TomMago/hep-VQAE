import numpy as np
import cirq
from sklearn.decomposition import PCA

def PCA_reduce(data, pca_components, val_data=None, test_data=None):
    data_flat = np.array(data).reshape(data.shape[0],-1)
    pca_dims = PCA(n_components=pca_components)
    data_reduced = pca_dims.fit_transform(data_flat)

    ret = data_reduced


    if not val_data is None:
        val_data_flat = np.array(val_data).reshape(val_data.shape[0],-1)
        val_data_reduced = pca_dims.transform(val_data_flat)
        ret = (data_reduced, val_data_reduced)

    if not test_data is None:
        test_data_flat = np.array(test_data).reshape(test_data.shape[0],-1)
        test_data_reduced = pca_dims.transform(test_data_flat)
        ret = (data_reduced, val_data_reduced, test_data_reduced)

    return ret

def data_to_circuit(data):
     qubits = cirq.GridQubit.rect(1, len(data))
     values = np.ndarray.flatten(data).astype(np.float32)
     values = values * (2 * np.pi) - np.pi
     # circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(data, qubits) if not v==0])
     circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(data, qubits)])
     return circuit
