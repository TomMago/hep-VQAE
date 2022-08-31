import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import time
from qutip import Bloch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import imageio
import os

class baseSQAE:

    def __init__(self, data_qbits, latent_qbits, device, diff_method="best"):
        self.dev = device
        self.data_qbits = data_qbits
        self.latent_qbits = latent_qbits
        self.trash_qbits = self.data_qbits - self.latent_qbits
        self.total_qbits = data_qbits + self.trash_qbits + 1
        self.circuit_node = qml.QNode(self.circuit, device, diff_method=diff_method)
        self.latent_node = qml.QNode(self.visualize_latent_circuit, device, diff_method=diff_method)

        self.auc_hist = []
        self.animation_filenames = []
        self.bloch_animation_filenames = []

        self.parameters_shape = (2 * data_qbits,)
        self.data_shape = (data_qbits,)

        self.params = np.random.uniform(size=self.parameters_shape, requires_grad=True)

    def encoder(self, params, data):
        for j in range(self.data_qbits):
            qml.RY(params[0:2*self.data_qbits][j*2]
                   + params[0:2*self.data_qbits][j*2+1]*data[j], wires=j)
        qml.CNOT(wires=[self.data_qbits-1, 0])
        for k in range(self.data_qbits-1):
            qml.CNOT(wires=[k, k+1])

    def circuit(self, params, data):

        self.encoder(params, data)

        qml.Hadamard(wires=self.total_qbits-1)
        for i in range(self.trash_qbits):
            qml.CSWAP(wires=[self.total_qbits - 1, self.latent_qbits + i, self.data_qbits + i])
        qml.Hadamard(wires=self.total_qbits-1)
        return qml.expval(qml.PauliZ(self.total_qbits-1))

    def visualize_latent_circuit(self, params, data, latent_bit, measure_gate):

        self.encoder(params, data)

        return qml.expval(measure_gate(latent_bit))

    def plot_latent_space(self, latent_bit, x_test_bg, x_test_signal, save_fig=None):
        b = Bloch()
        points_bg_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_bg]
        points_bg_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_bg]
        points_bg_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_bg]
        x = [points_bg_x[i] for i in range(len(points_bg_x))]
        y = [points_bg_y[i] for i in range(len(points_bg_y))]
        z = [points_bg_z[i] for i in range(len(points_bg_z))]
        bloch_points_bg = [x,y,z]
        b.add_points(bloch_points_bg)

        points_sig_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_signal]
        points_sig_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_signal]
        points_sig_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_signal]
        xs = [points_sig_x[i] for i in range(len(points_sig_x))]
        ys = [points_sig_y[i] for i in range(len(points_sig_y))]
        zs = [points_sig_z[i] for i in range(len(points_sig_z))]
        bloch_points_sig = [xs, ys, zs]
        b.add_points(bloch_points_sig)

        if save_fig:
            b.save(name=save_fig)
        else:
            b.show()
        b.clear()

    def plot_circuit(self):

        data = np.random.uniform(size=self.data_shape)

        fig, _ = qml.draw_mpl(self.circuit_node)(self.params, data)
        fig.show()

    def train(self, x_train, x_val, learning_rate, epochs, batch_size,
              print_step_size=20, make_animation=False, save_auc=False, x_val_signal=None):

        self.train_hist = {'loss': [], 'val_loss': []}

        opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

        start = time.time()

        for it in range(epochs):
            start_b = time.time()
            for j, Xbatch in enumerate(self.iterate_minibatches(x_train, batch_size=batch_size)):
                cost_fn = lambda p: self.cost_batch(p, Xbatch)
                self.params = opt.step(cost_fn, self.params)
                print(j, end="\r")
                if j % print_step_size == 0 and not j == 0:
                    end_b = time.time()
                    loss = self.cost_batch(self.params, Xbatch)
                    print(f"Step: {j:<7} | Loss: {loss:<10.3} | avg step time {(end_b - start_b) / print_step_size :.3}")
                    start_b = time.time()
                    if make_animation:
                        filename = f'imgs/animation_{len(self.animation_filenames) + 1}.png'
                        self.animation_filenames.append(filename)
                        self.evaluate(x_val, x_val_signal, save_fig=filename)

                        bloch_filename = f'imgs/bloch_{len(self.bloch_animation_filenames) + 1}.png'
                        self.bloch_animation_filenames.append(bloch_filename)
                        self.plot_latent_space(3, x_val[:25], x_val_signal[:25], save_fig=bloch_filename)
                    if save_auc:
                        pred_bg = np.array([self.circuit_node(self.params, i) for i in x_val])
                        pred_signal = np.array([self.circuit_node(self.params, i) for i in x_val_signal])
                        bce_background = 1-pred_bg
                        bce_signal = 1-pred_signal
                        y_true = np.append(np.zeros(len(bce_background)), np.ones(len(bce_signal)))
                        y_pred = np.append(bce_background, bce_signal)
                        auc = roc_auc_score(y_true, y_pred)
                        self.auc_hist.append(auc)

            loss = self.cost_batch(self.params, x_train[:len(x_val)])
            val_loss = self.cost_batch(self.params, x_val)
            self.train_hist['loss'].append(loss)
            self.train_hist['val_loss'].append(val_loss)
            print("____")
            print(f"Epoch: {it:<5} | Loss: {loss:<10.3} | Val Loss {val_loss:.3}")
            print("____")

        end = time.time()

        print(f"Time for {epochs} epochs: {end - start}")

        if make_animation:
            print("Building animation...", end="\r")
            with imageio.get_writer('animation3.gif', mode='I', fps=8) as writer:
                for filename in self.animation_filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print("Gif saved, removing files..", end="\r")
            writer.close()
            for filename in set(self.animation_filenames):
                os.remove(filename)
            print("Done.")

            print("Building bloch animation...", end="\r")
            with imageio.get_writer('bloch3.gif', mode='I', fps=8) as writer:
                for filename in self.bloch_animation_filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print("Gif saved, removing files..", end="\r")
            writer.close()
            for filename in set(self.bloch_animation_filenames):
                os.remove(filename)
            print("Done.")

    def iterate_minibatches(self, data, batch_size):
        for start_idx in range(0, data.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield data[idxs]

    def cost_batch(self, params, batch):
        loss = 0.0
        for i in batch:
            sample_loss = self.circuit_node(params, i)
            loss = loss + (1 - sample_loss) ** 2
        return loss / len(batch)

    def evaluate(self, x_test_bg, x_test_signal, save_fig=None):

        pred_bg = np.array([self.circuit_node(self.params, i) for i in x_test_bg])
        if not save_fig: print("Median fidelities bg: ", np.median(pred_bg))

        pred_signal = np.array([self.circuit_node(self.params, i) for i in x_test_signal])
        if not save_fig: print("Median fidelities signal: ", np.median(pred_signal))

        bce_background = 1-pred_bg
        bce_signal = 1-pred_signal

        fig, axs = plt.subplots(1, 3, figsize=(11, 4))

        if not save_fig: print(f'Median background: {np.median(bce_background):.3}')
        if not save_fig: print(f'Median signal: {np.median(bce_signal):.3}')
        bins = np.histogram(np.hstack((bce_background, bce_signal)), bins=25)[1]
        axs[0].hist(bce_background, histtype='step', label="background", bins=bins)
        axs[0].hist(bce_signal, histtype='step', label="signal", bins=bins)
        #axs[0].set_xlim(0,0.5)
        axs[0].set_xlabel("loss")
        axs[0].legend()

        thresholds = np.linspace(0, max(np.max(bce_background), np.max(bce_signal)), 1000)

        accs = []
        for i in thresholds:
            num_background_right = np.sum(bce_background < i)
            num_signal_right = np.sum(bce_signal > i)
            acc = (num_background_right + num_signal_right)/(len(pred_bg) + len(pred_signal))
            accs.append(acc)

        if not save_fig: print(f'Maximum accuracy: {np.max(accs):.3}')
        axs[1].plot(thresholds, accs)
        axs[1].set_xlabel("anomaly threshold")
        axs[1].set_ylabel("tagging accuracy")
        #axs[1].set_xlim(0,0.5)

        y_true = np.append(np.zeros(len(bce_background)), np.ones(len(bce_signal)))
        y_pred = np.append(bce_background, bce_signal)
        auc = roc_auc_score(y_true, y_pred)
        if not save_fig: print(f'AUC: {auc:.4}')
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tnr = 1 - fpr
        x = np.linspace(0, 1, 200)
        y_rnd = 1 - x
        axs[2].plot(tnr, tpr, label=f"anomaly tagger, auc:{auc:.3}")
        axs[2].plot(x, y_rnd, label="random tagger", color='grey')
        axs[2].set_xlabel("fpr")
        axs[2].set_ylabel("tpr")
        axs[2].legend(loc="lower left")

        fig.tight_layout()
        if save_fig:
            plt.savefig(save_fig)
            plt.close()

    def plot_train_hist(self, logscale=False):
        plt.plot(self.train_hist['loss'], label="train")
        plt.plot(self.train_hist['val_loss'], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if logscale:
            plt.yscale('log')
        plt.show()




class ViTConvSQAE(baseSQAE):

    def __init__(self, data_qbits, latent_qbits, device, img_dim, kernel_size, stride, DRCs, diff_method="best"):
        super().__init__(data_qbits, latent_qbits, device, diff_method=diff_method)
        self.kernel_size = kernel_size
        self.stride = stride
        self.DRCs = DRCs

        self.data_shape = (img_dim,img_dim)
        self.number_of_kernel_uploads = len(list(range(0, img_dim - kernel_size + 1, stride)))**2
        self.parameters_shape = (DRCs * 2 * self.number_of_kernel_uploads * kernel_size ** 2,)
        self.params_per_layer = self.parameters_shape[0] // DRCs

        self.params = np.random.uniform(size=self.parameters_shape, requires_grad=True)

    #def single_upload(self, params, data, wire):
    #    for i, d in enumerate(data.flatten()):
    #        if i % 3 == 0:
    #            qml.RX(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
    #        if i % 3 == 1:
    #            qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
    #        if i % 3 == 2:
    #            qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)

    def single_upload(self, params, data, wire):
        for i, d in enumerate(data.flatten()):
            if i % 3 == 0:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 1:
                qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 2:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)

    def conv_upload(self, params, img, kernel_size, stride, wires):
        number_of_kernel_uploads = len(list(range(0, img.shape[1]-kernel_size+1, stride))) * len(list(range(0, img.shape[0]-kernel_size+1, stride)))
        params_per_upload = len(params) // number_of_kernel_uploads
        upload_counter = 0
        wire = 0
        for y in range(0, img.shape[1]-kernel_size+1, stride):
            for x in range(0, img.shape[0]-kernel_size+1, stride):
                self.single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[y:y+kernel_size, x:x+kernel_size], wires[wire])
                upload_counter = upload_counter + 1
                wire = wire + 1

    def circular_entanglement(self, wires):
        qml.CNOT(wires=[wires[-1], 0])
        for i in range(len(wires)-1):
            qml.CNOT(wires=[i, i+1])

    def encoder(self, params, data):

        for i in range(self.DRCs):
            self.conv_upload(params[i * self.params_per_layer:(i + 1) * self.params_per_layer], data, self.kernel_size, self.stride, list(range(self.number_of_kernel_uploads)))
            self.circular_entanglement(list(range(self.number_of_kernel_uploads)))
