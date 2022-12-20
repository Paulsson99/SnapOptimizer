import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SnapOptimizer.optimization.snap_optimizer import SNAPGateOptimizer2Qubits, SNAPGateOptimizerStatePreparation
from SnapOptimizer.optimization.snap_pulse_optimizer import SNAPPulseOptimizer
import SnapOptimizer.qubit_gates as qubit_gates
from SnapOptimizer.visualize import show_state
from SnapOptimizer.encodings import Encoding
import SnapOptimizer.paths as local_paths


def optimize_SNAP_gates(
    encoding: Encoding, gates: list[str], n_gates: list[int], Ns: list[int], epochs: int, output_folder: Path, 
    show_figure: bool = False, c: float = 0.001, averages: int = 1
):
    """
    Automation for the optimization of SNAP gates
    
    Args:
        encoding: The encoding to use
        gates: The gates to optimize
        n_gates: How many SNAP gates to use to replicate the gate
        Ns: Size of the Hilbert space in the fock basis
        epochs: How many epochs to run the optimization for
        output_folder: folder to save the results to
        show_figure: If True the figures will pop up on screen when they are drawn. Otherwise they will only be saved
        c: Parameter to control the weight of the thetas in the optimization
        averages: Run the same optimization multiple times
    """
    fidelity_fig, fidelity_ax = plt.subplots(1, figsize=(8, 8))

    for N in Ns:
        code = encoding.get_encoding(N)
        snap_op = SNAPGateOptimizer2Qubits(code, c=c)

        for gate_name in gates:
            gate = getattr(qubit_gates, gate_name.upper(), None)
            if gate is None:
                print(f"The gate {gate_name} is not defined. Check your spelling and try again")
                continue

            for n in n_gates:
                for i in range(averages):
                    if averages == 1:
                        save_to = output_folder / f"{gate_name}-{n}-gates-{N}-fockstates"
                    else:
                        save_to = output_folder / f"{gate_name}-{n}-gates-{N}-fockstates_{i+1}"
                    alphas, thetas, _, fidelities = snap_op.optimize_gates(gate, n, epochs, output_folder=save_to)

                    # Generate figures
                    fidelity_ax.plot(range(epochs), fidelities, label=f"{gate_name} {n} gates {N} fock")

                    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

                    for i, qubit_state in enumerate(['L00', 'L01', 'L10', 'L11']):
                        logic_state = getattr(snap_op, qubit_state)
                        
                        evolved_state = snap_op.snap_gate(alphas, thetas, logic_state)
                        expected_state = snap_op.transform_gate(gate) @ logic_state

                        show_state(evolved_state, ax=axs[0][i], title=f"Gate on {qubit_state}")
                        show_state(expected_state, ax=axs[1][i], title="")

                    fig.savefig(save_to / 'wigner_plots.png', dpi=150)
                    if show_figure:
                        plt.show()
                    plt.close(fig)
    
    # Figure over the fidelities
    fidelity_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fidelity_fig.tight_layout()
    fidelity_ax.set_ylabel("Fidelity")
    fidelity_ax.set_xlabel("Epoch")
    fidelity_ax.set_ylim([0, 1.1])
    filename = output_folder / 'fidelity_plot.png'

    # Handle the possibility of the image already existing. (Multiple processes running the optimization)
    counter = 1
    while filename.exists():
        filename = output_folder / f'fidelity_plot_{counter}.png'
        counter += 1
    
    fidelity_fig.savefig(filename, dpi=150)
    if show_figure:
        plt.show()


def optimize_SNAP_gates_for_state_preparation(state: np.ndarray, n: int, N: int, epochs: int = 2000, output_folder: Path = None):
    op = SNAPGateOptimizerStatePreparation(N=N)
    ground_state = np.zeros((N, 1))
    ground_state[0, 0] = 1
    alphas, thetas, cost, fidelities = op.optimize_gates(state, n_gates=n, epochs=epochs, output_folder=output_folder)

    _, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax = plt.subplots(1)

    evolved_state = op.snap_gate(alphas, thetas, ground_state)

    show_state(evolved_state, ax=ax1, title=f"Evolved ground state")
    show_state(state, ax=ax2, title="Target state")

    ax.plot(range(len(fidelities)), fidelities)

    plt.show()


def optimize_SNAP_pulses(alphas: np.ndarray, thetas: np.ndarray, output_folder: Path = None):
    op = SNAPPulseOptimizer(
        dim_c = thetas.shape[-1],
        dim_t = 2,
        delta = -2.574749e6,
        xi = -2*np.pi* 2.217306e6,
        xip = -2*np.pi* 0.013763e6,
        K = -2*np.pi* 0.002692e6,
        alpha = 0,
        wt = 0,
        wc = 0,
        max_rabi_rate = 2*np.pi* 20e6,
        cutoff_frequency = 2*np.pi* 30e6,
        num_drives = 1
    )
    op.optimize_gate_pulses(thetas, alphas, 0.7e-6, output_folder=output_folder)


if __name__ == '__main__':
    # input_folder = local_paths.data('test_state_20')
    # output_folder = local_paths.pulses('test_state_20')
    # thetas = np.loadtxt(input_folder / 'thetas.csv', delimiter=',')
    # alphas = np.loadtxt(input_folder / 'alphas.csv', delimiter=',')
    # optimize_SNAP_pulses(alphas=alphas, thetas=thetas)

    output_folder = local_paths.data('fock_1')
    N = 12
    n = 1
    epochs = 3000
    fock1 = np.zeros((12, 1))
    fock1[1,:] = 1
    print(fock1)
    optimize_SNAP_gates_for_state_preparation(fock1, n, N, epochs=epochs, output_folder=output_folder)
