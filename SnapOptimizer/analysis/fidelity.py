from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np

from SnapOptimizer.paths import data


def plot_folder_against_epochs(folder: Path, file_pattern: str, recursive: bool = True, fig = None, ax = None, legend=True) -> plt.Figure:
    """
    Plot all fidility data found in a folder
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 8))
    folder = Path(folder)
    for fidelities_data in glob.glob('**/' + file_pattern, root_dir=folder, recursive=recursive):
        fidelities = np.loadtxt(folder / fidelities_data, delimiter=',')
        label = Path(fidelities_data).parent.name.replace('-', ' ')
        ax.plot(range(len(fidelities)), fidelities, label=label)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Fidelities")

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()


def plot_folder_against_fock_or_gates(folder: Path, file_pattern: str, plot_against: str, recursive: bool = True, fig = None, ax = None, error_plot: bool = False, legend: bool = True) -> plt.Figure:
    """
    Plot all fidility data found in a folder
    """
    legend_map = {
        'cnot1': 'CNOT with qubit 1 as control',
        'cnot2': 'CNOT with qubit 2 as control',
        'hadamard1': 'Hadamard on qubit 1',
        'hadamard2': 'Hadamard on qubit 2',
    }
    if plot_against.lower() == 'gates':
        extract_index = 1
        x_label = "Number of SNAP gates"
    elif plot_against.lower() == 'fock':
        extract_index = 3
        x_label = "Cutoff for the Hilbert space"
    else:
        raise ValueError(f"{plot_against} is not a valid argument")
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 6))
    folder = Path(folder)
    extracted_data = {}
    for raw_data in glob.glob('**/' + file_pattern, root_dir=folder, recursive=recursive):
        fidelities = np.loadtxt(folder / raw_data, delimiter=',')
        experiment_details = Path(raw_data).parent.name.split('-')
        x_data = int(experiment_details[extract_index])
        label = experiment_details[0].lower()
        if label not in extracted_data:
            extracted_data[label] = {}
        extracted_data[label][x_data] = extracted_data[label].get(x_data, []) + [np.max(fidelities)]

    y_fit_data = []
    x_fit_data = []

    for label, data in extracted_data.items():
        x_data = []
        y_data = []
        err_data = []
        xx = []
        yy = []
        for key, fidelities in data.items():
            x_data.append(key)
            fidelities = np.array(fidelities)
            if error_plot:
                fidelities = 1 - fidelities
            y_data.append(np.mean(fidelities))
            err_data.append(np.std(fidelities))
            xx.extend([key]*len(fidelities))
            yy.extend(fidelities)
        x_data, y_data, err_data = zip(*sorted(zip(x_data, y_data, err_data)))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        err_data = np.array(err_data)
        ax.plot(x_data, y_data, label=legend_map[label])
        if not error_plot:
            ax.fill_between(x_data, y_data - err_data, y_data + err_data, alpha=0.2)
        y_fit_data.extend(y_data)
        x_fit_data.extend(x_data)
    ax.set_xlabel(x_label, fontsize=14)

    if error_plot:
        ax.set_yscale('log')
        ax.set_ylabel("Error rate", fontsize=14)

        # Fit line
        poly = np.polyfit(x_fit_data, np.log(y_fit_data), deg=1)
        ax.plot(x_data, np.exp(np.polyval(poly, x_data)), '--', label=f'Fitted line with slope {poly[0]:.3f}')

        if legend:
            ax.legend(loc="upper right", fontsize=14)

    else:
        ax.set_ylabel("Average fidelity", fontsize=14)
        if legend:
            ax.legend(loc="lower right", fontsize=14)



if __name__ == '__main__':
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot_folder_against_epochs(data() / 'nfock', file_pattern='*16-fock*/fidelities.csv')
    # plot_folder_against_epochs(data() / 'NSnaps_nfock_data' / 'NSnaps', file_pattern='cost.csv', ax=ax2, fig=fig, legend=True)
    # plot_folder_against_epochs(data() / 'NSnap', file_pattern='*cnot2/cnot2-3-*/fidelities.csv')
    # plot_folder_against_epochs(data() / 'NSnap', file_pattern='*cnot2/cnot2-4-*/fidelities.csv')

    # Nice plot
    plot_folder_against_fock_or_gates(data() / 'nfock', file_pattern='fidelities.csv', plot_against='fock', legend=False)
    plot_folder_against_fock_or_gates(data() / 'NSnap', file_pattern='fidelities.csv', plot_against='gates')
    plot_folder_against_fock_or_gates(data() / 'NSnap', file_pattern='fidelities.csv', plot_against='gates', error_plot=True)
    plt.show()

