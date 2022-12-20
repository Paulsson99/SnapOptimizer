from qgrad.qgrad_qutip import fidelity

from jax import jit, grad
import jax.numpy as jnp
from jax.lib import xla_bridge
from jax.example_libraries import optimizers
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path

from SnapOptimizer.snap_gate import SNAPGate


class SNAPGateOptimizer(ABC):

    def __init__(self, N: int, c: float = 0.001) -> None:
        self.N = N
        self.c = c

        # Gates
        self.snap_gate = SNAPGate(self.N)

    @abstractmethod
    def _get_cost_function(self) -> Callable:
        """
        Get the cost function for the optimizer.
        This function should return a function with one argument. 
        This argument may be a tuple with more arguments though
        """

    @abstractmethod
    def _calculate_fidelity(self, alphas: np.ndarray, thetas: np.ndarray) -> float:
        """
        Calculate the current fidelity in the optimization
        """

    @abstractmethod
    def optimize_gates(self, n_gates: int, epochs: int, output_folder: Path = None):
        """
        Owerwrite this function in the subclass to implement extra behaivour
        """
        alphas, thetas, costs, fidelities = self._optimize_gates(n_gates=n_gates, epochs=epochs)
        self._save_results(alphas, thetas, costs, fidelities, output_folder=output_folder)
        return alphas, thetas, costs, fidelities

    def _save_results(self, alphas: np.ndarray, thetas: np.ndarray, costs: np.ndarray, fidelities: np.ndarray, output_folder: Path = None) -> None:
        """
        Save the results
        """
        output_folder = output_folder or Path('')
        output_folder.mkdir(parents=True, exist_ok=False)
        np.savetxt(output_folder / 'alphas.csv', alphas, delimiter=',')
        np.savetxt(output_folder / 'thetas.csv', thetas, delimiter=',')
        np.savetxt(output_folder / 'cost.csv', costs, delimiter=',')
        np.savetxt(output_folder / 'fidelities.csv', fidelities, delimiter=',')

    def _optimize_gates(self, n_gates: int, epochs: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimize a qubit gate built out of displacement and SNAP gates

        Args:
            gate: The gate to optimize (In the standard qubit basis)
            n_gates: Number of gates to use in the optimization
            epochs: Number of epochs to run the optimization

        Returns: (Optimized displacment gates, Optimized SNAP gates)
        """
        print(f"Will use the {xla_bridge.get_backend().platform} to optimize the gate")

        alphas = jnp.array(np.random.uniform(size=n_gates+1))
        thetas = jnp.array(np.random.uniform(size=(n_gates, self.N)))

        params = [alphas, thetas]

        opt_init, opt_update, get_params = optimizers.adam(0.001, 0.9, 0.9)
        opt_state = opt_init([alphas, thetas])

        costs = []
        fidelities = []

        cost_func = self._get_cost_function()

        pbar = tqdm(range(epochs))
        for i in pbar:
            # Update the params
            g = grad(cost_func)(params)
            opt_state = opt_update(i, g, opt_state)
            params = get_params(opt_state)

            # Calculate the new cost
            c = cost_func(params)

            # Update pbar and save cost
            pbar.set_description("Cost {}".format(c))
            costs.append(c)

            # Calculate current fidelity
            fidelities.append(self._calculate_fidelity(*params))

        alphas, thetas = params
        return alphas, thetas, costs, fidelities


class SNAPGateOptimizer2Qubits(SNAPGateOptimizer):

    def __init__(self, encoding: np.ndarray, c: float = 0.001) -> None:
        """
        Args: 
            encoding: Encoding to use. Each row is the fock representation of a logical qubit state
            c: Controls the weigth of theta magnitudes in the cost function
        """
        assert encoding.ndim == 2, f"Encoding has the wrong number of dimensions, must be two, not {encoding.dims}"
        assert encoding.shape[0] == 4, "Encoding must have the first dimension equal to 4 for a two qubit system"

        super().__init__(N=encoding.shape[-1], c=c)

        self.encoding = encoding

        self.L00 = self.encoding[0,:].reshape((self.N, 1))
        self.L01 = self.encoding[1,:].reshape((self.N, 1))
        self.L10 = self.encoding[2,:].reshape((self.N, 1))
        self.L11 = self.encoding[3,:].reshape((self.N, 1))

        # Set up all the basis states
        self.basis_states = []
        qubit_parametrization = [(0, 0), (np.pi/2, 0), (np.pi/2, np.pi/2), (np.pi/2, np.pi), (np.pi/2, 3*np.pi/2), (np.pi, 0)]
        for a1, b1 in qubit_parametrization:
            for a2, b2 in qubit_parametrization:
                q1 = np.array([np.cos(a1/2), np.sin(a1/2)*np.exp(1j*b1)])
                q2 = np.array([np.cos(a2/2), np.sin(a2/2)*np.exp(1j*b2)])
                qq = np.kron(q1, q2)

                self.basis_states.append(qq[0] * self.L00 + qq[1] * self.L01 + qq[2] * self.L10 + qq[3] * self.L11)

        # Transition matrix from fock basis to qubit basis
        self.inverse_encoding = np.linalg.pinv(self.encoding)

        # Placeholder for the implemented gate
        self.gate = None

    def optimize_gates(self, gate: np.ndarray, n_gates: int, epochs: int, output_folder: Path = None):
        # Transorm the gate to the fock basis
        self.gate = self.transform_gate(gate)
        return super().optimize_gates(n_gates=n_gates, epochs=epochs, output_folder=output_folder)

    def _calculate_fidelity(self, alphas: np.ndarray, thetas: np.ndarray) -> float:
        evos = [self.snap_gate(alphas, thetas, state) for state in self.basis_states]
        return np.sum([fidelity(self.gate @ state, evo)[0][0] / len(self.basis_states) for state, evo in zip(self.basis_states, evos)])

    def transform_gate(self, qubit_gate: np.ndarray) -> np.ndarray:
        """
        Transform a gate in the qubit basis into the same gate in the fock basis
        """
        return self.inverse_encoding @ qubit_gate @ self.encoding

    def _save_results(self, alphas: np.ndarray, thetas: np.ndarray, costs: np.ndarray, fidelities: np.ndarray, output_folder: Path = None) -> None:
        super()._save_results(alphas, thetas, costs, fidelities, output_folder)
        # Save the encoding also
        np.savetxt(output_folder / 'encoding.csv', self.encoding, delimiter=',')

    def _get_cost_function(self) -> Callable:

        @jit
        def cost(params):
            """
            Calculates the cost between the target state and 
            the one evolved by the action of three blocks.
            
            Args:
            -----
                params (jnp.array): alpha and theta params of Displace and SNAP respectively
            
            Returns:
            --------
                cost (float): cost at a particular parameter vector
            """
            alphas, thetas = params

            evos = [self.snap_gate(alphas, thetas, state) for state in self.basis_states]

            fidelities = jnp.array([fidelity(self.gate @ state, evo)[0][0] / 36 for state, evo in zip(self.basis_states, evos)])
            return 1 - jnp.sum(fidelities) + self.c * jnp.sum(jnp.abs(thetas)) / thetas.size

        return cost


class SNAPGateOptimizerStatePreparation(SNAPGateOptimizer):

    def __init__(self, N: int, c: float = 0.001) -> None:
        super().__init__(N, c)
        # Placeholder for the state to prepere
        self.state = None
        self.ground_state = jnp.zeros((self.N, 1))
        self.ground_state = self.ground_state.at[0].set(1)

    def _calculate_fidelity(self, alphas: np.ndarray, thetas: np.ndarray) -> float:
        final_state = self.snap_gate(alphas, thetas, self.ground_state)
        fval = fidelity(final_state, self.state)[0][0]
        return fval

    def _save_results(self, alphas: np.ndarray, thetas: np.ndarray, costs: np.ndarray, fidelities: np.ndarray, output_folder: Path = None) -> None:
        super()._save_results(alphas, thetas, costs, fidelities, output_folder)
        # Save the state also
        np.savetxt(output_folder / 'prepered_state.csv', self.state, delimiter=',')

    def optimize_gates(self, state: np.ndarray, n_gates: int, epochs: int, output_folder: Path = None):
        assert state.shape == (self.N, 1), f"The requested state has the wrong shape, must be ({self.N}, 1), not {state.shape}"
        self.state = state
        return super().optimize_gates(n_gates, epochs, output_folder)

    def _get_cost_function(self) -> Callable:

        @jit
        def cost(params):
            """
            Calculates the cost between the target state and 
            the one evolved by the action of three blocks.
            
            Args:
            -----
                params (jnp.array): alpha and theta params of Displace and SNAP respectively
            
            Returns:
            --------
                cost (float): cost at a particular parameter vector
            """
            alphas, thetas = params
            evo = self.snap_gate(alphas, thetas, self.ground_state)
            return 1 - fidelity(self.state, evo)[0][0] + self.c * jnp.sum(jnp.abs(thetas)) / thetas.size

        return cost
