from qgrad.qgrad_qutip import to_dm, Displace
import jax.numpy as jnp
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True)

from jax import jit


def SNAPGate(N: int):
    """
    Applies T blocks of operators to the initial state.
    
    Args:
    ------
        initial (jnp.ndarray): initial state to apply blocks on (ket |0> in our case)
        T (int): number of blocks to apply
        hilbert_size (int): Size of the Hilbert space
        params (jnp.ndarray): parameter array of alphas and thethas of size :math: `T * hilbert_size + T`, 
                wherein the first T parameters are alphas and the rest are T hilbert_size-dimensional vectors 
                representing corresponding theta vectors.

    Returns:
    -----------
        evolved (jnp.array): (hilbert_size, 1) dimensional array representing the action of the T
                blocks on the vacuum state
    """
    displace = Displace(N)
    snap = _snap(N)

    def inner(alphas, thetas, initial):
        x = initial
        for t in range(thetas.shape[0]):
            x = jnp.dot(displace(alphas[t]), x)
            x = jnp.dot(snap(thetas[t,:]), x)
        x = jnp.dot(displace(alphas[-1]), x)
        return x
    
    return inner


def _basis(N, n):
    """Generates the vector representation of a Fock state.
    
    Args:
        N (int): Number of Fock states in the Hilbert space
        n (int): Number state (defaults to vacuum state, n = 0)

    Returns:
        :obj:`jnp.ndarray`: Number state :math:`|n\rangle`

    """
    zeros = jnp.zeros((N, 1), dtype=jnp.complex64)  # column of zeros
    return zeros.at[n].set(1.0)


def _snap(N):

    @jit
    def snap(thetas):
        """
        SNAP gate matrix.
        
        Args:
        -----
            N (int): Hilbert space cuttoff
            thetas (:obj:`jnp.ndarray`): A vector of theta values to apply SNAP operation
        
        Returns:
        --------
            op (:obj:`jnp.ndarray`): matrix representing the SNAP gate
        """
        op = jnp.zeros(N)
        for i, theta in enumerate(thetas):
            op += jnp.exp(1j * theta) * to_dm(_basis(N, i))
        return op
    
    return snap
