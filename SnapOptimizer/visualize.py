from qgrad.qgrad_qutip import dag, Displace
import jax.numpy as jnp

from qutip.visualization import plot_wigner
from qutip import Qobj
from qutip.wigner import wigner

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from SnapOptimizer.snap_gate import _snap, SNAPGate


def show_state(state, fig=None, ax=None, title=None):
    """Shows the Hinton plot and Wigner function for the state"""
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 5))
    dm = Qobj(np.array(jnp.dot(state, dag(state))))
    plot_wigner(dm, ax=ax)

    if title is not None:
        ax.set_title(title)

    return fig, ax


class SnapPulseVisulizer:

    def __init__(self, N: int) -> None:
        self.N = N
        self.snap = _snap(N)
        self.displace = Displace(N)
        self.full_snap = SNAPGate(N)

    def visualize(self, initial_state: np.ndarray, thetas: np.ndarray, alphas: np.ndarray) -> None:
        gates, Np = thetas.shape
        assert Np == self.N, f"Each snap gate must match the size of the concatenated Hilbert space"
        assert alphas.size == gates + 1, f"Alphas and thetas must have the same length"
        assert initial_state.shape == (self.N, 1), f"Inital shape must have shape ({self.N}, 1), not {initial_state.shape}"

        fig, (ax1, ax2) = plt.subplots(1, 2)

        evolved_state = self.full_snap(alphas, thetas, initial_state)

        self.plot_wigner(evolved_state, ax=ax1)
        self.plot_wigner(initial_state, ax=ax2)


    def visualize_process(self, initial_state: np.ndarray, thetas: np.ndarray, alphas: np.ndarray, figsize: tuple[int, int] = None, alpha_max: float = 4):
        gates, Np = thetas.shape
        assert Np == self.N, f"Each snap gate must match the size of the concatenated Hilbert space"
        assert alphas.size == gates + 1, f"Alphas and thetas must have the same length"
        assert initial_state.shape == (self.N, 1), f"Inital shape must have shape ({self.N}, 1), not {initial_state.shape}"

        total_figs = 2 * gates + 2
        if figsize is None:
            rows = int(np.floor(np.sqrt(total_figs)))
            cols = rows
            while cols * rows < total_figs:
                cols += 1
        else:
            rows, cols = figsize

        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        flat_axs = axs.flatten()

        state = initial_state
        
        SnapPulseVisulizer.plot_wigner(state, flat_axs[0], alpha_max=alpha_max)
        flat_axs[0].set_title(r"|$\Psi_0\rangle$")
        flat_axs[0].set_aspect('equal', 'box')

        fig_index = 1
        for i in range(gates):
            state = np.dot(self.displace(alphas[i]), state)
            SnapPulseVisulizer.plot_wigner(state, flat_axs[fig_index], alpha_max=alpha_max)
            flat_axs[fig_index].set_title(fr"$|\Psi_{{{fig_index}}}\rangle=D_{{{i+1}}}|\Psi_{{{fig_index-1}}}\rangle$")
            flat_axs[fig_index].set_aspect('equal', 'box')
            fig_index += 1
            state = np.dot(self.snap(thetas[i,:]), state)
            SnapPulseVisulizer.plot_wigner(state, flat_axs[fig_index], alpha_max=alpha_max)
            flat_axs[fig_index].set_title(fr"$|\Psi_{{{fig_index}}}\rangle=S_{{{i+1}}}|\Psi_{{{fig_index-1}}}\rangle$")
            flat_axs[fig_index].set_aspect('equal', 'box')
            fig_index += 1
        state = np.dot(self.displace(alphas[-1]), state)
        cmap = SnapPulseVisulizer.plot_wigner(state, flat_axs[fig_index], alpha_max=alpha_max, get_cmap=True)
        flat_axs[fig_index].set_title(fr"$|\Psi_{{{total_figs-1}}}\rangle=D_{{{gates+1}}}|\Psi_{{{total_figs-2}}}\rangle$")
        flat_axs[fig_index].set_aspect('equal', 'box')

        # Make a colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(cmap, cax=cbar_ax)
        # fig.tight_layout()

    @staticmethod
    def plot_wigner(state: np.ndarray, ax, alpha_max: float = 4, get_cmap: bool = False) -> None:
        state = np.copy(state)
        dm = Qobj(np.dot(state, dag(state)))
        plot_wigner(dm, ax=ax, alpha_max=alpha_max)

        if get_cmap:
            cmap = cm.get_cmap('RdBu')
            # Normilize cmap
            xvec = np.linspace(-alpha_max, alpha_max, 200)
            W0 = wigner(dm, xvec, xvec, method='clenshaw')
            W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
            wlim = abs(W).max()
            norm = Normalize(-wlim, wlim)
            return cm.ScalarMappable(norm=norm, cmap=cmap)
        

if __name__ == '__main__':
    thetas = np.array(
        [
            [-3.298317576835431697e-01,2.304001683119982491e-01,2.485572922212311298e-01,1.942965018956552159e+00,-3.408790230523779385e-01,9.748353087601625555e-01,-5.757418408810377475e-01,1.408791349696820516e+00,-6.260406453588069384e-05,4.443751687284836318e-05,1.155056773157905030e-04,-1.095428642827285585e-04], 
            [-3.258037981533372651e-01,2.859562619437964415e+00,7.988405687052090309e-01,-1.509073407650175433e-02,1.241539642779063146e-01,-1.535152661258065709e-01,-4.021308996377573886e-01,1.409852064198948485e+00,4.562056664828753050e-01,1.219405168689316758e-05,-2.336930860296993016e-04,-2.610987514928118534e-04], 
            [2.076894241295585086e+00,1.307117303467628444e+00,-7.606018165454099256e-01,-8.209045813948765424e-01,-8.258029853499746498e-01,-5.651095334184479402e-01,1.413889858077303219e-01,2.253602489976073431e-01,-5.026509555445662220e-02,-3.538932677929261805e-01,-5.230202817254341330e-01,-3.324295615911743162e-01], 
            [-1.254387515422792365e+00,1.160782916664712561e+00,1.111302675185738220e+00,7.541707509030551870e-01,8.963233799737869989e-02,1.557991037250436639e+00,-4.342204943056054978e-01,1.974182009658780546e+00,-1.684428179375879742e+00,1.820947241576538422e+00,1.851252970188069160e-04,-1.268718400351971365e-01], 
            [1.237902780653018064e+00,5.317261543100122029e-02,-1.191300743761011693e+00,1.803906358615195549e+00,-8.021391040274414852e-01,2.222796869538938846e+00,-1.027137858365132050e+00,1.795291473115562253e+00,-1.164109616317685170e+00,2.632231197540158441e+00,3.156018285105894965e-05,2.867772815003984399e-01]
        ]
    )
    alphas = np.array([-2.310750867562696709e-01, -3.636820008595296749e-01, -3.197928404817585446e-01, 7.239389716336550595e-01, 4.279227758317115105e-01, 6.501711003082247808e-01])

    initial_state = np.zeros((12, 1))
    initial_state[0] = 1

    vis = SnapPulseVisulizer(12)
    vis.visualize_process(initial_state, thetas, alphas)

    plt.show()
