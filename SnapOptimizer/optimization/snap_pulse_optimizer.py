import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime

import SnapOptimizer.paths as local_paths

try:
    from qctrl import Qctrl
    from qctrlvisualizer import plot_controls
    qctrl = Qctrl()
except ModuleNotFoundError:
    print("Module qctrl not installed. You wont be able to optimize any pusles")


class SNAPPulseOptimizer:
    """
    Optimizer for the pulses for a SNAP gate
    """

    def __init__(self, 
        dim_t: int,                 # Number of transmon levels
        dim_c: int,                 # Number of cavity levels
        delta: float,               # Hz drive spacing
        xi: float,                  # Dispersive shift
        xip: float,                 # Dispersive shift???
        K: float,                   # Kerr coefficient
        alpha: float,               # Trasmon qubit anharmonicity (not used)
        wt: float,                  # Transmon qubit frequency
        wc: float,                  # Cavity frequency
        max_rabi_rate: float,       # Maximum rabi frequency
        cutoff_frequency: float,    # Cutoff frequenzy
        num_drives: int,            # The number of drives in the system
    ):
        self.dim_t = dim_t
        self.dim_c = dim_c
        self.delta = delta
        self.xi = xi
        self.xip = xip
        self.K = K
        self.alpha = alpha
        self.wt = wt
        self.wc = wc
        self.max_rabi_rate = max_rabi_rate
        self.cutoff_frequency = cutoff_frequency
        self.num_drives = num_drives

    def hamiltonian(self):
        """
        Get the system Hamiltonian
        """
        # Annihilation and creation operators for the transmon and cavity
        a = np.zeros((self.dim_c, self.dim_c))
        for ind in range(len(a)-1):
            a[ind][ind+1] = np.sqrt(ind+1)
        b = np.zeros((self.dim_t, self.dim_t))
        for ind in range(len(b)-1):
            b[ind][ind+1] = np.sqrt(ind+1)
        
        Hosc = self.wc*np.kron(np.eye(self.dim_t), np.dot(a.T,a)) + self.K/2*np.kron(np.eye(self.dim_t), np.dot(np.dot(a.T,a.T),np.dot(a,a)))
        Htrans = self.wt*np.kron(np.dot(b.T,b), np.eye(self.dim_c)) + self.alpha/2*np.kron(np.dot(np.dot(b.T,b.T),np.dot(b,b)), np.eye(self.dim_c))
        Hint = self.xi*np.kron(np.dot(b.T,b), np.dot(a.T,a)) + self.xip/2*np.kron(np.dot(b.T,b), np.dot(np.dot(a.T,a.T),np.dot(a,a)))
        control_T = np.kron(b.T, np.eye(self.dim_c))

        return Hosc, Htrans, Hint, control_T

    def run_optimization(
        self, 
        target: np.ndarray,                 # The SNAP gate to realize with pulses
        gate_duration: float,               # How long the gate pulse should be 
        number_of_optimizer_vars: int = 64, # Number of degrees of freedom per pulse for the optimizer
        number_of_segments: int = 700,      # Number of pulse segments
        optimization_count: int = 10        # Number of optimization runs
    ):  
        # triangular envelope function to not have spikes at the beginning and end
        time_points= np.linspace(-1.0, 1.0, number_of_optimizer_vars + 2)[1:-1]
        envelope_function = 1 - np.abs(time_points)

        # Get the different components of the Hamiltonian
        Hosc, Htrans, Hint, control_T = self.hamiltonian()

        graph = qctrl.create_graph()
    
        # Create I and Q variables
        # set up optimizer variables for the i component of the transmon qubit drive
        drive_iT_vars = graph.optimization_variable(
            count=number_of_optimizer_vars,
            lower_bound=-self.max_rabi_rate,
            upper_bound=self.max_rabi_rate)
        # set up optimizer variables for the q component of the transmon qubit drive
        drive_qT_vars = graph.optimization_variable(
            count=number_of_optimizer_vars,
            lower_bound=-self.max_rabi_rate,
            upper_bound=self.max_rabi_rate)     

        # Anchor ends to zero with amplitude rise/fall envelope
        anchored_drive_iT_vars = drive_iT_vars * envelope_function
        anchored_drive_qT_vars = drive_qT_vars * envelope_function        

        # Create I & Q signals: build the pulse signal with segments based on the optimizer variables 
        drive_iT_raw = graph.pwc_signal(
            values=anchored_drive_iT_vars,
            duration=gate_duration)
        drive_qT_raw = graph.pwc_signal(
            values= anchored_drive_qT_vars,
            duration=gate_duration)     

        # set up a sinc filter to bandlimit the pulses 
        sinc_filter = graph.sinc_convolution_kernel(self.cutoff_frequency)

        # apply the sinc filter to the raw pulse to remove higher frequencies
        drive_iT_filtered = graph.convolve_pwc(
            pwc=drive_iT_raw,
            kernel=sinc_filter)
        drive_qT_filtered = graph.convolve_pwc(
            pwc=drive_qT_raw,
            kernel=sinc_filter)        

        # re-discretize the filtered drive into the desired number of segments
        drive_iT_signal = graph.discretize_stf(
            stf=drive_iT_filtered,
            duration=gate_duration,
            segment_count=number_of_segments)       
        drive_qT_signal = graph.discretize_stf(
            stf=drive_qT_filtered,
            duration=gate_duration,
            segment_count=number_of_segments)


        # combine the i and q components
        combined_values = graph.complex_value(drive_iT_signal.values, 
                                                        drive_qT_signal.values)

        # Create envelope function with a complex exponential term
        driveT_signal=[]
        time_points= np.linspace(0, gate_duration, number_of_segments)
        for i in range(self.num_drives):
            exponential_function = np.exp(1j*2*np.pi*i*self.delta*time_points)
            rotated_values = combined_values * exponential_function
            driveT_signal.append(graph.pwc_signal(
                values= rotated_values,
                duration=gate_duration,
                name="alpha_t"+str(i)))


        # build the system Hamiltonian terms
        H_osc = graph.constant_pwc_operator(
            operator=Hosc,
            duration=gate_duration)        
        H_trans = graph.constant_pwc_operator(
            operator=Htrans,
            duration=gate_duration)    
        H_int = graph.constant_pwc_operator(
            operator=Hint,
            duration=gate_duration) 

        H_drive_T=[]
        for i in range(self.num_drives):
            H_drive_T.append(graph.hermitian_part(graph.pwc_operator(
                signal= driveT_signal[i],
                operator=control_T)))


        # Construct the total Hamiltonian
        hamiltonian = graph.pwc_sum([H_osc, H_trans, H_int, *H_drive_T])
        noise_list=[]

        # gate infidelity cost
        cost = graph.infidelity_pwc(
            hamiltonian=hamiltonian,
            target=graph.target(target),
            noise_operators=noise_list,
            name='cost')

        node_names=[]
        for i in range(self.num_drives):
            node_names.append("alpha_t"+str(i))
        # run optimization
        return qctrl.functions.calculate_optimization(
            graph=graph,
            cost_node_name='cost',
            output_node_names=node_names,
            optimization_count=optimization_count)

    def simulate_unitaries(self, controls, sample_points):
        drive_control_segments = []
        for i in range(self.num_drives):
            drive_control_segments.append(controls["alpha_t"+str(i)])
        
        gate_duration = np.sum([s['duration'] for s in drive_control_segments[0]])
        sample_times = np.linspace(0,gate_duration, sample_points)
        
        # Get the different components of the Hamiltonian
        Hosc, Htrans, Hint, control_T = self.hamiltonian()
        
        H_drive_term_list = []
        # Set up Hamiltonian terms
        for i in range(self.num_drives):
            H_drive_term_list.append(qctrl.types.coherent_simulation.Drive(
                control=[qctrl.types.ComplexSegmentInput(duration=s['duration'], value=s['value'])
                    for s in drive_control_segments[i]],
                    operator=control_T/2))
        
        H_osc = qctrl.types.coherent_simulation.Drift(operator=Hosc)
        H_trans = qctrl.types.coherent_simulation.Drift(operator=Htrans)
        H_int = qctrl.types.colored_noise_simulation.Drift(operator=Hint)
        
        
        # Run a simulation with optimized values
        simulation_result =  qctrl.functions.calculate_coherent_simulation(
            duration=gate_duration, 
            sample_times=sample_times,
            drives=H_drive_term_list, 
            drifts=[H_osc,H_trans,H_int]
        )   
        
        return simulation_result

    def optimize_SNAP_pulse(self, 
        thetas: np.ndarray,                 # The SNAP gate to realize with pulses
        gate_duration: float,               # How long the gate pulse should be 
        number_of_optimizer_vars: int = 64, # Number of degrees of freedom per pulse for the optimizer
        number_of_segments: int = 700,      # Number of pulse segments
        optimization_count: int = 10        # Number of optimization runs
    ) -> None:
        """
        Optimize pulses for a SNAP gate
        """
        # target operation for the cavity only
        cav_target_operation = scipy.linalg.expm(np.diag(1j*thetas).astype(complex))

        # target operatrion for the full system
        full_target_operation = np.kron(np.eye(self.dim_t), cav_target_operation).astype(complex)
        # work in the subspace of |0> qubit state 
        cavity_subspace_projector = np.diag(np.kron(np.array([1.,0.]), np.ones(self.dim_c)))
        # net system target
        subspace_target = np.matmul(full_target_operation, cavity_subspace_projector)

        return self.run_optimization(
            target=subspace_target,
            gate_duration=gate_duration,
            number_of_optimizer_vars=number_of_optimizer_vars,
            number_of_segments=number_of_segments,
            optimization_count=optimization_count
        )

    def optimize_gate_pulses(self, thetas: np.ndarray, alphas: np.ndarray, gate_times: float, output_folder: Path = None) -> None:
        """
        Optimize pulses for the SNAP gates and save all parameters to a file
        """
        results = [self.optimize_SNAP_pulse(theta, gate_times) for theta in thetas]

        # Save the results
        self.save_pulses(results, thetas, alphas, output_folder=output_folder)
        self.plot_results(results)

    def plot_results(self, results):
        str1 = r"$\alpha_{"
        str2 = "alpha_t"

        for i in range(self.num_drives):
            for j, result in enumerate(results):
                str11 = str1 + f"{j+1}{i+1}" + r"}$"
                str22 = str2 + str(i)
                plot_controls(controls={str11: result.output[str22]}, polar=False)
        plt.show()


    def save_pulses(self, results, thetas: np.ndarray, alphas: np.ndarray, output_folder: Path = None) -> None:
        """
        Save an optimized pulse
        """
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        output_folder = output_folder or local_paths.pulses(f"pulses_{timestamp}")
        output_folder.mkdir(parents=True, exist_ok=False)

        # Save the data we have generated the pulses for
        np.savetxt(output_folder / "thetas.csv", thetas, delimiter=',')
        np.savetxt(output_folder / "alphas.csv", alphas, delimiter=',')

        gate_dict = {}
        sample_rate = 2e9 # Hz, of waveform generator    

        # Save SNAP pulses
        for i, result in enumerate(results):
            base_filename = f"SNAP{i+1}"
            # Save raw optimization data
            raw_data_folder = output_folder / "raw_data"
            raw_data_folder.mkdir(parents=False, exist_ok=True)
            with open(output_folder / "raw_data" / f"{base_filename}", "wb") as f:
                pickle.dump(result.output, f)
            
            ### LOOP THROUGH PULSES ###        
            string = "alpha_t"
            for j in range(self.num_drives):
                string_idx = string + str(j)
                filename = base_filename + '_pulse' + str(j+1)
                
                pulses = result.output[string_idx]
                duration =  pulses[0]['duration'] # duration of every entry in pulses (in s)
                # repeat each entry in array "pulses" "repeat" number of times to get an array with 4 points for every ns
                repeat = sample_rate * duration

                I = []
                Q = []
                for entry in pulses:
                    for _ in np.arange(repeat):
                        amplitude = np.abs(entry['value']) # scale amplitude
                        phase = np.angle(entry['value']) # in radians
                        I.append(amplitude*np.cos(phase))
                        Q.append(amplitude*np.sin(phase))

                max_rabi_rate = max(max(np.abs(I)),max(np.abs(Q)))
                I = np.array(I) / max_rabi_rate
                Q = np.array(Q) / max_rabi_rate

                pulses_folder = output_folder / "pulses"
                pulses_folder.mkdir(parents=False, exist_ok=True)
                np.savetxt(pulses_folder / f"{filename}_I.csv" , I, delimiter=',')
                np.savetxt(pulses_folder / f"{filename}_Q.csv" , Q, delimiter=',')

                # Add to large file
                gate_dict[f"I{i+1}_{j+1}"] = I
                gate_dict[f"Q{i+1}_{j+1}"] = Q
            
        # Save qubit pulses ??? 
        # FOR AMPLITUDE SCALING
        # 2*pi*y = ax + b
        # y = rabi rate over 2pi (Hz)
        # x = instrument units
        # aI = 49907948 * 2 *np.pi
        # bI = 50643 * 2 *np.pi

        # aQ = 49907948 * 2 *np.pi
        # bQ = 50643 * 2 *np.pi

        # # x corresponding to 2*pi*y = max_rabi_rate
        # qI = (max_rabi_rate - bI)/aI
        # qQ = (max_rabi_rate - bQ)/aQ
        # x = np.array([xI,xQ])

        # Save displacemnt gates
        a_c = 6.333
        b_c = 0
        
        gate_dict['D1'] = (alphas[0] - b_c) / a_c
        for i, (prev_alpha, next_alpha) in enumerate(zip(alphas[:-1], alphas[1:])):
            gate_dict[f"D{i+2}"] = (next_alpha - prev_alpha - b_c) / a_c
        gate_dict[f"D{len(alphas) + 1}"] = (-alphas[-1] - b_c) / a_c

        np.savez(output_folder / f"{filename}.npz", **gate_dict)


if __name__ == '__main__':
    op = SNAPPulseOptimizer(
        dim_t=2,
        dim_c=12,
        delta = -2.574749e6,
        xi = -2550000.0,
        xip = -2*np.pi* 0.013763e6,
        K = 2*np.pi* 6000,
        alpha = 0,
        wt = 5462837912.075907,
        wc = 3550318353.300307,
        max_rabi_rate = 2*np.pi* 2e6,
        cutoff_frequency = 2*np.pi* 30e6,
        num_drives = 1
    )

    thetas = np.array([[-3.298317576835431697e-01,2.304001683119982491e-01,2.485572922212311298e-01,1.942965018956552159e+00,-3.408790230523779385e-01,9.748353087601625555e-01,-5.757418408810377475e-01,1.408791349696820516e+00,-6.260406453588069384e-05,4.443751687284836318e-05,1.155056773157905030e-04,-1.095428642827285585e-04]])
    alphas = np.array([-0.6, 0.6])

    op.optimize_gate_pulses(
        thetas=thetas,
        alphas=alphas,
        gate_times=0.7e-6
    )

