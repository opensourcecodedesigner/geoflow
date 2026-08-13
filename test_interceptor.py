import torch
import matplotlib.pyplot as plt

class ReflexInterceptor:
    def __init__(self, tau_upper=0.05, tau_lower=0.01, T_min=0.1, alpha_beta=0.2):
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower
        self.T_min = T_min 
        self.alpha_beta = alpha_beta 
        
        self.gate_state = 0.0
        # Initialized to -2*T_min so condition holds at t=0.0
        self.last_switch_time = -2.0 * self.T_min
        self.current_alpha = 0.0

    def step(self, nominal_twist, corrective_twist, geometric_error, current_time):
        r"""
        Intercepts the control flow at every timestep.

        Args:
            nominal_twist (Tensor): The 6-DoF command from the slow visual planner (\xi_{flow}).
            corrective_twist (Tensor): RELATIVE 6-DoF twist delta (\delta\xi_t), NOT an absolute target.
            geometric_error (float/Tensor): The Lie-algebraic prediction error E_t.
            current_time (float): The current system time in seconds.

        Returns:
            final_twist (Tensor): The safely blended output command (\xi_{cmd}).
            gate_state (float): Discrete gate state (0.0 = nominal, 1.0 = reflex active).
            current_alpha (float): Low-pass filtered blending weight \alpha(t) \in [0, 1].
        """
        time_since_switch = current_time - self.last_switch_time
        dwell_satisfied = time_since_switch > self.T_min
        
        if geometric_error > self.tau_upper and dwell_satisfied and self.gate_state == 0.0:
            self.gate_state = 1.0
            self.last_switch_time = current_time
        elif geometric_error < self.tau_lower and dwell_satisfied and self.gate_state == 1.0:
            self.gate_state = 0.0
            self.last_switch_time = current_time
            
        self.current_alpha = (1.0 - self.alpha_beta) * self.current_alpha + (self.alpha_beta * self.gate_state)
        final_twist = nominal_twist + (self.current_alpha * corrective_twist)
        
        return final_twist, self.gate_state, self.current_alpha


def run_interceptor_simulation():
    interceptor = ReflexInterceptor(tau_upper=0.05, tau_lower=0.01, T_min=0.1)
    
    dt = 0.005 # 200 Hz loop
    total_steps = 200
    
    times, errors, gate_states, alphas = [], [], [], []
    nominal_cmds, final_cmds = [], []
    
    for step in range(total_steps):
        t = step * dt
        
        nominal_twist = torch.tensor([0.1 * torch.sin(torch.tensor(t * 5.0)), 0.0, 0.0, 0.0, 0.0, 0.0])
        corrective_twist = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        
        # Simulate a bump between step 60 and step 110
        if 60 <= step <= 110:
            error = 0.08 + (0.01 * torch.randn(1).item())
        else:
            error = 0.005 + (0.002 * torch.randn(1).item())
            
        final_twist, gate, alpha = interceptor.step(nominal_twist, corrective_twist, error, t)
        
        times.append(t)
        errors.append(error)
        gate_states.append(gate)
        alphas.append(alpha)
        nominal_cmds.append(nominal_twist[0].item())
        final_cmds.append(final_twist[0].item())
        
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(times, errors, label='Lie Geometric Error ($E_t$)', color='red', alpha=0.6)
    axs[0].axhline(y=0.05, color='black', linestyle='--', label='$\\tau_{upper}$ (Trigger Threshold)')
    axs[0].axhline(y=0.01, color='gray', linestyle=':', label='$\\tau_{lower}$ (Reset Threshold)')
    axs[0].fill_between(times, 0, gate_states, color='orange', alpha=0.2, label='Interceptor Gate Active ($g_t$)')
    axs[0].set_ylabel('Error / Gate State')
    axs[0].set_title('Reflex Interceptor: Disturbance Detection & Dwell-Time Gate')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(times, nominal_cmds, label='Nominal Visual Command ($\\xi_{flow}$)', color='blue', linestyle='--')
    axs[1].plot(times, final_cmds, label='Intercepted Output Command ($\\xi_{cmd}$)', color='green', linewidth=2)
    axs[1].plot(times, alphas, label='Bumpless Filter ($\\alpha(t)$)', color='purple', alpha=0.5)
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Control Twist Velocity')
    axs[1].set_title('Control Stream Interception & Smooth Reflex Injection')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interceptor_test_result.png', dpi=300)
    print("Simulation complete! Results saved to 'interceptor_test_result.png'.")

if __name__ == "__main__":
    run_interceptor_simulation()