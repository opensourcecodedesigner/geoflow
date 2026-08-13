import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from active_geoflow_pipeline import ActiveGeoFlowPipeline

def run_live_physics_experiment():
    pipeline = ActiveGeoFlowPipeline(reflex_weights_path='checkpoints/reflex_best_model.pth')
    
    dt = 0.002           # 500 Hz control step
    total_steps = 300    # Extended slightly to show the release clearly
    
    times, one_step_residuals, standing_errors, gates, alphas = [], [], [], [], []
    applied_forces, blended_outputs = [], []
    
    actual_state = torch.zeros(1, 13)
    actual_state[0, 3] = 1.0
    prev_command = torch.zeros(1, 6)
    prev_predicted_state = None
    
    nominal_position = 0.0
    
    print("\nExecuting Live Physics Integration Run (with Ground-Truth Telemetry)...")
    
    for step in range(total_steps):
        t = step * dt
        
        if step % 33 == 0:
            nominal_twist = torch.tensor([[0.15 * np.sin(t * 4.0), 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            
        nominal_position += nominal_twist[0, 0].item() * dt
            
        external_force = 0.0
        disturbed_state = actual_state.clone()
        
        if 80 <= step <= 140:
            external_force = 12.5  
            disturbed_state[0, 0] += (external_force / 1.5) * (dt ** 2) + 0.35  
            disturbed_state[0, 7] += (external_force / 1.5) * dt              
            
        final_twist, one_step_residual, gate, alpha, prev_predicted_state = pipeline.step(
            nominal_twist=nominal_twist,
            actual_state=disturbed_state,
            prev_command=prev_command,
            current_time=t,
            prev_predicted_state=prev_predicted_state
        )
        
        actual_state = disturbed_state.clone()
        actual_state[0, 0:3] += final_twist[0, 0:3] * dt 
        actual_state[0, 7:13] = final_twist[0, 0:6]      
        
        prev_command = final_twist
        
        standing_error = (actual_state[0, 0].item() - nominal_position) ** 2

        times.append(t)
        one_step_residuals.append(one_step_residual)
        standing_errors.append(standing_error)
        gates.append(gate)
        alphas.append(alpha)
        applied_forces.append(external_force)
        blended_outputs.append(final_twist[0, 0].item())

    # --- GROUND-TRUTH TELEMETRY ANALYSIS ---
    residuals_arr = np.array(one_step_residuals)
    gates_arr = np.array(gates)
    
    # Find when the gate first opened
    gate_indices = np.where(gates_arr > 0)[0]
    if len(gate_indices) > 0:
        gate_start_idx = gate_indices[0]
        # Minimum residual reached AFTER the impact window started
        min_post_impact_residual = np.min(residuals_arr[gate_start_idx:])
    else:
        min_post_impact_residual = np.min(residuals_arr)

    print("\n" + "="*50)
    print(" GROUND-TRUTH INTERCEPTOR TELEMETRY REPORT")
    print("="*50)
    print(f"Tau Lower Threshold      : {pipeline.interceptor.tau_lower}")
    print(f"Tau Upper Threshold      : {pipeline.interceptor.tau_upper}")
    print(f"Min Residual Post-Impact : {min_post_impact_residual:.5f}")
    print(f"Steady-State Residual    : {residuals_arr[-1]:.5f}")
    print(f"Final Gate State at t=0.6: {gates_arr[-1]}")
    print("="*50)
    
    if min_post_impact_residual > pipeline.interceptor.tau_lower:
        print(" DIAGNOSIS: The residual never dropped below tau_lower.")
        print("   -> The gate logic is working correctly; the model's post-disturbance")
        print("      noise floor is simply higher than 0.01.")
    else:
        print(" DIAGNOSIS: The residual DID drop below tau_lower, but the gate stayed open.")
        print("   -> There is a latching or reset bug in the interceptor logic.")
    print("="*50 + "\n")

    # --- Telemetry Visualization ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(times, applied_forces, color='crimson', linewidth=2, label='External Impact Force (N)')
    axs[0].set_ylabel('Force (N)')
    axs[0].set_title('Live Physics Environment: Telemetry Verified')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    ax2 = axs[1]
    ax2.plot(times, one_step_residuals, color='red', linewidth=2, label='1-Step Residual ($E_t$)')
    ax2.axhline(y=pipeline.interceptor.tau_upper, color='black', linestyle='--', label='$\\tau_{upper}$')
    ax2.axhline(y=pipeline.interceptor.tau_lower, color='gray', linestyle=':', label='$\\tau_{lower}$')
    ax2.fill_between(times, 0, [g * 0.15 for g in gates], color='orange', alpha=0.2, label='Gate Active ($g_t$)')
    ax2.set_ylabel('Residual Error')
    ax2.set_ylim(-0.01, 0.2)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, standing_errors, color='blue', linestyle='--', alpha=0.6, label='Standing Error (Right Axis)')
    ax2_twin.set_ylabel('Standing Error Scale', color='blue')
    ax2_twin.legend(loc='upper right')
    
    axs[2].plot(times, blended_outputs, color='green', linewidth=2, label='Active-GeoFlow Command ($\\xi_{cmd}$)')
    axs[2].plot(times, alphas, color='purple', alpha=0.5, label='Bumpless Filter ($\\alpha(t)$)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Control Velocity')
    axs[2].legend(loc='lower right')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_live_physics_experiment()