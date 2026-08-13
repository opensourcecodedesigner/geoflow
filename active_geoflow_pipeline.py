import os
import torch
import torch.nn as nn

# Import your Phase 1 model and Interceptor from existing files
from train_phase1_reflex import PredictiveMotorModel
from test_interceptor import ReflexInterceptor

class ActiveGeoFlowPipeline(nn.Module):
    def __init__(self, reflex_weights_path='checkpoints/reflex_best_model.pth'):
        super().__init__()
        # 1. Load trained Predictive Motor Model
        self.motor_model = PredictiveMotorModel()
        if os.path.exists(reflex_weights_path):
            device_loc = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.motor_model.load_state_dict(torch.load(reflex_weights_path, map_location=device_loc))
            print(f"Successfully loaded reflex weights from '{reflex_weights_path}'.")
        else:
            print(f"Warning: '{reflex_weights_path}' not found. Running with uninitialized weights.")
            
        self.motor_model.eval()
        
        # 2. Instantiate Interceptor Middleware
        self.interceptor = ReflexInterceptor(tau_upper=0.05, tau_lower=0.01, T_min=0.1)
        
        # Fixed Proportional Gain for position error correction
        self.K_p = 2.5

    def compute_lie_error(self, actual_state, predicted_state):
        """Computes geometric distance across 13-DoF kinematics."""
        e_p = actual_state[:, :3] - predicted_state[:, :3]
        err_p = torch.norm(e_p, dim=-1).pow(2)
        
        # Quaternion dot product for rotational discrepancy
        dot = (actual_state[:, 3:7] * predicted_state[:, 3:7]).sum(dim=-1).abs()
        err_q = 1.0 - dot
        
        err_vw = torch.norm(actual_state[:, 7:] - predicted_state[:, 7:], dim=-1).pow(2)
        
        E_t = (err_p + err_q + err_vw).mean().item()
        return E_t, e_p

    def step(self, nominal_twist, actual_state, prev_command, current_time, prev_predicted_state=None):
        """
        Dual-timescale execution step.
        Compares actual_state to expected state from the previous step (prev_predicted_state)
        to accurately capture unexpected physical perturbations.
        """
        with torch.no_grad():
            predicted_state = self.motor_model(actual_state, prev_command)
            
        # Use previous step's prediction as reference; fallback to actual_state on step 0
        reference_state = prev_predicted_state if prev_predicted_state is not None else actual_state
        E_t, e_p = self.compute_lie_error(actual_state, reference_state)
        
        # Compute proportional corrective twist delta (\delta\xi_t)
        batch_size = actual_state.shape[0]
        corrective_twist = -self.K_p * torch.cat([e_p, torch.zeros(batch_size, 3, device=e_p.device)], dim=-1)
        
        # --- COMMAND SATURATION (Clamping to training distribution) ---
        # The Phase 1 model was trained on commands with a max norm of ~0.3
        u_train_max_norm = 0.3 
        delta_xi_norm = torch.norm(corrective_twist, dim=-1, keepdim=True)
        
        # Scale down the vector if its magnitude exceeds the trained maximum, preserving direction
        scale_factor = torch.clamp(u_train_max_norm / (delta_xi_norm + 1e-8), max=1.0)
        corrective_twist = corrective_twist * scale_factor
        # --------------------------------------------------------------
        
        final_twist, gate, alpha = self.interceptor.step(
            nominal_twist=nominal_twist,
            corrective_twist=corrective_twist,
            geometric_error=E_t,
            current_time=current_time
        )
        
        # Return predicted_state so the caller can track expectation for next step
        return final_twist, E_t, gate, alpha, predicted_state