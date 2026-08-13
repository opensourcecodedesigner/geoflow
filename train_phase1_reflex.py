import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. KINEMATIC MATH HELPERS
# ==========================================
def quat_multiply(q1, q2):
    """Batched quaternion multiplication."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)

def composite_kinematic_loss(predictions, targets):
    """
    Computes loss respecting the SO(3) double-cover.
    Returns individual components to track learning across channels.
    """
    pred_p, pred_q, pred_vw = predictions[:, :3], predictions[:, 3:7], predictions[:, 7:]
    targ_p, targ_q, targ_vw = targets[:, :3], targets[:, 3:7], targets[:, 7:]
    
    # Standard MSE for Euclidean components
    loss_p = F.mse_loss(pred_p, targ_p)
    loss_vw = F.mse_loss(pred_vw, targ_vw)
    
    # Sign-Invariant Quaternion Loss: 1 - |q_pred dot q_target|
    dot_product = (pred_q * targ_q).sum(dim=1).abs()
    loss_q = (1.0 - dot_product).mean()
    
    return loss_p, loss_q, loss_vw

# ==========================================
# 1. THE PREDICTIVE MOTOR MODEL (Phase 1)
# ==========================================
class PredictiveMotorModel(nn.Module):
    """
    The 'Fast Cerebellum' MLP. 
    Inputs strictly 13-DoF state + 6-DoF command. Vision-blind.
    """
    def __init__(self, state_dim=13, twist_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + twist_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, command):
        x = torch.cat([state, command], dim=-1)
        state_delta = self.net(x)
        next_state = state + state_delta
        
        # Ensure predicted quaternion remains a valid unit quaternion
        norm_q = next_state[:, 3:7] / torch.norm(next_state[:, 3:7], dim=1, keepdim=True)
        next_state = torch.cat([next_state[:, :3], norm_q, next_state[:, 7:]], dim=1)
        
        return next_state

# ==========================================
# 2. SYNTHETIC DATA GENERATOR
# ==========================================
def get_synthetic_dataloader(num_samples=1000, batch_size=32):
    """Generates synthetic 13-DoF proprioceptive states and 6-DoF twists."""
    dt = 0.05 # Simulated time step
    
    p = torch.randn(num_samples, 3)
    q = torch.randn(num_samples, 4)
    q = q / torch.norm(q, dim=1, keepdim=True) 
    v = torch.randn(num_samples, 3) * 0.1
    w = torch.randn(num_samples, 3) * 0.1
    states = torch.cat([p, q, v, w], dim=1)
    
    commands = torch.randn(num_samples, 6) * 0.1
    
    # Physics rollout
    next_v = v + commands[:, :3] * dt
    next_w = w + commands[:, 3:] * dt
    next_p = p + next_v * dt
    
    # First-order orientation update
    delta_q_w = torch.ones(num_samples, 1)
    delta_q_xyz = next_w * (dt / 2.0)
    delta_q = torch.cat([delta_q_w, delta_q_xyz], dim=1)
    delta_q = delta_q / torch.norm(delta_q, dim=1, keepdim=True)
    next_q = quat_multiply(q, delta_q)
    
    # Add minor observation noise
    next_states = torch.cat([next_p, next_q, next_v, next_w], dim=1) + (torch.randn(num_samples, 13) * 0.001)
    
    dataset = torch.utils.data.TensorDataset(states, commands, next_states)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 3. TRAINING LOOP WITH CHECKPOINTING
# ==========================================
def train_reflex_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictiveMotorModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader = get_synthetic_dataloader(num_samples=3000)
    val_loader = get_synthetic_dataloader(num_samples=500)
    
    epochs = 50
    train_losses, val_losses = [], []
    train_p_track, train_q_track, train_vw_track = [], [], []
    best_val_loss = float('inf')
    
    os.makedirs('checkpoints', exist_ok=True)
    print(f"Starting Training on {device} for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # -- Training --
        model.train()
        epoch_train_loss, epoch_p, epoch_q, epoch_vw = 0.0, 0.0, 0.0, 0.0
        
        for states, commands, next_states in train_loader:
            states, commands, next_states = states.to(device), commands.to(device), next_states.to(device)
            
            optimizer.zero_grad()
            predictions = model(states, commands)
            loss_p, loss_q, loss_vw = composite_kinematic_loss(predictions, next_states)
            loss = loss_p + loss_q + loss_vw
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_p += loss_p.item()
            epoch_q += loss_q.item()
            epoch_vw += loss_vw.item()
            
        num_batches = len(train_loader)
        train_losses.append(epoch_train_loss / num_batches)
        train_p_track.append(epoch_p / num_batches)
        train_q_track.append(epoch_q / num_batches)
        train_vw_track.append(epoch_vw / num_batches)
        
        # -- Validation --
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for states, commands, next_states in val_loader:
                states, commands, next_states = states.to(device), commands.to(device), next_states.to(device)
                predictions = model(states, commands)
                v_loss_p, v_loss_q, v_loss_vw = composite_kinematic_loss(predictions, next_states)
                epoch_val_loss += (v_loss_p + v_loss_q + v_loss_vw).item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # -- Dr. Balaji's Directive: 5-Epoch Checkpointing --
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, f'checkpoints/reflex_epoch_{epoch}.pth')
            
        # -- Insurance Policy: Best Model Tracking --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/reflex_best_model.pth')

    # ==========================================
    # 4. COMPONENT LOSS CURVE VISUALIZATION
    # ==========================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = range(1, epochs + 1)

    # Plot 1: Total Loss
    axs[0, 0].plot(epochs_range, train_losses, label='Train Total', color='black')
    axs[0, 0].plot(epochs_range, val_losses, label='Val Total', linestyle='--', color='gray')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Total Composite Loss (Log Scale)')
    axs[0, 0].legend()

    # Plot 2: Position Loss
    axs[0, 1].plot(epochs_range, train_p_track, label='Train Position', color='blue')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Position Dynamics Loss')
    axs[0, 1].legend()

    # Plot 3: Quaternion Loss
    axs[1, 0].plot(epochs_range, train_q_track, label='Train Quaternion', color='purple')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Rotational Dynamics Loss (Sign-Invariant)')
    axs[1, 0].legend()

    # Plot 4: Velocity Loss
    axs[1, 1].plot(epochs_range, train_vw_track, label='Train Velocity/Angular', color='green')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Velocity Dynamics Loss')
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Loss')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('reflex_loss_components.png', dpi=300)
    print("\nTraining Complete! Checkpoints saved to '/checkpoints' and loss curve saved as 'reflex_loss_components.png'.")

if __name__ == "__main__":
    train_reflex_model()