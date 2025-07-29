# AUTHOR: ESHAN K KAUSHAL

import os, json, re, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import defaultdict
import random

# config
ALL_FEATURE_KEYS = [
    "bev_vector", "position", "velocity", "orientation", "altitude", "battery",
    "is_hover_mode", "min_bev_distance", "relative_fire_pos", "relative_landing_pad_pos"
]
LOG_DIR = "autodroneenv - isolatedBC - MSP/myruns/"
GLOBAL_NORMALIZER_PATH = "bc_models_seq/global_normalizer.json"
WINDOW_SIZE = 50
VAL_SPLIT = 0.15
BATCH_SIZE = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=42, strict=False):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ut funcs
def normalize_binary_flags(action):
    action["toggle_water"] = 1.0 if action.get("toggle_water", False) else 0.0
    action["toggle_hover"] = 1.0 if action.get("toggle_hover", 0) else 0.0
    action["toggle_hover_flip"] = 1.0 if action.get("toggle_hover_flip", 0) else 0.0
    return action

MAX_LIDAR_POINTS = 512  # sweet spot

def fix_num_points(pts_np, k=MAX_LIDAR_POINTS, deterministic=False):
    n = len(pts_np)
    if n == 0:
        return np.zeros((k, 3), dtype=np.float32)
    if n >= k:
        if deterministic:
            # uniform stride selection (deterministic, order-preserving)
            idx = np.linspace(0, n-1, k).astype(np.int64)
            return pts_np[idx]
        idx = np.random.choice(n, k, replace=False)
        return pts_np[idx]
    out = np.zeros((k, 3), dtype=np.float32)
    out[:n] = pts_np
    return out

def patch_all_log_files(log_dir):
    for fname in os.listdir(log_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f" Failed to load {fname}: {e}")
            continue

        modified = False
        for entry in data:
            if "action" in entry:
                old_action = dict(entry["action"])
                entry["action"] = normalize_binary_flags(entry["action"])
                if entry["action"] != old_action:
                    modified = True

        if modified:
            try:
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=2)
                print(f" Patched: {fname}")
            except Exception as e:
                print(f" Failed to write {fname}: {e}")

# model
class HybridDronePolicy(nn.Module):
    def __init__(self, state_input_dim, lidar_points_dim=3, output_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(state_input_dim, 128, batch_first=True)
        self.lidar_fc = nn.Linear(lidar_points_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.lidar_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head_cont = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()  # continuous outputs in [-1, 1]
        )
        self.head_bin = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)      # logits for BCEWithLogits
        )

    def forward(self, state_seq, lidar_points):
        # state_seq: (B, T, D), lidar_points: (B, P, 3)
        _, (h_n, _) = self.lstm(state_seq)
        state_repr = h_n.squeeze(0)  # (B, 128)

        lidar_embed = self.lidar_fc(lidar_points)          # (B, P, 32)
        lidar_encoded = self.lidar_encoder(lidar_embed)    # (B, P, 32)
        lidar_repr = lidar_encoded.mean(dim=1)             # (B, 32)

        fused = torch.cat([state_repr, lidar_repr], dim=1) # (B, 160)
        cont_out = self.head_cont(fused)                   # (B, 3) in [-1,1]
        bin_out = self.head_bin(fused)                     # (B, 3) logits
        return torch.cat([cont_out, bin_out], dim=1)       # (B, 6)

# nrmlzer
class FeatureNormalizer:
    def __init__(self, feature_keys):
        self.feature_keys = feature_keys
        self.min_vals = {}
        self.max_vals = {}

    def fit(self, all_episodes):
        for feat in self.feature_keys:
            values = []
            for ep in all_episodes:
                val = ep['state'].get(feat, [0.0])
                values.append(val if isinstance(val, list) else [val])
            arr = np.array(values, dtype=np.float32)
            if arr.ndim == 1:
                self.min_vals[feat] = float(arr.min())
                self.max_vals[feat] = float(arr.max())
            else:
                self.min_vals[feat] = arr.min(axis=0).tolist()
                self.max_vals[feat] = arr.max(axis=0).tolist()

    def normalize(self, feat, val):
        val = np.array(val if isinstance(val, list) else [val], dtype=np.float32)
        minv = np.array(self.min_vals.get(feat, 0.0), dtype=np.float32)
        maxv = np.array(self.max_vals.get(feat, 1.0), dtype=np.float32)
        return ((val - minv) / (maxv - minv + 1e-8)).tolist()

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"min_vals": self.min_vals, "max_vals": self.max_vals}, f)

class SequenceDataset(Dataset):
    """
    Builds sequence windows that DO NOT cross episode boundaries.
    Also uses deterministic LiDAR padding/subsample for validation (no noise).
    """
    def __init__(self, episodes, feature_keys, normalizer, window_size=50, add_noise=False):
        self.feature_keys = feature_keys
        self.normalizer = normalizer
        self.window_size = window_size
        self.add_noise = add_noise
        self.samples = []

        # (1) group by episode_id
        by_ep = defaultdict(list)
        for s in episodes:
            if 'state' in s and 'action' in s and 'episode_id' in s:
                by_ep[s['episode_id']].append(s)

        # (2) sort each episode by step and build windows entirely within that episode
        for eid, seq in by_ep.items():
            seq.sort(key=lambda s: s.get('step', -1))
            if len(seq) < window_size:
                continue
            for i in range(len(seq) - window_size + 1):
                window = seq[i:i + window_size]
                # sanity: all have state/action
                if all(('state' in w and 'action' in w) for w in window):
                    self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        lstm_seq = []

        for step in window:
            vec = []
            for feat in self.feature_keys:
                val = step['state'].get(feat, [0.0])
                norm_val = self.normalizer.normalize(feat, val)
                if self.add_noise:
                    norm_val = (np.array(norm_val) + np.random.normal(0, 0.01, size=len(norm_val))).tolist()
                vec.extend(norm_val)
            lstm_seq.append(vec)

        # LiDAR from last step of the window
        lidar_raw = window[-1]['state'].get("lidar_points", [])
        lidar_np = np.array(lidar_raw, dtype=np.float32).reshape(-1, 3)

        # Deterministic for validation (no noise) to stabilize metrics
        lidar_np = fix_num_points(
            lidar_np,
            k=MAX_LIDAR_POINTS,
            deterministic=(not self.add_noise)
        )

        lidar_tensor = torch.from_numpy(lidar_np)
        if self.add_noise:
            lidar_tensor = lidar_tensor + torch.randn_like(lidar_tensor) * 0.01

        # Targets
        action = normalize_binary_flags(window[-1]['action'])
        ascend_force = action.get('ascend', 0.0) - action.get('descend', 0.0)
        y = torch.tensor([
            action.get('pitch', 0.0),
            action.get('yaw', 0.0),
            ascend_force,
            action['toggle_water'],
            action['toggle_hover'],
            action['toggle_hover_flip']
        ], dtype=torch.float32)

        return torch.tensor(lstm_seq, dtype=torch.float32), lidar_tensor, y

def load_all_episodes(log_dir):
    all_data = []
    for fname in os.listdir(log_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(log_dir, fname)) as f:
                all_data.extend(json.load(f))
        except Exception:
            continue
    return all_data

# training -------------------------------------------------------------
def train_model(samples, feature_keys, model_name, normalizer,
                out_dir="bc_models_seq", window_size=50, seed=42,
                max_epochs=100, es_patience=8, es_min_delta=0.01,
                clip_grad=1.0):
    """
    es_min_delta is relative: 0.01 = require 1% improvement vs current best.
    """
    os.makedirs(out_dir, exist_ok=True)

    full_ds = SequenceDataset(samples, feature_keys, normalizer, window_size)
    if len(full_ds) == 0:
        print(f" No sequences available for phase '{model_name}' (window_size={window_size}). Skipping.")
        return

    # deterministic split/shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    val_len = int(len(full_ds) * VAL_SPLIT)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = full_ds[0][0].shape[-1]
    model = HybridDronePolicy(state_input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # losses
    loss_cont = nn.MSELoss(reduction='sum')

    # class imbalance handling for toggles
    toggle_counts = torch.zeros(3)
    for _, _, yb in train_loader:
        toggle_counts += yb[:, 3:].sum(dim=0)
    total = len(train_loader.dataset)
    pos_weight = (total - toggle_counts) / (toggle_counts + 1e-6)
    pos_weight = pos_weight.clamp(min=1.0, max=100.0).to(device)

    loss_bin_water = nn.BCEWithLogitsLoss(pos_weight=pos_weight[0])
    loss_bin_hover = nn.BCEWithLogitsLoss(pos_weight=pos_weight[1])
    loss_bin_flip  = nn.BCEWithLogitsLoss(pos_weight=pos_weight[2])

    print(f" Binary toggle pos_weights: {pos_weight.tolist()}")

    # --- early stoppin / best save state ---
    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_path = os.path.join(out_dir, f"hybrid_bc_{model_name}.pt")
    last_path = os.path.join(out_dir, f"hybrid_bc_{model_name}_last.pt")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for state_seq, lidar_pts, yb in train_loader:
            state_seq = state_seq + torch.randn_like(state_seq) * 0.01
            lidar_pts = lidar_pts + torch.randn_like(lidar_pts) * 0.01

            state_seq = state_seq.to(device)
            lidar_pts = lidar_pts.to(device)
            yb = yb.to(device)

            pred = model(state_seq, lidar_pts)
            pred_cont, pred_bin_logits = pred[:, :3], pred[:, 3:]
            target_cont, target_bin = yb[:, :3], yb[:, 3:]

            loss = (
                3.0 * loss_cont(pred_cont, target_cont) +
                1.5 * loss_bin_water(pred_bin_logits[:, 0], target_bin[:, 0]) +
                1.5 * loss_bin_hover(pred_bin_logits[:, 1], target_bin[:, 1]) +
                1.5 * loss_bin_flip(pred_bin_logits[:, 2], target_bin[:, 2])
            )

            optimizer.zero_grad()
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            total_count += state_seq.size(0)

        # ---- validation loop ----
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for state_seq, lidar_pts, yb in val_loader:
                state_seq = state_seq.to(device)
                lidar_pts = lidar_pts.to(device)
                yb = yb.to(device)

                pred = model(state_seq, lidar_pts)
                pred_cont, pred_bin_logits = pred[:, :3], pred[:, 3:]
                target_cont, target_bin = yb[:, :3], yb[:, 3:]

                loss = (
                    3.0 * loss_cont(pred_cont, target_cont) +
                    1.5 * loss_bin_water(pred_bin_logits[:, 0], target_bin[:, 0]) +
                    1.5 * loss_bin_hover(pred_bin_logits[:, 1], target_bin[:, 1]) +
                    1.5 * loss_bin_flip(pred_bin_logits[:, 2], target_bin[:, 2])
                )
                val_loss += loss.item()
                val_count += state_seq.size(0)

        # average-per-sample losses
        train_loss_avg = total_loss / max(total_count, 1)
        val_loss_avg = val_loss / max(val_count, 1)
        print(f"[{model_name}] Epoch {epoch:02d} | "
              f"Train Loss: {total_loss:.2f} ({train_loss_avg:.4f}/sample) | "
              f"Val Loss: {val_loss:.2f} ({val_loss_avg:.4f}/sample)")

        # LR scheduler on the raw validation loss (sum or avg — be consistent)
        scheduler.step(val_loss)

        # relative improvement: require val_loss < best_val * (1 - es_min_delta)
        improved = val_loss < best_val * (1.0 - es_min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST weights @ epoch {epoch} → {best_path} (val={val_loss:.2f})")
        else:
            epochs_no_improve += 1

        torch.save(model.state_dict(), last_path)

        # stop if no improvement for `es_patience` epochs
        if epochs_no_improve >= es_patience:
            print(f" Early stopping at epoch {epoch} "
                  f"(no val improvement for {es_patience} epochs; best @ {best_epoch} with {best_val:.2f}).")
            break
    print(f" Saved LAST weights (eshan training version 7.21.25): {last_path}")
    if os.path.exists(best_path):
        print(f" Best weights (eshan training version 7.21.25): {best_path}")

if __name__ == "__main__":
    set_seed(42, strict=False)

    print(" Patching all logs for consistent toggle flags...")
    patch_all_log_files(LOG_DIR)

    print(" Loading all data...")
    all_eps = load_all_episodes(LOG_DIR)
    normalizer = FeatureNormalizer(ALL_FEATURE_KEYS)
    normalizer.fit(all_eps)
    os.makedirs("bc_models_seq", exist_ok=True)
    normalizer.save(GLOBAL_NORMALIZER_PATH)
    print(f" Normalizer saved to {GLOBAL_NORMALIZER_PATH}")
    phase_map = defaultdict(list)
    for fname in os.listdir(LOG_DIR):
        if not fname.endswith(".json"):
            continue

        phase = None
        match = re.match(r"^episode_([a-zA-Z0-9_]+)_\d+\.json$", fname)
        if match:
            phase = match.group(1)
        else:
            trans_match = re.match(r"^episode_transition_([a-zA-Z0-9_]+)_to_([a-zA-Z0-9_]+)_\d+\.json$", fname)
            if trans_match:
                phase = f"{trans_match.group(1)}_to_{trans_match.group(2)}"

        if phase:
            try:
                with open(os.path.join(LOG_DIR, fname)) as f:
                    phase_map[phase].extend(json.load(f))
            except Exception:
                print(f" Skipping invalid: {fname}")

    for phase_name, phase_data in sorted(phase_map.items()):
        print(f" Training phase: {phase_name} ({len(phase_data)} raw steps)")
        train_model(
            samples=phase_data,
            feature_keys=ALL_FEATURE_KEYS,
            model_name=phase_name,
            normalizer=normalizer,
            window_size=WINDOW_SIZE,
            seed=42,
        )
