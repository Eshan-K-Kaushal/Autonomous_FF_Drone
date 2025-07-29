# AUTHOR: ESHAN K KAUSHAL

import os
import json
import struct
import time
import socket
import select
from collections import deque

import numpy as np
import torch
import torch.nn as nn

SEQUENCE_WINDOW = 50
TRANSITION_DURATION = 100
MODEL_DIR = "bc_models_seq"
GLOBAL_NORMALIZER_PATH = f"{MODEL_DIR}/global_normalizer.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_KEYS = [
    "bev_vector", "position", "velocity", "orientation", "altitude", "battery",
    "is_hover_mode", "min_bev_distance", "relative_fire_pos", "relative_landing_pad_pos"
]

MAX_LIDAR_POINTS = 512

def fix_num_points(pts_np, k=MAX_LIDAR_POINTS):
    n = len(pts_np)
    if n == 0:
        return np.zeros((k, 3), dtype=np.float32)
    if n >= k:
        idx = np.linspace(0, n-1, k).astype(np.int64)
        return pts_np[idx]
    out = np.zeros((k, 3), dtype=np.float32)
    out[:n] = pts_np
    return out

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
            nn.Linear(64, 3)
        )

    def forward(self, state_seq, lidar_points):
        # state_seq: (B, T, D), lidar_points: (B, P, 3)
        _, (h_n, _) = self.lstm(state_seq)
        state_repr = h_n.squeeze(0)              # (B, 128)

        lidar_embed = self.lidar_fc(lidar_points)  # (B, P, 32)
        lidar_encoded = self.lidar_encoder(lidar_embed)  # (B, P, 32)
        lidar_repr = lidar_encoded.mean(dim=1)     # (B, 32)

        fused = torch.cat([state_repr, lidar_repr], dim=1)  # (B, 160)
        cont_out = self.head_cont(fused)  # (B, 3) in [-1,1]
        bin_out = self.head_bin(fused)    # (B, 3) logits
        return torch.cat([cont_out, bin_out], dim=1)  # (B, 7)

class FeatureNormalizer:
    def __init__(self, stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.min_vals = stats["min_vals"]
        self.max_vals = stats["max_vals"]

    def normalize(self, feat, val):
        val = np.array(val if isinstance(val, list) else [float(val)], dtype=np.float32)
        minv = np.array(self.min_vals.get(feat, [0.0]), dtype=np.float32)
        maxv = np.array(self.max_vals.get(feat, [1.0]), dtype=np.float32)
        return (val - minv) / (maxv - minv + 1e-8)

models, normalizers = {}, {}

def load_model(model_path, input_dim):
    if model_path not in models:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = HybridDronePolicy(state_input_dim=input_dim).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models[model_path] = model
        print(f" Loaded model: {model_path}")
    return models[model_path]

def load_normalizer(path):
    if path not in normalizers:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalizer not found: {path}")
        normalizers[path] = FeatureNormalizer(path)
        print(f" Loaded normalizer: {path}")
    return normalizers[path]

def extract_features(state, keys, normalizer):
    vec = []
    for k in keys:
        val = state.get(k, [0.0])
        norm = normalizer.normalize(k, val)
        vec.extend(norm)
    return np.array(vec, dtype=np.float32)

# FSM - Finite Sate Machine
current_phase = None
previous_phase = None
transition_steps_remaining = 0
raw_state_window = deque(maxlen=SEQUENCE_WINDOW)
last_input_dim = None


def run_inference(state_dict, reported_phase):
    """
    Returns a 6-element action:
      [pitch, yaw, ascend_force, toggle_water(bool), toggle_hover(bool), toggle_hover_flip(bool)]
    or None during warmup.
    """
    global current_phase, previous_phase, transition_steps_remaining, last_input_dim

    # append latest state
    raw_state_window.append(state_dict)

    # phase change handling
    if current_phase != reported_phase:
        if current_phase is not None:
            previous_phase = current_phase
            transition_steps_remaining = TRANSITION_DURATION
        current_phase = reported_phase
        print(f" Phase switched: {previous_phase} → {current_phase}")

    use_transition = transition_steps_remaining > 0

    model_name = f"{previous_phase}_to_{current_phase}" if use_transition else current_phase

    if use_transition:
        step_idx = TRANSITION_DURATION - transition_steps_remaining + 1
        print(f" Using transition model: {model_name} ({step_idx}/{TRANSITION_DURATION})")
        transition_steps_remaining -= 1
    else:
        print(f" Using phase model: {model_name}")

    # warmup until we have SEQUENCE_WINDOW states for the LSTM context
    if len(raw_state_window) < SEQUENCE_WINDOW:
        print(f" Warmup: Collected {len(raw_state_window)}/{SEQUENCE_WINDOW} states")
        return None

    # loadin normalizer and build normalized sequence (T, D)
    normalizer = load_normalizer(GLOBAL_NORMALIZER_PATH)
    state_seq = [extract_features(s, STATE_KEYS, normalizer) for s in raw_state_window]
    input_dim = len(state_seq[0])

    if last_input_dim != input_dim:
        print(f" Input dim updated: {last_input_dim} → {input_dim}")
        last_input_dim = input_dim

    # convert to tensors
    state_np = np.stack(state_seq, axis=0)  # (T, D)
    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)  # (1, T, D)

    # LiDAR: pad/subsample to fixed length (P, 3) -> (1, P, 3)
    lidar_points = state_dict.get("lidar_points", [])
    lidar_np = np.array(lidar_points, dtype=np.float32).reshape(-1, 3)
    lidar_np = fix_num_points(lidar_np, k=MAX_LIDAR_POINTS)
    lidar_tensor = torch.from_numpy(lidar_np).unsqueeze(0).to(device)  # (1, P, 3)

    # model loading for this phase
    model_path = os.path.join(MODEL_DIR, f"hybrid_bc_{model_name}.pt")
    model = load_model(model_path, input_dim)

    # inference part
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                output = model(state_tensor, lidar_tensor)[0].cpu()
        else:
            output = model(state_tensor, lidar_tensor)[0].cpu()

    pred_cont = output[:3].tolist()

    # logits -> probs -> booleans
    pred_bin_probs = torch.sigmoid(output[3:6]).tolist()
    pred_bin = [bool(p > 0.5) for p in pred_bin_probs]

    final_action = pred_cont + pred_bin

    if len(final_action) != 6:
        print(f" Warning: Incomplete action vector (len={len(final_action)}), using fallback")
        return [0.0, 0.0, 0.0, False, False, False]

    return final_action

def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk
    return data

def recv_json(sock):
    header = recv_exact(sock, 4)
    length = struct.unpack(">I", header)[0]
    payload = recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))

def send_json(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    length = struct.pack(">I", len(payload))
    sock.sendall(length + payload)


# ----------- SERVER LOOP -------------------
HOST = "127.0.0.1"
PORT = 5005

def serve_once():
    """Accept a single client and serve until disconnect."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        print(f" Hybrid FSM Server running on {HOST}:{PORT}")
        conn, addr = server_sock.accept()
        print(f" Connected by {addr}")

        with conn:
            send_json(conn, {"status": "ready"})
            while True:
                try:
                    ready_to_read, _, _ = select.select([conn], [], [], 0.01)
                    if not ready_to_read:
                        time.sleep(0.001)
                        continue

                    msg = recv_json(conn)
                    state = msg["state"]
                    phase = msg["phase"]
                    print(f" Phase: {phase}")
                    action = run_inference(state, phase)

                    if action is not None:
                        send_json(conn, {"action": action})
                        print(f" Action: {action} → sent to Godot")
                    else:
                        send_json(conn, {"status": "warmup_in_progress"})

                    if "action" in msg and action is not None:
                        expected = msg["action"]
                        predicted = action
                        print(f" Expected: {expected}")
                        print(f" Predicted: {predicted}")

                except ConnectionError as e:
                    print(f"🔌 Client disconnected: {e}")
                    break
                except Exception as e:
                    print(f" Error: {e}")
                    break

if __name__ == "__main__":
    # simple reconnect loop
    while True:
        try:
            serve_once()
        except Exception as e:
            print(f" Server error: {e}")
        print(" Waiting 1s before accepting a new client…")
        time.sleep(1.0)
