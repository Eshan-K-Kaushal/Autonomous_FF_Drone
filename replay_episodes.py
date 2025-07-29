import json
import socket
import struct
import os

FSM_HOST = "127.0.0.1"
FSM_PORT = 5005

def send_json(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    length = struct.pack(">I", len(payload))
    sock.sendall(length + payload)

def recv_json(sock):
    header = sock.recv(4)
    if len(header) < 4:
        raise ConnectionError("Incomplete header")
    length = struct.unpack(">I", header)[0]
    payload = sock.recv(length)
    return json.loads(payload.decode("utf-8"))

def test_replay(log_path):
    with open(log_path) as f:
        episode = json.load(f)

    print(f"🎞 Replaying {len(episode)} steps from: {log_path}")
    prev_phase = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((FSM_HOST, FSM_PORT))
        for i, step in enumerate(episode):
            state = step['state']
            phase = step['mission_state']
            true_action = step['action']

            # Send to FSM inference server
            send_json(sock, {
                "state": state,
                "phase": phase,
                "previous_phase": prev_phase
            })

            response = recv_json(sock)
            model_action = response.get("action", None)

            if model_action is None:
                print(f" - No action at step {i}")
                continue

            print(f"Step {i:03} | Phase: {phase}")
            print(f" - True:  {true_action}")
            print(f" - Model: {model_action}\n")

            prev_phase = phase

if __name__ == "__main__":
    test_replay("autodroneenv - isolatedBC - MSP/myruns/episode_takeoff_1752781618.json")
