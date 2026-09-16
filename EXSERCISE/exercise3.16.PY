# Exercise 3.16: Eight-State Q-Learning Routing Problem
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 1. Visualisasi Graph Routing Diagram
def showgraph(points_list):
    G = nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(1, figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    plt.title("8-State Routing Graph (Goal = State 7)")
    plt.tight_layout()

# 2. Membuat Reward Matrix (R)
def createRmat(MATRIX_SIZE, points_list, goal):
    R = np.full((MATRIX_SIZE, MATRIX_SIZE), -1.0)
    for point in points_list:
        u, v = point
        # Arah u -> v
        R[u, v] = 100.0 if v == goal else 0.0
        # Arah v -> u (Graph Undirected)
        R[v, u] = 100.0 if u == goal else 0.0
    
    R[goal, goal] = 100.0  # Self-loop pada goal
    print("--- Matrix R (Reward) ---")
    print(R)
    return R

# 3. Fungsi Helper Q-Learning
def available_actions(R, state):
    current_state_row = R[state]
    return np.where(current_state_row >= 0)[0]

def sample_next_action(available_act):
    return int(np.random.choice(available_act))

def update(R, Q, current_state, action, gamma):
    max_value = np.max(Q[action])
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    
    max_q = np.max(Q)
    if max_q > 0:
        return np.sum(Q / max_q * 100)
    return 0

# --- PARAMETER & INISIALISASI ---
np.random.seed(1)
points_list = [(0,1), (0,2), (1,3), (2,3), (2,4), (3,5), (4,5), (5,6), (5,7), (6,7)]
goal = 7
MATRIX_SIZE = 8
gamma = 0.8

showgraph(points_list)

R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

# --- PROCESS TRAINING Q-LEARNING ---
scores = []
for i in range(700):
    current_state = np.random.randint(0, MATRIX_SIZE)
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("\n--- Trained Q Matrix (Normalized) ---")
print(np.round(Q / np.max(Q) * 100, 2))

# --- MENCARI RUTE TERBAIK (MOST EFFICIENT PATH) ---
current_state = 0
steps = [current_state]
while current_state != goal:
    # Pilih aksi dengan nilai Q tertinggi dari state saat ini
    next_step = int(np.argmax(Q[current_state]))
    steps.append(next_step)
    current_state = next_step

print("\n--- Hasil Jalur Tercepat (Most Efficient Path from 0 to 7) ---")
print(" -> ".join(map(str, steps)))

# 4. Visualisasi Grafik Konvergensi Skor
plt.figure(2, figsize=(7, 4))
plt.plot(scores, color='green')
plt.xlabel("Iteration")
plt.ylabel("Score / Q-Value Sum")
plt.title("Q-Learning Convergence Scores")
plt.grid(True, linestyle='--', alpha=0.6)

# Tampilkan kedua jendela grafik
plt.show()
