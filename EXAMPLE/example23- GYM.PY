# Example 3.23 - Gymnasium CartPole
import gymnasium as gym

# Perbaikan: 'CartPole-v1' tanpa spasi
env = gym.make('CartPole-v1', render_mode='human')

observation, info = env.reset()

for _ in range(1000):
    # Pilih aksi acak
    action = env.action_space.sample()
    
    # Jalankan simulasi (mengembalikan 5 nilai di Gymnasium)
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Reset jika tiang jatuh atau waktu habis
    if terminated or truncated:
        observation, info = env.reset()

env.close()