import sys, time, os
import pybullet as p
from stable_baselines3 import DQN
from anomaly_env import AnomalyEnv

PRUEBAS = {"banderas": 0, "objetos": 1, "patrones": 2, "diferencias": 3}

if len(sys.argv) < 2 or sys.argv[1] not in PRUEBAS:
    print("Uso: python evaluate_dqn.py [banderas|objetos|patrones|diferencias]")
    sys.exit(1)

cap = PRUEBAS[sys.argv[1]]

# --- Clase GUI ---
class AnomalyEnvGUI(AnomalyEnv):
    def __init__(self, cap):
        super().__init__()
        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI)
        self.set_world(cap)
        self.reset()

# --- Crear entorno ---
env = AnomalyEnvGUI(cap)

# --- Rutas ---
os.makedirs("dqn_models", exist_ok=True)
model_path = os.path.join("dqn_models", "dqn_final.zip")

# --- Entrenar o cargar ---
if os.path.exists(model_path):
    print(f"[INFO] Cargando modelo DQN desde {model_path}")
    model = DQN.load(model_path, env=env)
    env.set_anomaly(True) 
else:
    print("[INFO] Entrenando nuevo modelo DQN...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        verbose=1,
        tensorboard_log="./dqn_tb/"
    )
    timesteps = 300
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, log_interval=10)
    model.save(model_path)
    print(f"[INFO] Modelo guardado en {model_path}")

# --- Evaluar ---
env.set_anomaly(True)
obs = env.reset()
done = False
total_reward = 0.0
step_count = 0
model.exploration_rate = 0.05
while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    step_count += 1
    time.sleep(1/120)
    print(f"[Eval] Paso {step_count}: acción={action} | recompensa={reward:.4f}")

print(f"[RESULTADO] Recompensa total (eval): {total_reward:.4f}")
env.close()
