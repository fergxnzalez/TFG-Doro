from collections import deque
from pathlib import Path
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from keras.models import load_model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class AnomalyEnv(gym.Env):
    def __init__(self, cap: int = 0, max_steps: int = 1000):
        super().__init__()
        self.action_counter = [0, 0, 0]
        self.action_space = spaces.Discrete(3)
        self.anomaly = False 

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2049,), dtype=np.float32)

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.embeddings_buffer = []
        self.img_buffer = []
        self.current_step = 0
        self.max_steps = max_steps
        self.cap = cap
        self.turn_streak = 0
        self.last_pos = None


        self.anom_buffer = deque(maxlen=8)     # acumula muestras anómalas antes de fit
        self.umbral = 0.09                  # umbral un poco más alto para evitar drift
        self.reentrenar_min_hits = 2           # #detecciones consecutivas requeridas
        self.hit_streak = 0

        self.error_history = []  # Guarda error por timestep
        self.path_history = []  # Guarda (x, y) por timestep

        self.reentrained = False

        self._setup_world(self.cap)
        self.set_world(self.cap)               # carga modelo LSTM correspondiente

    
    def set_anomaly(self, is_anomalous: bool):
        """Activa/desactiva el mapa anómalo y resetea el mundo."""
        self.anomaly = bool(is_anomalous)
        self._setup_world(self.cap)
    
    # ---------- Utilidades bandera ----------
    

    def make_flag_wall(self, pos, ori, *, wall_half, stripes, axis='y', interior_point=(0.0, 0.0, 0.0)):
        def to_rgba(c):
            if isinstance(c, str) and c.startswith('#'):
                c = c.lstrip('#'); r=int(c[0:2],16)/255; g=int(c[2:4],16)/255; b=int(c[4:6],16)/255
                return (r,g,b,1.0)
            return c

        def local_to_world(delta_local):
            R = p.getMatrixFromQuaternion(ori)
            r00,r01,r02, r10,r11,r12, r20,r21,r22 = R
            x = r00*delta_local[0] + r01*delta_local[1] + r02*delta_local[2]
            y = r10*delta_local[0] + r11*delta_local[1] + r12*delta_local[2]
            z = r20*delta_local[0] + r21*delta_local[1] + r22*delta_local[2]
            return [pos[0] + x, pos[1] + y, pos[2] + z]

        # Muro físico sin visual (evita z-fighting)
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col,
                          baseVisualShapeIndex=-1, basePosition=pos, baseOrientation=ori)

        hx, hy, hz = wall_half
        num_stripes = len(stripes)
        eps_nrm = 1e-3
        eps_olp = 1e-4

        # Empuje hacia interior (cara vista desde el centro)
        R = p.getMatrixFromQuaternion(ori)
        n_world = [R[0], R[3], R[6]]
        to_center = [0.0 - pos[0], 0.0 - pos[1], 0.0 - pos[2]]
        dot = n_world[0]*to_center[0] + n_world[1]*to_center[1] + n_world[2]*to_center[2]
        push_sign = 1.0 if dot > 0.0 else -1.0
        eps_nrm *= push_sign

        # Tres franjas iguales
        full = 2.0 * (hy if axis == 'y' else hz)
        span = full / num_stripes

        for i, (_, col) in enumerate(stripes):
            left = i * span - (full / 2.0)
            right = (i + 1) * span - (full / 2.0)
            center = (left + right) / 2.0
            half_span = (right - left) / 2.0

            if axis == 'y':
                half_local = [hx, half_span + eps_olp, hz]
                center_local = [eps_nrm, center, 0.0]
            else:
                half_local = [hx, hy, half_span + eps_olp]
                center_local = [eps_nrm, 0.0, center]

            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_local, rgbaColor=to_rgba(col))
            stripe_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=vis,
                                          basePosition=pos, baseOrientation=ori)
            p.resetBasePositionAndOrientation(stripe_id, local_to_world(center_local), ori)

    # ---------- Mundo ----------
    def _setup_world(self, cap):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # Carga plano tras reset
        p.loadURDF("plane.urdf")

        wall_height = 10.0
        wall_thickness = 0.1
        room_size = 10
        wall_positions = [
            [room_size/2, 0, wall_height/2],
            [-room_size/2, 0, wall_height/2],
            [0, room_size/2, wall_height/2],
            [0, -room_size/2, wall_height/2]
        ]
        wall_orientations = [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            p.getQuaternionFromEuler([0, 0, math.pi/2]),
            p.getQuaternionFromEuler([0, 0, math.pi/2]),
        ]

        if cap == 0:
            sierraleone = [(1, "#04FF00"), (1, '#FFFFFF'), (1, "#0419FF")]
            self.make_flag_wall(pos=wall_positions[0], ori=wall_orientations[0],
                                wall_half=[wall_thickness/2, room_size/2, wall_height/2],
                                stripes=sierraleone, axis='z')
            if self.anomaly:
                # Variante ANÓMALA (colores cambiados)
                espania = [(1, '#7DFF7D'), (2, '#FF00A6'), (1, '#000000')]
            else:
                # Variante NORMAL (entrenamiento)
                espania = [(1, "#00D0FF"), (2, "#FFFFFF"), (1, "#440092")]
            self.make_flag_wall(pos=wall_positions[1], ori=wall_orientations[1],
                                wall_half=[wall_thickness/2, room_size/2, wall_height/2],
                                stripes=espania, axis='z')
            lithuania = [(1, "#B4B102"), (2, "#035500"), (1, "#A00404")]
            self.make_flag_wall(pos=wall_positions[2], ori=wall_orientations[2],
                                wall_half=[wall_thickness/2, room_size/2, wall_height/2],
                                stripes=lithuania, axis='z')
            alemania = [(1, '#000000'), (1, '#DD0000'), (1, '#FFCE00')]
            self.make_flag_wall(pos=wall_positions[3], ori=wall_orientations[3],
                                wall_half=[wall_thickness/2, room_size/2, wall_height/2],
                                stripes=alemania, axis='z')
        else:
            for pos, ori in zip(wall_positions, wall_orientations):
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, room_size/2, wall_height/2])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, room_size/2, wall_height/2], rgbaColor=[0.7,0.7,0.7,1])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=ori)
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.8]),
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.8], rgbaColor=[0,0,1,1]),
                                    basePosition=[3, 3, 0.8])
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_CYLINDER, radius=0.4, height=1.4),
                                    p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=1.4, rgbaColor=[0,1,0,1]),
                                    basePosition=[-3, -4, 0.6])
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=0.9),
                                    p.createVisualShape(p.GEOM_SPHERE, radius=0.9, rgbaColor=[1,0,0,1]),
                                    basePosition=[-3, 3, 0.5])
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_CYLINDER, radius=1, height=2),
                                    p.createVisualShape(p.GEOM_CYLINDER, radius=1, length=2, rgbaColor=[0,1,1,1]),
                                    basePosition=[3, -4, 0.7])
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.8]),
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.8], rgbaColor=[1,0,1,1]),
                                    basePosition=[1.5, 1.5, 0.8])
            p.createMultiBody(0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=1.2),
                                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=1.2, rgbaColor=[0,0,0,1]),
                                    basePosition=[-1.5, -1.5, 0.8])
            pass

        yaw_rad = math.radians(-90)
        robot_ori = p.getQuaternionFromEuler([0, 0, yaw_rad])
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.3], robot_ori)

    # ---------- Dinámica ----------
    def _move_robot(self, action):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]
        R = p.getMatrixFromQuaternion(ori)
        fwd = np.array([R[1], R[4], R[7]], dtype=float)
    
        step_len = 0.3
        min_clearance = 0.7
        safety = 0.1
    
        if action == 0:  # AVANZAR
            # Usa ultrasónico
            dist = self.get_ultrasonic_distance(collision_extra_rays=True, max_distance=10.0)
            # Si hay espacio suficiente, avanza; si no, se quedas en su sitio
            if dist > (min_clearance + safety):
                new_pos = np.array(pos) + fwd * step_len
            else:
                new_pos = np.array(pos)  # bloqueado: no avanza
        elif action == 1:  # GIRAR IZQ
            yaw += math.radians(30)
            new_pos = np.array(pos)
        else:              # GIRAR DER
            yaw -= math.radians(30)
            new_pos = np.array(pos)
    
        # Mantenerse siempre dentro de la sala
        room_size = 10
        wall_thickness = 0.1
        margin = 0.2
        xmin = -room_size/2 + wall_thickness/2 + margin
        xmax =  room_size/2 - wall_thickness/2 - margin
        ymin = -room_size/2 + wall_thickness/2 + margin
        ymax =  room_size/2 - wall_thickness/2 - margin
        new_pos[0] = float(np.clip(new_pos[0], xmin, xmax))
        new_pos[1] = float(np.clip(new_pos[1], ymin, ymax))
    
        new_ori = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot, new_pos.tolist(), new_ori)

    def get_ultrasonic_distance(self, collision_extra_rays=False, max_distance=10.0):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        rot = p.getMatrixFromQuaternion(ori)
        fwd = [rot[1], rot[4], rot[7]]
        right = [rot[0], rot[3], rot[6]]

        start_center = [pos[0] + fwd[0]*0.2, pos[1] + fwd[1]*0.2, pos[2] + 0.5]
        end_center   = [start_center[0] + fwd[0]*max_distance,
                        start_center[1] + fwd[1]*max_distance,
                        start_center[2] + fwd[2]*max_distance]
        result_center = p.rayTest(start_center, end_center)[0]
        distances = [result_center[2]*max_distance if result_center[0] != -1 else max_distance]

        if collision_extra_rays:
            shoulder = 0.3
            # izquierdo
            sL = [start_center[0]-right[0]*shoulder, start_center[1]-right[1]*shoulder, start_center[2]]
            eL = [sL[0]+fwd[0]*max_distance, sL[1]+fwd[1]*max_distance, sL[2]+fwd[2]*max_distance]
            rL = p.rayTest(sL, eL)[0]
            distances.append(rL[2]*max_distance if rL[0] != -1 else max_distance)
            # derecho
            sR = [start_center[0]+right[0]*shoulder, start_center[1]+right[1]*shoulder, start_center[2]]
            eR = [sR[0]+fwd[0]*max_distance, sR[1]+fwd[1]*max_distance, sR[2]+fwd[2]*max_distance]
            rR = p.rayTest(sR, eR)[0]
            distances.append(rR[2]*max_distance if rR[0] != -1 else max_distance)

        return min(distances)

    def _render_image(self):
        width = height = 224
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        R = p.getMatrixFromQuaternion(ori)
        fwd = np.array([R[1], R[4], R[7]], dtype=float)
        up  = np.array([R[2], R[5], R[8]], dtype=float)

        # --- Ajustes de cámara ---
        eye_height     = 0.8
        forward_offset = 0.25
        pitch_down_deg = 12.0

        eye = np.array(pos, dtype=float) + fwd * forward_offset
        eye[2] += eye_height

        k = math.tan(math.radians(pitch_down_deg))
        dir_vec = fwd - k * up
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-9)
        tgt = eye + dir_vec

        view = p.computeViewMatrix(eye, tgt, up)
        proj = p.computeProjectionMatrixFOV(fov=100, aspect=1.0, nearVal=0.05, farVal=20.0)

        img = p.getCameraImage(width, height, view, proj)
        rgba = img[2]
        np_img = np.reshape(rgba, (height, width, 4)).astype(np.uint8)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        return bgr_img

    def _get_embedding(self, img):
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)
            emb = self.resnet(tensor).squeeze().numpy()
        return emb

    def _cosine_error(self,y_true, y_pred, eps=1e-9):
        yt = y_true / (np.linalg.norm(y_true) + eps)
        yp = y_pred / (np.linalg.norm(y_pred) + eps)
        cos = float(np.dot(yt, yp))
        return max(0.0, 1.0 - cos)

    def _eval_anomaly(self, embedding):
        # mantiene ventana de 5
        self.embeddings_buffer.append(embedding)
        if len(self.embeddings_buffer) > 5:
            self.embeddings_buffer.pop(0)
        if len(self.embeddings_buffer) < 5:
            return None

        x_seq = np.array([
            self.embeddings_buffer[0],
            self.embeddings_buffer[1],
            self.embeddings_buffer[3],
            self.embeddings_buffer[4]
        ]).reshape(1, 4, -1)
        y_true = self.embeddings_buffer[2].reshape(1, -1)
        y_pred = self.lstm_model.predict(x_seq, verbose=0)
        error = self._cosine_error(y_true.ravel(), y_pred.ravel())
        
        error = float(np.mean((y_true - y_pred) ** 2))
        print("[ Error ]: "+str(error))

        # --- reentrenar online ---
        if error > self.umbral:
            self.hit_streak += 1
            self.anom_buffer.append((x_seq.copy(), y_true.copy()))
            if self.hit_streak >= self.reentrenar_min_hits:
                xs = np.vstack([x for x, _ in self.anom_buffer])
                ys = np.vstack([y for _, y in self.anom_buffer])
                self.reentrained = True
                self._reentrenar_lstm_online(xs, ys, epochs=1)
                self.anom_buffer.clear()
                self.hit_streak = 0
        else:
            self.hit_streak = 0
        
        return error

    # ---------- Gym API ----------
    def reset(self):
        self._setup_world(self.cap)
        self.embeddings_buffer.clear()
        self.anom_buffer.clear()
        self.current_step = 0

        img = self._render_image()
        emb = self._get_embedding(img)
        dist = self.get_ultrasonic_distance()
        obs = np.concatenate([emb, [dist]])
        return obs.astype(np.float32)

    def _final_reports(self):
        if not self.error_history or not self.path_history:
            return

        e = np.array(self.error_history, dtype=float)
        traj = np.array(self.path_history, dtype=float)

        mu  = float(e.mean())
        std = float(e.std() + 1e-9)
        z   = (e - mu) / std
        umbral_z = 3.0
        ratio_anom = float((z > umbral_z).mean())

        print(f"[Reporte] atención: media={mu:.4f}  std={std:.4f}  %anom(>3σ)={100*ratio_anom:.2f}%")

        # --- 1) Evolución de la atención ---
        plt.figure(figsize=(10,4))
        plt.plot(range(len(e)), e, label="Atención (error)")
        plt.xlabel("Paso del agente")
        plt.ylabel("Nivel de atención")
        plt.title("Evolución de la atención durante el episodio")
        plt.grid(True); plt.legend()
        try:
            os.makedirs("reportes", exist_ok=True)
            plt.savefig("reportes/atencion_vs_paso.png", dpi=150, bbox_inches="tight")
        except Exception:
            pass
        plt.show()

        # --- 2) Vista cenital coloreada por atención ---
        plt.figure(figsize=(6,6))
        sc = plt.scatter(traj[:,0], traj[:,1], c=e, s=15, cmap="viridis")
        plt.colorbar(sc, label="Atención (error)")
        plt.title("Recorrido del robot (vista cenital, coloreado por atención)")
        plt.xlabel("X (m)"); plt.ylabel("Y (m)")
        plt.axis("equal"); plt.grid(True)
        try:
            plt.savefig("reportes/recorrido_coloreado.png", dpi=150, bbox_inches="tight")
        except Exception:
            pass
        plt.show()

    def _plot_attention_and_path(self):
        plt.figure(figsize=(10,4))
        plt.plot(range(len(self.error_history)), self.error_history, label="Atención (error)")
        plt.xlabel("Paso del agente")
        plt.ylabel("Nivel de atención")
        plt.title("Evolución de la atención durante el episodio")
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- Gráfica de recorrido ---
        trayectoria = np.array(self.path_history)
        plt.figure(figsize=(6,6))
        plt.plot(trayectoria[:,0], trayectoria[:,1], marker='o', markersize=2)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Recorrido del robot (vista cenital)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def set_world(self, new_cap: int):
        self.cap = new_cap
        model_name = f"modelo_lstm_{new_cap}.h5"
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"No se encontró {model_name} en el directorio de trabajo.")
        self.lstm_model = load_model(model_name, compile=False)

    def _reentrenar_lstm_online(self, xs, ys, epochs=1):
        try:
            self.lstm_model.compile(optimizer="adam", loss="mse")
            self.lstm_model.fit(xs, ys, epochs=epochs, verbose=0)
            print(f"[Reentrenado] LSTM actualizado con {len(xs)} muestras.")
        except Exception as e:
            print("[ERROR] Reentrenando LSTM:", e)

    def step(self, action):
        self._move_robot(action)
        p.stepSimulation()
        
        dist = self.get_ultrasonic_distance()
        img  = self._render_image()
        emb  = self._get_embedding(img)
        obs  = np.concatenate([emb, [dist]])

        pos, _ = p.getBasePositionAndOrientation(self.robot)
        if self.last_pos is None:
            moved = 0.0
        else:
            moved = float(np.linalg.norm(np.array(pos[:2]) - np.array(self.last_pos[:2])))
        self.last_pos = pos

        # actualizar racha de giros
        if action in (1, 2):
            self.turn_streak += 1
        else:
            self.turn_streak = 0

        self.reentrained = False
        error = self._eval_anomaly(emb)
        reward = 0.0 if error is None else error

        # pequeño premio por avanzar de verdad
        reward += 0.03 * moved

        # penalizar giros repetidos (solo si hay espacio por delante)
        dist_ahead = self.get_ultrasonic_distance()
        if self.turn_streak >= 6 and dist_ahead > 1.0:
            reward -= 0.03  # castigo a bucles de giro

        if self.reentrained:
            reward += 0.07

        self.error_history.append(error if error is not None else 0.0)
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        self.path_history.append((pos[0], pos[1]))


        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            self._final_reports()
        return obs.astype(np.float32), float(reward), done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.physics_client)
