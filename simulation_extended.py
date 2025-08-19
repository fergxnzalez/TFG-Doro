import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
import math
import sys

PRUEBAS = {
    "banderas": 0,
    "objetos": 1,
    "patrones": 2,
    "diferencias": 3
}

if len(sys.argv) < 2 or sys.argv[1] not in PRUEBAS:
    print("Uso: python simulation_extended.py [banderas|objetos|patrones|diferencias]")
    sys.exit(1)

cap = PRUEBAS[sys.argv[1]]
os.makedirs("frames", exist_ok=True)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.8)

p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


def make_flag_wall(pos, ori, *, wall_half, stripes, axis='y', interior_point=(0.0, 0.0, 0.0)):

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

    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col,
                      baseVisualShapeIndex=-1, basePosition=pos, baseOrientation=ori)

    hx, hy, hz = wall_half
    num_stripes = len(stripes)
    eps_nrm = 1e-3
    eps_olp = 1e-4

    # Empuje hacia interior:
    R = p.getMatrixFromQuaternion(ori)
    n_world = [R[0], R[3], R[6]]
    to_center = [interior_point[0]-pos[0], interior_point[1]-pos[1], interior_point[2]-pos[2]]
    dot = n_world[0]*to_center[0] + n_world[1]*to_center[1] + n_world[2]*to_center[2]
    push_sign = 1.0 if dot > 0.0 else -1.0
    eps_nrm *= push_sign

    # Dimensión total a repartir
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


p.resetSimulation()
# Plano base
plane = p.loadURDF("plane.urdf")
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
    make_flag_wall(
        pos=wall_positions[0],
        ori=wall_orientations[0],
        wall_half=[wall_thickness/2, room_size/2, wall_height/2],
        stripes=sierraleone,
        axis='z'
    )
    espania = [(1, '#AA151B'), (2, '#F1BF00'), (1, '#AA151B')]
    make_flag_wall(
        pos=wall_positions[1],
        ori=wall_orientations[1],
        wall_half=[wall_thickness/2, room_size/2, wall_height/2],
        stripes=espania,
        axis='z'
    )
    lithuania = [(1, "#B4B102"), (2, "#035500"), (1, "#A00404")]
    make_flag_wall(
        pos=wall_positions[2],
        ori=wall_orientations[2],
        wall_half=[wall_thickness/2, room_size/2, wall_height/2],
        stripes=lithuania,
        axis='z'
    )
    alemania = [(1, '#000000'), (1, '#DD0000'), (1, '#FFCE00')]
    make_flag_wall(
        pos=wall_positions[3],
        ori=wall_orientations[3],
        wall_half=[wall_thickness/2, room_size/2, wall_height/2],
        stripes=alemania,
        axis='z'
    )
    
else:
    plane = p.loadURDF("plane.urdf")
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
    


yaw_deg = -90
yaw_rad = math.radians(yaw_deg)
robot_ori = p.getQuaternionFromEuler([0, 0, yaw_rad])
robot = p.loadURDF("r2d2.urdf", [0, 0, 0], robot_ori)

# Detectar si se ha pulsado alguna letra y cual
def is_key_pressed():
    keys = p.getKeyboardEvents()

    if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
        key_pressed = "W"

    elif ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        key_pressed = "A"

    elif ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        key_pressed = "D"
    
    else:
        key_pressed = None

    return key_pressed

# Función sensor ultrasónico
def get_ultrasonic_distance(robot_id, collision_extra_rays=False, max_distance=10.0):
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    rot_matrix = p.getMatrixFromQuaternion(ori)
    forward = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
    right = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

    # Rayo central (desde el centro del robot)
    start_center = [pos[0] + forward[0]*0.2,
                    pos[1] + forward[1]*0.2,
                    pos[2] + 0.5]
    end_center = [start_center[0] + forward[0]*max_distance,
                  start_center[1] + forward[1]*max_distance,
                  start_center[2] + forward[2]*max_distance]

    result_center = p.rayTest(start_center, end_center)[0]
    distances = []

    if result_center[0] != -1:
        distances.append(result_center[2] * max_distance)
    else:
        distances.append(max_distance)

    if collision_extra_rays:
        shoulder_offset = 0.3  # distancia lateral desde el centro

        # Hombro izquierdo
        start_left = [start_center[0] - right[0]*shoulder_offset,
                      start_center[1] - right[1]*shoulder_offset,
                      start_center[2]]
        end_left = [start_left[0] + forward[0]*max_distance,
                    start_left[1] + forward[1]*max_distance,
                    start_left[2] + forward[2]*max_distance]

        result_left = p.rayTest(start_left, end_left)[0]
        if result_left[0] != -1:
            distances.append(result_left[2] * max_distance)
        else:
            distances.append(max_distance)

        # Hombro derecho
        start_right = [start_center[0] + right[0]*shoulder_offset,
                       start_center[1] + right[1]*shoulder_offset,
                       start_center[2]]
        end_right = [start_right[0] + forward[0]*max_distance,
                     start_right[1] + forward[1]*max_distance,
                     start_right[2] + forward[2]*max_distance]

        result_right = p.rayTest(start_right, end_right)[0]
        if result_right[0] != -1:
            distances.append(result_right[2] * max_distance)
        else:
            distances.append(max_distance)

    return min(distances)

# Cámara primera persona
def render_first_person_view(robot_id, width=320, height=240):
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    rot_matrix = p.getMatrixFromQuaternion(ori)
    forward = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
    up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
    camera_eye = [pos[0] + forward[0]*0.2, pos[1] + forward[1]*0.2, pos[2] + 0.5]
    camera_target = [camera_eye[0] + forward[0], camera_eye[1] + forward[1], camera_eye[2] + forward[2]]
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=100, aspect=width/height, nearVal=0.1, farVal=15)
    img = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgba = img[2]
    np_img = np.reshape(rgba, (height, width, 4)).astype(np.uint8)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
    return bgr_img

# Movimiento: controla dirección con teclas W,A,D
def move_robot_with_keys(robot_id, speed=0.15, turn_angle_deg=30, min_distance=0.5):
    pos, ori = p.getBasePositionAndOrientation(robot_id)

    euler = p.getEulerFromQuaternion(ori)
    yaw = euler[2]

    rot_matrix = p.getMatrixFromQuaternion(ori)
    forward = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])
    new_pos = np.array(pos)

    key_pressed = is_key_pressed()

    # Avanzar con W
    if key_pressed == "W":
        distance = get_ultrasonic_distance(robot_id,collision_extra_rays=True)
        if distance > min_distance:
            new_pos += forward * speed

    # Girar con A y D
    if key_pressed == "A":
        yaw += math.radians(turn_angle_deg)
    if key_pressed == "D":
        yaw -= math.radians(turn_angle_deg)

    new_ori = p.getQuaternionFromEuler([euler[0], euler[1], yaw])

    p.resetBasePositionAndOrientation(robot_id, new_pos.tolist(), new_ori)

# Texto distancia ultrasónico
text_id = p.addUserDebugText("Distancia: -- m", [0,0,1.5], textColorRGB=[1,1,0], textSize=1.5)
cap_str = str(cap)
frame_count = 0
while True:
    p.stepSimulation()
    move_robot_with_keys(robot)

    dist = get_ultrasonic_distance(robot)
    pos, _ = p.getBasePositionAndOrientation(robot)
    text_pos = [pos[0], pos[1], pos[2] + 1.5]
    p.addUserDebugText(f"Distancia: {dist:.2f} m", text_pos, replaceItemUniqueId=text_id, textColorRGB=[1,1,0], textSize=1.5)

    frame_img = render_first_person_view(robot)
    cv2.imshow("Vista Robot", frame_img)
    cv2.waitKey(1)

    # Guardar frame para entrenamiento
    if is_key_pressed() != None:
        cv2.imwrite(f"frames{cap_str}/frame_{frame_count:05d}.png", frame_img)
        frame_count += 1

    time.sleep(1/60)
