import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "ctrgcn"))

import cv2
import mediapipe as mp
import numpy as np
import torch
import sys
from pathlib import Path


# Add CTR-GCN to path (hyphenated names can't be imported directly)
# the folder contains an importable package called `model` once the path is set


# Import CTR-GCN model (IDE linters may complain; ignore since path is modified at runtime)
from model.ctrgcn import Model  # type: ignore  # noqa: E402, F401, F811

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# -----------------------------
# COCO joint mapping
# -----------------------------
COCO_MAP = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28   # right ankle
]

# -----------------------------
# Normalize skeleton
# -----------------------------
def normalize_pose(seq):

    seq = np.array(seq)

    l_hip = seq[:,7]
    r_hip = seq[:,8]

    mid = (l_hip + r_hip)/2
    seq = seq - mid[:,None,:]

    scale = np.linalg.norm(l_hip-r_hip,axis=1).mean()
    if scale < 1e-6:
        scale = 1

    seq = seq/scale
    return seq

# -----------------------------
# Load CTR-GCN model
# -----------------------------
model = Model(
    num_class=100,
    num_point=25,
    num_person=1,
    in_channels=2,
    graph='graph.ntu_rgb_d.Graph',
    graph_args={}
)
model.eval()

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

pose_sequence = []

print("Press Q to run CTR-GCN inference")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(frame,
                               results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

        joints = []

        for lm in results.pose_landmarks.landmark:
            joints.append([lm.x, lm.y])

        joints = np.array(joints)

        coco_joints = joints[COCO_MAP]

        pose_sequence.append(coco_joints)

    cv2.imshow("Live Pose", frame)

    key = cv2.waitKey(1)

    # press q to run CTR-GCN
    if key & 0xFF == ord('q'):

        if len(pose_sequence) < 20:
            print("Not enough frames")
            continue

        seq = np.array(pose_sequence)

        seq = normalize_pose(seq)

        T,V,C = seq.shape

        seq = seq.transpose(2,0,1)
        seq = seq[np.newaxis,:,:,:]
        seq = seq[:,:,:,:,None]

        data = torch.tensor(seq).float()

        with torch.no_grad():
            emb = model(data)

        print("Gait embedding:",emb)

        pose_sequence = []

    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()