#!/usr/bin/env python3
# -----------------------------
# RealSense + ArUco + ROS 2 (SciPy quaternion)
# OpenCV 4.11.0 ArUco: use ArucoDetector + solvePnPGeneric(IPPE_SQUARE)
# Stabilize pose by:
#   1) choose best IPPE solution via reprojection error + z>0
#   2) quaternion hemisphere continuity (dot<0 => negate)
#   3) optional EMA for translation + SLERP for rotation
#   4) outlier rejection with auto-reacquire (prevents "stuck no output" when marker moves)
# -----------------------------
import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

# -----------------------------
# 1) Camera intrinsics & distortion 相机内参和畸变
# -----------------------------
K = np.array([[597.00150665,   0.0,         328.05591866],
              [  0.0,         598.22467831, 251.49384323],
              [  0.0,           0.0,           1.0]], dtype=np.float64)

dist = np.array([[0.02952984, 0.47996238, 0.00040253, 0.000753, -1.89223474]],
                dtype=np.float64)

# -----------------------------
# 2) Extrinsics: camera -> world
#    X_world = R_wc * X_cam + t_wc
# -----------------------------
R_wc = np.array([
    [0.0, 0.0,  1.0],
    [0.0, 1.0, 0.0],
    [ -1.0, 0.0, 0.0]
], dtype=np.float64)

t_wc = np.array([
    [0.05745],
    [-0.12733],
    [0.08115]
], dtype=np.float64)

# -----------------------------
# 3) ArUco settings (OpenCV 4.11.0)
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

target_ids = {23, 33}

marker_length = 0.045  # meters (确认这个长度与角点定义一致)

half_len = marker_length / 2.0
obj_points = np.array([
    [-half_len,  half_len, 0.0],
    [ half_len,  half_len, 0.0],
    [ half_len, -half_len, 0.0],
    [-half_len, -half_len, 0.0]
], dtype=np.float32)

# -----------------------------
# 4) Stabilization parameters
# -----------------------------
ENABLE_FILTER = True
ENABLE_OUTLIER_REJECT = True

# EMA/SLERP smoothing factor (0~1): 越小越稳但延迟越大
ALPHA_POS = 0.2   # translation EMA
ALPHA_ROT = 0.2   # rotation SLERP

# Outlier thresholds
MAX_POS_JUMP_M = 0.05        # 5 cm
MAX_ANG_JUMP_DEG = 10.0      # 10 degrees

# Auto reacquire: prevents "stuck with []" when marker truly moves
DROP_LIMIT = 5               # 连续丢 5 帧后自动重新锁定（可调：2~10）
drop_count = {}              # mid -> 连续丢帧次数

# State caches per marker id
last_pos = {}    # mid -> (3,1)
last_quat = {}   # mid -> (4,) xyzw


def reproj_error(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))


def solve_pnp_best_ippe(obj_pts, img_pts, K, dist):
    """
    Use solvePnPGeneric with IPPE_SQUARE and pick best solution by:
      1) prefer z>0
      2) minimal reprojection error
    Returns (rvec, tvec) or (None, None)
    """
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if (not ok) or (rvecs is None) or (len(rvecs) == 0):
        return None, None

    best_idx = None
    best_err = 1e18

    # First pass: only z>0
    for i in range(len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        if tvec[2, 0] <= 0:
            continue
        err = reproj_error(obj_pts, img_pts, rvec, tvec, K, dist)
        if err < best_err:
            best_err = err
            best_idx = i

    # Fallback: if none z>0, pick minimal error anyway
    if best_idx is None:
        best_idx = 0
        best_err = reproj_error(obj_pts, img_pts, rvecs[0], tvecs[0], K, dist)
        for i in range(1, len(rvecs)):
            err = reproj_error(obj_pts, img_pts, rvecs[i], tvecs[i], K, dist)
            if err < best_err:
                best_err = err
                best_idx = i

    return rvecs[best_idx], tvecs[best_idx]


def quat_continuity(mid, q_xyzw):
    """
    Make quaternion continuous: if dot(prev, curr)<0 => negate curr.
    """
    if mid in last_quat:
        if float(np.dot(last_quat[mid], q_xyzw)) < 0.0:
            q_xyzw = -q_xyzw
    return q_xyzw


def quat_angle_deg(q1, q2):
    """
    Quaternion angle difference in degrees.
    Both are xyzw.
    """
    d = float(np.dot(q1, q2))
    d = max(-1.0, min(1.0, abs(d)))
    ang = 2.0 * np.arccos(d)  # rad
    return float(ang * 180.0 / np.pi)


def slerp_quat(q_prev, q_curr, alpha):
    """
    SLERP between two quaternions (xyzw). alpha in [0,1].
    """
    if float(np.dot(q_prev, q_curr)) < 0.0:
        q_curr = -q_curr

    r_prev = R.from_quat(q_prev)
    r_curr = R.from_quat(q_curr)
    r_rel = r_prev.inv() * r_curr
    r_filt = r_prev * R.from_rotvec(r_rel.as_rotvec() * alpha)
    return r_filt.as_quat()


def run_camera(pub):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)

    profile = pipeline.start(config)
    try:
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()

        K = np.array([[intr.fx, 0.0, intr.ppx],
                      [0.0, intr.fy, intr.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        dist = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

        print("Using intrinsics:", intr.width, intr.height)
        print("K=\n", K)
        print("dist=", dist)

        counter = 0

        while rclpy.ok():
            counter += 1

            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())  # BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detector.detectMarkers(gray)

            msg = Float64MultiArray()
            data = []  # [x y z qx qy qz qw id] * N

            if ids is not None and len(ids) > 0:
                ids = ids.flatten().astype(int)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, mid in enumerate(ids):
                    if mid not in target_ids:
                        continue

                    img_points = corners[i].reshape(-1, 2).astype(np.float32)

                    # ---- best PnP solution ----
                    rvec_cm, t_cm = solve_pnp_best_ippe(obj_points, img_points, K, dist)
                    if rvec_cm is None:
                        continue

                    cv2.drawFrameAxes(frame, K, dist, rvec_cm, t_cm, marker_length * 0.5)

                    # -------- position in world --------
                    P_world = R_wc @ t_cm + t_wc  # 3x1

                    # -------- orientation in world --------
                    R_cm, _ = cv2.Rodrigues(rvec_cm)   # marker -> camera
                    R_wm = R_wc @ R_cm                 # marker -> world
                    q = R.from_matrix(R_wm).as_quat().astype(np.float64)  # xyzw

                    # ---- quaternion continuity ----
                    q = quat_continuity(mid, q)

                    # ---- outlier rejection + auto reacquire ----
                    if ENABLE_OUTLIER_REJECT and (mid in last_pos) and (mid in last_quat):
                        pos_jump = float(np.linalg.norm(P_world - last_pos[mid]))
                        ang_jump = quat_angle_deg(last_quat[mid], q)

                        if (pos_jump > MAX_POS_JUMP_M) or (ang_jump > MAX_ANG_JUMP_DEG):
                            drop_count[mid] = drop_count.get(mid, 0) + 1

                            # 丢几帧以抵抗偶发跳解
                            if drop_count[mid] < DROP_LIMIT:
                                continue

                            # 超过阈值：认为目标在移动/基准改变，接受当前帧作为新基准
                            drop_count[mid] = 0

                    # ---- filtering (optional) ----
                    if ENABLE_FILTER:
                        if mid in last_pos:
                            P_world = ALPHA_POS * P_world + (1.0 - ALPHA_POS) * last_pos[mid]
                        if mid in last_quat:
                            q = slerp_quat(last_quat[mid], q, ALPHA_ROT)

                    # update caches (and clear drop counter)
                    drop_count[mid] = 0
                    last_pos[mid] = P_world
                    last_quat[mid] = q

                    data.extend([
                        float(P_world[0, 0]),
                        float(P_world[1, 0]),
                        float(P_world[2, 0]),
                        float(q[0]),
                        float(q[1]),
                        float(q[2]),
                        float(q[3]),
                        float(mid)
                    ])

            msg.data = data

            if counter % 20 == 0:
                print(msg.data)
                print("*" * 40)

            if msg.data and rclpy.ok():
                pub.publish(msg)

            cv2.imshow("RealSense D435 ArUco", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        try:
            pipeline.stop()
        finally:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = rclpy.create_node("realsense_aruco_pose_pub")
        pub = node.create_publisher(Float64MultiArray, "/aruco_poses", 10)
        run_camera(pub)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
