import math


# VISION_RIGHT_TRANSFORM_XYZ_WXYZ = [
#     -0.153120, -0.139930, -0.109680, 0.648397, -0.269843, -0.657337, -0.273266
# ]
# VISION_LEFT_TRANSFORM_XYZ_WXYZ = [
#     0.154162, -0.137877, -0.190333, 0.654702, -0.277334, 0.647870, 0.273341
# ]

VISION_RIGHT_TRANSFORM_XYZ_WXYZ = [
    -0.152770, -0.139391, -0.109405, 0.648141, -0.270300, -0.657271, -0.273580
]
VISION_LEFT_TRANSFORM_XYZ_WXYZ = [
    0.154525, -0.138222, -0.190644, 0.654745, -0.277931, 0.647655, 0.273140
]
def xyz_wxyz_to_matrix(transform):
    """x, y, z, qw, qx, qy, qz -> 4x4 homogeneous matrix."""
    if len(transform) != 7:
        raise ValueError("transform must be [x, y, z, qw, qx, qy, qz]")

    x, y, z, qw, qx, qy, qz = [float(value) for value in transform]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")

    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    rot = [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul4(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def invert_transform(matrix):
    """Invert a rigid 4x4 transform."""
    rot = [row[:3] for row in matrix[:3]]
    trans = [matrix[i][3] for i in range(3)]
    rot_t = [[rot[j][i] for j in range(3)] for i in range(3)]
    inv_trans = [-sum(rot_t[i][j] * trans[j] for j in range(3)) for i in range(3)]
    return [
        [rot_t[0][0], rot_t[0][1], rot_t[0][2], inv_trans[0]],
        [rot_t[1][0], rot_t[1][1], rot_t[1][2], inv_trans[1]],
        [rot_t[2][0], rot_t[2][1], rot_t[2][2], inv_trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_wxyz(matrix):
    rot = [row[:3] for row in matrix[:3]]
    trace = rot[0][0] + rot[1][1] + rot[2][2]

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2][1] - rot[1][2]) / scale
        qy = (rot[0][2] - rot[2][0]) / scale
        qz = (rot[1][0] - rot[0][1]) / scale
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
        qw = (rot[2][1] - rot[1][2]) / scale
        qx = 0.25 * scale
        qy = (rot[0][1] + rot[1][0]) / scale
        qz = (rot[0][2] + rot[2][0]) / scale
    elif rot[1][1] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
        qw = (rot[0][2] - rot[2][0]) / scale
        qx = (rot[0][1] + rot[1][0]) / scale
        qy = 0.25 * scale
        qz = (rot[1][2] + rot[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
        qw = (rot[1][0] - rot[0][1]) / scale
        qx = (rot[0][2] + rot[2][0]) / scale
        qy = (rot[1][2] + rot[2][1]) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return [qw / norm, qx / norm, qy / norm, qz / norm]


def print_matrix(name, matrix):
    print(f"{name} = [")
    for row in matrix:
        print("    [" + ", ".join(f"{value: .9f}" for value in row) + "],")
    print("]")


def main():
    # T_left_camera: camera frame relative to left arm base frame.
    # T_right_camera: camera frame relative to right arm base frame.
    # T_left_right = T_left_camera * inv(T_right_camera)
    t_left_camera = xyz_wxyz_to_matrix(VISION_LEFT_TRANSFORM_XYZ_WXYZ)
    t_right_camera = xyz_wxyz_to_matrix(VISION_RIGHT_TRANSFORM_XYZ_WXYZ)
    t_left_right = matmul4(t_left_camera, invert_transform(t_right_camera))

    print_matrix("T_left_right", t_left_right)

    xyz = [t_left_right[i][3] for i in range(3)]
    quat_wxyz = matrix_to_wxyz(t_left_right)
    print()
    print("RIGHT_BASE_IN_LEFT_BASE_XYZ_WXYZ = [")
    print("    " + ", ".join(f"{value:.9f}" for value in xyz + quat_wxyz))
    print("]")


if __name__ == "__main__":
    main()
