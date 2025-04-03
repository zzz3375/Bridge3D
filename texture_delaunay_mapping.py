import open3d as o3d
import numpy as np
import cv2


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    :param q: 四元数 [qw, qx, qy, qz]
    :return: 3x3 旋转矩阵
    """
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return R


def read_images_txt(file_path):
    """
    读取 images.txt 文件，提取相机位姿信息和相机 ID
    :param file_path: images.txt 文件路径
    :return: 包含元组的列表，每个元组是 (相机位姿矩阵, 相机 ID, 图像文件名)
    """
    poses_with_camera_id = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#'):
                i += 1
                continue
            parts = lines[i].strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
            T = np.array([tx, ty, tz])
            # 构建 4x4 变换矩阵
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            poses_with_camera_id.append((extrinsic, camera_id, image_name))
            i += 2
    return poses_with_camera_id


def read_cameras_txt(file_path):
    """
    从 cameras.txt 文件中读取相机内参、宽度和高度
    :param file_path: cameras.txt 文件路径
    :return: 一个字典，键为相机 ID，值为 (宽度, 高度, 内参矩阵) 元组
    """
    camera_info = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#'):
                i += 1
                continue
            parts = lines[i].strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "SIMPLE_PINHOLE":
                fx = params[0]
                cx = params[1]
                cy = params[2]
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fx, cy],
                    [0, 0, 1]
                ])
            elif model == "PINHOLE":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            elif model == "SIMPLE_RADIAL":
                fx = params[0]
                cx = params[1]
                cy = params[2]
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fx, cy],
                    [0, 0, 1]
                ])
            else:
                raise ValueError(f"不支持的相机模型: {model}")

            camera_info[camera_id] = (width, height, intrinsics)
            i += 1
    return camera_info


def texture_mapping(poses_with_camera_id, camera_info, pointcloud_path, image_folder):
    """
    进行纹理映射
    :param poses_with_camera_id: 包含元组的列表，每个元组是 (相机位姿矩阵, 相机 ID, 图像文件名)
    :param camera_info: 相机信息字典，键为相机 ID，值为 (宽度, 高度, 内参矩阵) 元组
    :param pointcloud_path: 点云文件路径
    :param image_folder: 图像文件夹路径
    :return: 带有纹理的三角网格
    """
    # 读取点云
    mesh = o3d.io.read_triangle_mesh(pointcloud_path)
    points = np.asarray(mesh.vertices)
    num_points = len(points)
    textures = np.zeros((num_points, 3), dtype=np.float64)

    for pose, camera_id, image_name in poses_with_camera_id:
        width, height, intrinsic = camera_info[camera_id]
        image_path = f"{image_folder}/{image_name}"
        image = cv2.imread(image_path)

        for i, point in enumerate(points):
            # 将点从世界坐标系转换到相机坐标系
            point_homogeneous = np.append(point, 1)
            point_camera = np.dot(np.linalg.inv(pose), point_homogeneous)[:3]
            # 投影到图像平面
            point_image = np.dot(intrinsic, point_camera)
            u = int(point_image[0] / point_image[2])
            v = int(point_image[1] / point_image[2])

            # 检查投影点是否在图像范围内
            if 0 <= u < width and 0 <= v < height:
                texture = image[v, u]
                textures[i] = texture / 255.0

    mesh.vertex_colors = o3d.utility.Vector3dVector(textures)
    return mesh


if __name__ == "__main__":
    images_file_path = r'sparse\1\images.txt'
    cameras_file_path = r'sparse\1\cameras.txt'
    pointcloud_path = r'sparse\1\Dense\fused.ply'
    image_folder = 'DJI_202411231419_071'

    poses_with_camera_id = read_images_txt(images_file_path)
    camera_info = read_cameras_txt(cameras_file_path)

    textured_mesh = texture_mapping(poses_with_camera_id, camera_info, pointcloud_path, image_folder)

    # 可视化结果
    o3d.visualization.draw_geometries([textured_mesh])
