import open3d as o3d
import numpy as np


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
    :return: 包含元组的列表，每个元组是 (相机位姿矩阵, 相机 ID)
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
            R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
            T = np.array([tx, ty, tz])
            # 构建 4x4 变换矩阵
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            poses_with_camera_id.append((extrinsic, camera_id))
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


def visualize_cameras_and_pointcloud(poses_with_camera_id, camera_info, pointcloud_path,  point_size=2.0):
    """
    可视化相机位姿和点云
    :param poses_with_camera_id: 包含元组的列表，每个元组是 (相机位姿矩阵, 相机 ID)
    :param camera_info: 相机信息字典，键为相机 ID，值为 (宽度, 高度, 内参矩阵) 元组
    :param pointcloud_path: 点云文件路径
    :param scale_factor: 点云缩放因子
    :param point_size: 点的显示大小
    """
    geometries = []
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coordinate_frame)

    # 读取点云
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        # 对点云进行缩放
        # pcd.scale(scale_factor, center=pcd.get_center())
        geometries.append(pcd)
    except Exception as e:
        print(f"读取点云文件时出错: {e}")

    for pose, camera_id in poses_with_camera_id:
        width, height, intrinsic = camera_info.get(camera_id, (800, 600, np.eye(3)))
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=width,
                                                                       view_height_px=height,
                                                                       intrinsic=intrinsic[:3, :3],
                                                                       extrinsic=pose)
        geometries.append(cameraLines)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    # 设置点的大小
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    images_file_path = r'sparse\1\images.txt'
    cameras_file_path = r'sparse\1\cameras.txt'
    pointcloud_path = r'dense\fused_masked.ply'

    poses_with_camera_id = read_images_txt(images_file_path)
    camera_info = read_cameras_txt(cameras_file_path)

    # 可以调整点的大小
    point_size = 1

    visualize_cameras_and_pointcloud(poses_with_camera_id, camera_info, pointcloud_path, point_size)
    