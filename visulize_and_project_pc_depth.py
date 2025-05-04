import open3d as o3d
import numpy as np
import cv2
import pycolmap
from pathlib import Path

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
            poses_with_camera_id.append((extrinsic, camera_id, image_id))
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


def visualize_cameras_and_pointcloud(poses_with_camera_id, camera_info, pointcloud_path, point_size=2.0):
    """
    可视化相机位姿和点云
    :param poses_with_camera_id: 包含元组的列表，每个元组是 (相机位姿矩阵, 相机 ID, 图像 ID)
    :param camera_info: 相机信息字典，键为相机 ID，值为 (宽度, 高度, 内参矩阵) 元组
    :param pointcloud_path: 点云文件路径
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

    for pose, camera_id, _ in poses_with_camera_id:
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

def get_image_id_from_database(database_path, image_name):
    """
    Query image ID from database using image name
    :param database_path: path to database file
    :param image_name: name of the image file
    :return: image ID if found, None otherwise
    """
    db = pycolmap.Database(database_path)
    try:
        # Method 1: Directly read image by name
        try:
            img = db.read_image(image_name)
            return img.image_id
        except:
            pass
        
        # Method 2: Iterate through all images
        images = db.read_all_images()
        for img in images:
            if img.name == image_name:
                return img.image_id
        return None
    finally:
        db.close()

def project_pointcloud_to_views(poses_with_camera_id, camera_info, pointcloud_path, full_depth_dir, database_path=None):
    """
    Project point cloud to each camera view and save depth maps
    :param poses_with_camera_id: list of tuples (camera pose matrix, camera ID, image ID)
    :param camera_info: dict mapping camera ID to (width, height, intrinsic matrix)
    :param pointcloud_path: path to point cloud file
    :param full_depth_dir: directory to save depth maps
    :param database_path: path to database file (optional)
    """
    # Read point cloud
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pcd.points)
    
    # Create output directory
    Path(full_depth_dir).mkdir(parents=True, exist_ok=True)
    
    # Build image name mapping if database is provided
    image_name_map = {}
    if database_path:
        db = pycolmap.Database(database_path)
        try:
            images = db.read_all_images()
            image_name_map = {img.image_id: img.name for img in images}
        finally:
            db.close()
    target_image_names = {'DJI_20241123142657_0020_V.JPG'}
    for pose, camera_id, image_id in poses_with_camera_id:
        if image_name_map[image_id] not in target_image_names:
            continue
        width, height, intrinsic = camera_info[camera_id]
        
        # Get rotation matrix and translation vector
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Project points to image plane
        projected_points, _ = cv2.projectPoints(points, R, t, intrinsic, None)
        projected_points = projected_points.squeeze()
        
        # Calculate depth (Z coordinate)
        cam_points = (R @ points.T).T + t
        depths = cam_points[:, 2]  # Z coordinate is depth
        
        # Create depth map
        depth_map = np.full((height, width), np.inf)
        
        # Filter points within image bounds
        valid = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
        
        projected_points = projected_points[valid]
        depths = depths[valid]
        
        # Round coordinates to integers
        epsilon = 1e-6  # Small offset to prevent boundary issues
        x_coords = np.round(projected_points[:, 0] - epsilon).astype(int)
        y_coords = np.round(projected_points[:, 1] - epsilon).astype(int)

        # Ensure coordinates are within valid range
        valid_pixels = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        x_coords = x_coords[valid_pixels]
        y_coords = y_coords[valid_pixels]
        depths = depths[valid_pixels]
        
        # Keep minimum depth for each pixel
        for x, y, depth in zip(x_coords, y_coords, depths):
            if depth < depth_map[y, x]:
                depth_map[y, x] = depth
        
        # Replace infinity with 0 or other background value
        depth_map[depth_map == np.inf] = -1
        
        # Determine filename
        if image_id in image_name_map:
            image_name = image_name_map[image_id]
            stem = Path(image_name).stem
        else:
            stem = str(image_id)
        
        # Save depth map
        np.save(str(Path(full_depth_dir) / stem), depth_map)

if __name__ == "__main__":
    images_file_path = r'dense\sparse\images.txt'
    cameras_file_path = r'dense\sparse\cameras.txt'
    pointcloud_path = r'dense\fused_masked.ply'
    database_path = r'database.db'  # 数据库路径
    full_depth_dir = r'full_depth_maps'  # 深度图保存目录
    Path(full_depth_dir).mkdir(parents=True, exist_ok=True)

    poses_with_camera_id = read_images_txt(images_file_path)
    camera_info = read_cameras_txt(cameras_file_path)

    # 可以调整点的大小
    point_size = 1

    # visualize_cameras_and_pointcloud(poses_with_camera_id, camera_info, pointcloud_path, point_size)
    
    # 执行点云投影并保存深度图
    import time
    t0 = time.time()
    project_pointcloud_to_views(poses_with_camera_id, camera_info, pointcloud_path, full_depth_dir, database_path)
    print(f"耗时: {time.time()-t0:.4f} 秒")