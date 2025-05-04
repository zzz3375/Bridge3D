import open3d as o3d
import numpy as np
import cv2
import pycolmap
from pathlib import Path
from typing import List, Dict, Tuple
import time

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])

def read_images_txt(file_path):
    """读取images.txt文件，提取相机位姿信息"""
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
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            poses_with_camera_id.append((extrinsic, camera_id, image_id))
            i += 2
    return poses_with_camera_id

def read_cameras_txt(file_path):
    """从cameras.txt文件中读取相机内参"""
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
                intrinsics = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
            elif model == "PINHOLE":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
                intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            elif model == "SIMPLE_RADIAL":
                fx = params[0]
                cx = params[1]
                cy = params[2]
                intrinsics = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
            else:
                raise ValueError(f"不支持的相机模型: {model}")

            camera_info[camera_id] = (width, height, intrinsics)
            i += 1
    return camera_info

def setup_renderer(width: int, height: int, intrinsics: np.ndarray):
    """设置Open3D渲染器 - 兼容不同版本的窗口尺寸控制"""
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    
    # 设置相机内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    return vis, intrinsic

def render_depth(vis, pcd, extrinsic, intrinsic, width, height):
    """使用可见窗口渲染深度图 - 兼容版本"""
    # 清除之前的几何体
    vis.clear_geometries()
    
    # 添加点云
    vis.add_geometry(pcd, reset_bounding_box=True)
    
    # 将COLMAP位姿转换为Open3D格式
    open3d_extrinsic = extrinsic
    
    # 创建相机参数对象
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = open3d_extrinsic
    
    # 获取视图控制器
    view_control = vis.get_view_control()
    
    # 设置视距范围
    view_control.set_constant_z_far(1000)
    view_control.set_constant_z_near(0.1)
    
    # 强制应用相机参数
    view_control.convert_from_pinhole_camera_parameters(
        camera_params, allow_arbitrary=True)
    
    
    # 确保渲染器更新

    vis.poll_events()
    vis.update_renderer()

    
    # 捕获深度图
    depth_image = vis.capture_depth_float_buffer(do_render=True)
    depth_array = np.asarray(depth_image)
    depth_array[depth_array == np.inf] = -1
    
    # 验证尺寸
    if depth_array.shape[0] != height or depth_array.shape[1] != width:
        print(f"警告: 深度图尺寸({depth_array.shape})与请求尺寸({width}x{height})不符")
    
    return depth_array

def project_pointcloud_to_views_gpu(poses_with_camera_id: List[Tuple], 
                                  camera_info: Dict, 
                                  pointcloud_path: str, 
                                  full_depth_dir: str, 
                                  database_path: str = None,
                                  target_image_names: set = None):
    """
    使用GPU渲染管线将点云投影到每个相机视图并保存深度图
    
    参数:
        poses_with_camera_id: 相机位姿列表 (外参矩阵, 相机ID, 图像ID)
        camera_info: 相机信息字典 {相机ID: (宽度, 高度, 内参矩阵)}
        pointcloud_path: 点云文件路径
        full_depth_dir: 深度图保存目录
        database_path: COLMAP数据库路径(可选)
        target_image_names: 需要处理的特定图像名集合(可选)
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    
    # 创建输出目录
    Path(full_depth_dir).mkdir(parents=True, exist_ok=True)
    
    # 构建图像名映射
    image_name_map = {}
    if database_path:
        db = pycolmap.Database(database_path)
        try:
            images = db.read_all_images()
            image_name_map = {img.image_id: img.name for img in images}
        finally:
            db.close()
    
    # 初始化计时器
    total_time = 0
    processed_count = 0
    
    # 预分配渲染器字典 (每种相机参数一个渲染器)
    renderers = {}
    
    for pose, camera_id, image_id in poses_with_camera_id:
        # 检查是否为目标图像
        if target_image_names and image_name_map.get(image_id) not in target_image_names:
            continue
            
        width, height, intrinsics = camera_info[camera_id]
        
        # 获取或创建渲染器
        if camera_id not in renderers:
            renderer, intrinsic = setup_renderer(width, height, intrinsics)
            renderers[camera_id] = (renderer, intrinsic)
        else:
            renderer, intrinsic = renderers[camera_id]
        
        # 使用GPU渲染深度图
        start_time = time.time()
        depth_map = render_depth(renderer, pcd, pose, intrinsic, width, height)
        elapsed = time.time() - start_time
        total_time += elapsed
        processed_count += 1
        
        # 确定文件名
        if image_id in image_name_map:
            image_name = image_name_map[image_id]
            stem = Path(image_name).stem
        else:
            stem = str(image_id)
        
        # 保存深度图
        np.save(str(Path(full_depth_dir) / stem), depth_map)
    
    # 关闭所有渲染窗口
    for renderer, _ in renderers.values():
        renderer.destroy_window()
    
    # 打印性能信息
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"GPU渲染完成，共处理 {processed_count} 张图像")
        print(f"平均每张图像耗时: {avg_time:.4f} 秒")
        print(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    # 文件路径配置
    images_file_path = r'dense\sparse\images.txt'
    cameras_file_path = r'dense\sparse\cameras.txt'
    pointcloud_path = r'dense\fused_masked.ply'
    database_path = r'database.db'
    full_depth_dir = r'full_depth_maps'
    
    # 读取数据
    poses_with_camera_id = read_images_txt(images_file_path)
    camera_info = read_cameras_txt(cameras_file_path)
    
    # 设置目标图像(可选)
    target_images = {'DJI_20241123142256_0003_V.JPG'}  # 设为None则处理所有图像
    
    # 使用GPU渲染管线生成深度图
    project_pointcloud_to_views_gpu(
        poses_with_camera_id, 
        camera_info, 
        pointcloud_path, 
        full_depth_dir, 
        database_path,
        target_images
    )