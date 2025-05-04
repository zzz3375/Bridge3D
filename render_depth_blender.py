import bpy
import mathutils
import shutil

# blender_bin = shutil.which("blender")
# if blender_bin:
#     print("Found:", blender_bin)
#     bpy.app.binary_path = blender_bin
# else:
#     print("Unable to find blender!")

import numpy as np
import cv2
import pycolmap
from pathlib import Path
from typing import List, Dict, Tuple
import time


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵 (与之前相同)"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])

def read_images_txt(file_path):
    """读取images.txt文件 (与之前相同)"""
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
    """从cameras.txt文件中读取相机内参 (与之前相同)"""
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

def setup_blender_scene(pcd_path, resolution_x, resolution_y):
    """初始化Blender场景"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 设置渲染引擎和GPU
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'
    
    # 启用所有CUDA设备并打印信息
    print("Available CUDA devices:")
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'CUDA':
            device.use = True
            print(f"- {device.name} (Enabled: {device.use})")
    
    # 设置分辨率
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.render.resolution_percentage = 100
    
    # 启用合成器节点
    bpy.context.scene.use_nodes = True
    
    # 导入点云
    bpy.ops.wm.ply_import(filepath=pcd_path)
    point_cloud = bpy.context.selected_objects[0]
    
    # 创建材质
    mat = bpy.data.materials.new(name="PointCloudMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs['Color'].default_value = (1, 1, 1, 1)
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    point_cloud.data.materials.append(mat)
    
    return point_cloud

def configure_blender_camera(intrinsics, extrinsic, width, height):
    """配置Blender相机参数"""
    # 创建相机
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.data.type = 'PERSP'

    # 设置内参
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    sensor_width = 36.0
    camera.data.lens = fx * sensor_width / width
    camera.data.sensor_width = sensor_width
    camera.data.sensor_height = sensor_width * height / width * fy / fx

    # 主点偏移
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    camera.data.shift_x = (cx - width/2) / width
    camera.data.shift_y = (cy - height/2) / height

    # 设置裁剪范围（关键修复）
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000.0

    # 坐标系转换
    blender_matrix = extrinsic.copy()
    blender_matrix[1:3, :] *= -1  # Y和Z轴翻转
    camera.matrix_world = mathutils.Matrix(blender_matrix.tolist())

    # 调试输出
    print(f"Camera location: {camera.location}")
    print(f"Camera rotation: {camera.rotation_euler}")
    
    bpy.context.scene.camera = camera
    return camera

def render_depth_with_blender(output_path, resolution_x, resolution_y):
    """使用Blender渲染深度图"""
    # 验证设备设置
    print(f"Using device: {bpy.context.scene.cycles.device}")
    
    # 设置渲染输出
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.filepath = output_path
    bpy.context.view_layer.use_pass_z = True

    # 配置合成器节点
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    nodes.clear()
    render_layer = nodes.new('CompositorNodeRLayers')
    output = nodes.new('CompositorNodeOutputFile')
    output.base_path = str(Path(output_path).parent)
    output.file_slots[0].path = Path(output_path).stem
    links.new(render_layer.outputs['Depth'], output.inputs[0])

    # 渲染并读取深度
    bpy.ops.render.render(write_still=True)
    depth_exr = bpy.data.images.load(output_path)
    depth_pixels = np.array(depth_exr.pixels)
    depth_channel = depth_pixels[0::4]  # 提取深度通道
    depth_map = depth_channel.reshape(resolution_y, resolution_x)
    
    # 清理并返回
    bpy.data.images.remove(depth_exr)
    print(f"Depth range: [{depth_map.min()}, {depth_map.max()}]")
    return depth_map

def project_pointcloud_to_views_blender(poses_with_camera_id, camera_info, pointcloud_path, full_depth_dir, database_path=None, target_image_names=None):
    """使用Blender渲染深度图的主函数"""
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
    
    # 初始化场景（只做一次）
    point_cloud = setup_blender_scene(pointcloud_path, 1024, 1024)  # 初始分辨率会被覆盖
    
    # 初始化计时器
    total_time = 0
    processed_count = 0
    
    for pose, camera_id, image_id in poses_with_camera_id:
        # 检查是否为目标图像
        if target_image_names and image_name_map.get(image_id) not in target_image_names:
            continue
            
        width, height, intrinsics = camera_info[camera_id]
        
        # 配置相机
        start_time = time.time()
        configure_blender_camera(intrinsics, pose, width, height)
        
        # 设置渲染分辨率
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        
        # 确定输出文件名
        if image_id in image_name_map:
            image_name = image_name_map[image_id]
            stem = Path(image_name).stem
        else:
            stem = str(image_id)
        
        output_path = str(Path(full_depth_dir) / f"{stem}.exr")
        
        # 渲染深度图
        depth_map = render_depth_with_blender(output_path, width, height)
        elapsed = time.time() - start_time
        total_time += elapsed
        processed_count += 1
        
        # 保存为npy格式
        np.save(str(Path(full_depth_dir) / stem), depth_map)
    
    # 打印性能信息
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"Blender渲染完成，共处理 {processed_count} 张图像")
        print(f"平均每张图像耗时: {avg_time:.4f} 秒")
        print(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    # 文件路径配置
    images_file_path = r'dense\sparse\images.txt'
    cameras_file_path = r'dense\sparse\cameras.txt'
    pointcloud_path = r'dense\fused_masked.ply'
    database_path = r'database.db'
    full_depth_dir = r'full_depth_maps_blender'
    
    # 读取数据
    poses_with_camera_id = read_images_txt(images_file_path)
    camera_info = read_cameras_txt(cameras_file_path)
    
    # 设置目标图像(可选)
    target_images = {'DJI_20241123142256_0003_V.JPG'}  # 设为None则处理所有图像
    
    # 使用Blender渲染管线生成深度图
    project_pointcloud_to_views_blender(
        poses_with_camera_id, 
        camera_info, 
        pointcloud_path, 
        full_depth_dir, 
        database_path,
        target_images
    )