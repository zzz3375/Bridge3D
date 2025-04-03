import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d


def parse_transform(transform_str):
    """
    将变换矩阵的字符串表示转换为 4x4 的 numpy 数组
    """
    values = list(map(float, transform_str.split()))
    transform_matrix = np.array(values).reshape(4, 4)
    return transform_matrix


def parse_xml(file_path):
    """
    解析 XML 文件，提取每个相机的内参和外参
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 解析传感器信息获取内参
    sensor = root.find('.//sensor')
    width = int(sensor.find('resolution').get('width'))
    height = int(sensor.find('resolution').get('height'))
    calibration = sensor.find(".//calibration[@class='adjusted']")
    f = float(calibration.find('f').text)
    cx_raw = float(calibration.find('cx').text)
    cy_raw = float(calibration.find('cy').text)
    # 计算实际光心位置
    cx = width / 2 + cx_raw
    cy = height / 2 + cy_raw

    # 构建相机内参矩阵
    intrinsic = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    cameras = root.findall('.//cameras/camera')
    poses = []
    for camera in cameras:
        transform_element = camera.find('transform')
        if transform_element is not None:
            transform_str = transform_element.text
            transform_matrix = parse_transform(transform_str)
            poses.append(transform_matrix)

    return width, height, intrinsic, poses


def visualize_cameras(width, height, intrinsic, poses):
    """
    使用 Open3D 可视化相机的视野范围
    """
    geometries = []
    for pose in poses:
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=width,
            view_height_px=height,
            intrinsic=intrinsic[:3, :3],
            extrinsic=pose
        )
        geometries.append(camera_lines)

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    file_path = "camera-metashape.xml"
    # 解析 XML 文件获取相机参数
    width, height, intrinsic, poses = parse_xml(file_path)
    if poses:
        # 可视化相机视野
        visualize_cameras(width, height, intrinsic, poses)
    else:
        print("未找到相机的变换矩阵。")