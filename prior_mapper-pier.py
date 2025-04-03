import subprocess
from pathlib import Path
import shutil

# 检查 database.db 文件是否存在，如果存在则删除
database_path = Path(r"pier-sparse\database.db")
# if database_path.exists():
#     database_path.unlink()

# 检查 sparse 目录是否存在，如果存在则删除
sparse_dir = Path("pier-sparse")
# if sparse_dir.exists():
#     shutil.rmtree(sparse_dir)

# 检查 sparse-aligned 目录是否存在，如果存在则删除
# sparse_aligned_dir = Path("sparse-aligned")
# if sparse_aligned_dir.exists():
#     shutil.rmtree(sparse_aligned_dir)
image_path = Path(r"pier-images")
# 设置精度变量
prior_position_std = 2

# 自动从 EXIF 数据中提取姿态先验
feature_extractor_cmd = [
    "colmap", "feature_extractor",
    "--database_path", str(database_path),
    "--image_path", str(image_path)
]
subprocess.run(feature_extractor_cmd, check=True, shell=True)

# 对于较大数据集的匹配方式
spatial_matcher_cmd = [
    "colmap", "spatial_matcher",
    "--database_path", str(database_path)
]
subprocess.run(spatial_matcher_cmd, check=True, shell=True)

# 创建 sparse 目录（如果不存在）
sparse_dir.mkdir(exist_ok=True)

# 执行先验姿态映射
mapper_cmd = [
    "colmap", "pose_prior_mapper",
    "--database_path", str(database_path),
    "--image_path", str(image_path),
    "--output_path", str(sparse_dir),]

subprocess.run(mapper_cmd, check=True, shell=True)
visulize_cmd = f"colmap gui --database_path {database_path} --image_path {image_path} --import_path {sparse_dir/"0"}"
subprocess.run(visulize_cmd, shell=True)