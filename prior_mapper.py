import subprocess
from pathlib import Path
import shutil

database_path = Path("database.db")

sparse_dir = Path("sparse")
sparse_dir.mkdir(exist_ok=True)

sparse_aligned_dir = Path("sparse-aligned")
sparse_aligned_dir.mkdir(exist_ok=True)

feature_extractor_cmd = [
    "colmap", "feature_extractor",
    "--database_path", database_path,
    "--image_path", "DJI_202411231419_071",
]
subprocess.run(feature_extractor_cmd, check=True, shell=True)

spatial_matcher_cmd = [
    "colmap", "spatial_matcher",
    "--database_path", database_path
]
subprocess.run(spatial_matcher_cmd, check=True, shell=True)

mapper_cmd = [
    "colmap", "pose_prior_mapper",
    "--database_path", database_path,
    "--image_path", "DJI_202411231419_071",
    "--output_path", sparse_dir,
]
subprocess.run(mapper_cmd, check=True, shell=True)

# 手动创建 sparse-aligned 下的子目录并执行对齐操作
# for model_path in Path("sparse").iterdir():
#     sub_dir = sparse_aligned_dir / model_path.name
#     sub_dir.mkdir(exist_ok=True)
#     sub_dir.mkdir(exist_ok=True)

#     input_path = model_path
#     output_path = sub_dir

#     print(f"Aligning model from {input_path.name} to {output_path}...")
#     model_aligner_cmd = [
#         "colmap", "model_aligner",
#         "--input_path", str(input_path),
#         "--output_path", str(output_path),
#         "--database_path", database_path,
#         "--alignment_max_error", "2",
#     ]
#     subprocess.run(model_aligner_cmd, check=True, shell=True)

# Dense 3D reconstruction
dense_dir = Path("dense")
dense_dir.mkdir(exist_ok=True)

undistortor_cmd = [
    "colmap", "image_undistorter", 
    "--image_path", "DJI_202411231419_071",
    "--input_path", Path(sparse_dir)/"1",
    "--output_path", str(dense_dir),
]
subprocess.run(undistortor_cmd, check=True, shell=True)

patch_match_stereo_cmd = [
    "colmap", "patch_match_stereo",
    "--workspace_path", dense_dir,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true",
    "--PatchMatchStereo.max_image_size", "3000",
]
subprocess.run(patch_match_stereo_cmd, check=True, shell=True)

stereo_fusion_masked_cmd = [
    "colmap", "stereo_fusion",
    "--workspace_path", dense_dir,
    "--StereoFusion.mask_path", dense_dir/"masks",
    "--output_path", dense_dir/"fused_masked.ply"
]
subprocess.run(stereo_fusion_masked_cmd, check=True, shell=True)


