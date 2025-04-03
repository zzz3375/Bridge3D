import pycolmap
from pathlib import Path
def stereo_fusion(output_path, working_folder=None):
    """
    Perform stereo fusion using PyCOLMAP.

    Args:
        database_path (str): Path to the COLMAP database file.
        image_path (str): Path to the folder containing images.
        output_path (str): Path to store the fused 3D model.
        working_folder (str, optional): Path to a working directory. Defaults to None.
    """
    # Load the reconstructed model
    # reconstruction = pycolmap.Reconstruction()
    # reconstruction.read(database_path)
    
    # Perform stereo fusion
    stereo_fusion_options = pycolmap.StereoFusionOptions(
        mask_path=r"dense\masks"
    )
    fused_points = pycolmap.stereo_fusion(
        output_path,
        workspace_path=working_folder,
        options=stereo_fusion_options

    )
    
    print(f"Stereo fusion completed. Fused model saved to {output_path}")
    return fused_points

# Example usage
if __name__ == "__main__":
    
    working_folder = "dense"  
    output_path = Path(working_folder)/"fused_masked.ply"
    
    stereo_fusion( output_path, working_folder)
