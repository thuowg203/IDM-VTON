import os
import subprocess

def pifuhd_predict(input_image_path, output_dir):
    """
    Gọi script PIFuHD để tạo mesh từ ảnh input
    """
    # Đường dẫn đến script recon.py của PIFuHD
    script_path = os.path.join(os.path.dirname(__file__), "apps", "recon.py")

    command = [
        "python",
        script_path,
        "--img_path", input_image_path,
        "--out_folder", output_dir
    ]

    subprocess.run(command, check=True)
