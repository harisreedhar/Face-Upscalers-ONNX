import os
import cv2
import numpy as np

from GPEN.GPEN import GPEN
from GFPGAN.GFPGAN import GFPGAN
from Codeformer.Codeformer import CodeFormer
from Restoreformer.Restoreformer import RestoreFormer

gpen256 = GPEN(model_path="GPEN-BFR-256.onnx", device="cpu")
gpen512 = GPEN(model_path="GPEN-BFR-512.onnx", device="cpu")

gfpganv12 = GFPGAN(model_path="GFPGANv1.2.onnx", device="cpu")
gfpganv13 = GFPGAN(model_path="GFPGANv1.3.onnx", device="cpu")
gfpganv14 = GFPGAN(model_path="GFPGANv1.4.onnx", device="cpu")

codeformer = CodeFormer(model_path="codeformer.onnx", device="cpu")

restoreformer = RestoreFormer(model_path="restoreformer.onnx", device="cpu")

image_directory = "./test_images"

enhanced_images = []

for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(image_directory, filename)
        img = cv2.imread(image_path)

        hstacked = np.hstack([
            cv2.resize(img, (512,512)),
            cv2.resize(gpen256.enhance(img), (512,512)),
            gpen512.enhance(img),
            gfpganv12.enhance(img),
            gfpganv13.enhance(img),
            gfpganv14.enhance(img),
            codeformer.enhance(img),
            restoreformer.enhance(img)
        ])

    enhanced_images.append(hstacked)

cv2.imwrite("output.jpg", np.vstack(enhanced_images))
