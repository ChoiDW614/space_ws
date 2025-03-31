import cv2
import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage


def ros_image_to_pil(ros_img: RosImage) -> PILImage:
    if ros_img.encoding == "rgb8":
        mode = "RGB"
    elif ros_img.encoding == "mono8":
        mode = "L"
    else:
        raise NotImplementedError(f"Encoding {ros_img.encoding} not supported")
    
    pil_img = PILImage.frombytes(mode, (ros_img.width, ros_img.height), bytes(ros_img.data))
    return pil_img

def pil_to_cv2(pil_img: PILImage) -> np.ndarray:
    cv_img = np.array(pil_img)
    if pil_img.mode == "RGB":
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def ros_to_cv2(ros_img: RosImage) -> np.ndarray:
    pil_img = ros_image_to_pil(ros_img)
    cv_img = pil_to_cv2(pil_img)
    return cv_img

