def get_app_name() -> str:
    return "Face Mask Detector"

def get_prototext_file() -> str:
    return "models/deploy.prototxt"


def get_caffe_model() -> str:
    return "models/face_detector.caffemodel"


def get_face_mask_detector_model() -> str:
    return "models/face_mask_detector.h5"


def get_face_mask_model_classes() -> list[str]:
    return ["With Mask", "Without Mask"]


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
