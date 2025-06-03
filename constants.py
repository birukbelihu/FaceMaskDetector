def prototxt_file() -> str:
    return "models/deploy.prototxt"


def caffe_model() -> str:
    return "models/face_detector.caffemodel"


def face_mask_detector_model() -> str:
    return "models/face_mask_detector.h5"


def face_mask_model_classes() -> list[str]:
    return ["With Mask", "Without Mask"]


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
