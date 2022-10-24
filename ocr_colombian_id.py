
from utils import create_directories, choose_image, infer_model, perform_ocr, save_result


if __name__ == '__main__':

    create_directories()

    img_path = choose_image()
    detection, image_size, image = infer_model(img_path)

    results, detected_rois = perform_ocr(detection, image_size, image)

    save_result(results, detected_rois)