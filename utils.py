import os
import cv2 
import time
import numpy as np
import tensorflow as tf
import pytesseract
from pytesseract import Output


def load_model():
    model_path = os.path.join('export', 'saved_model')
    PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), model_path)

    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    return detect_fn


def infer_model(img_path):
    img = cv2.imread(img_path)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    model = load_model()

    print("\nExtracting ROIs...")
    detections = model(input_tensor)

    print("\nDone Extracting ROIs")

    return detections, (image_np.shape[0], image_np.shape[1]), image_np


def perform_ocr(detections, image_size, image):
    print("\nPerforming OCR")

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
        
    detection_threshold = 0.2

    scores = list(filter(lambda x: x >detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    height, width = image_size

    ocr_result = []

    for idx, box in enumerate(boxes):
    
        roi = box*[height, width, height, width]

        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        region = cv2.resize( region, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

            
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region = cv2.GaussianBlur(region, (5,5), 0)
        region = cv2.medianBlur(region, 3)

        ret, region = cv2.threshold(region, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # dilation
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        region = cv2.dilate(region, rect_kern, iterations = 1)

        result = pytesseract.image_to_data(region,output_type=Output.DICT,config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')

        ocr_result.append(result)

    return ocr_result, classes


def save_result(ocr_result, classes):
    print("\nSaving results...")
    labels = [{'name':'placeofbirth', 'id':1}, {'name':'dateofbirth', 'id':2},{'name':'height', 'id':3},{'name':'bloodgroup', 'id':4},{'name':'sex', 'id':5},\
          {'name':'expirelocation', 'id':6},{'name':'id1', 'id':7},{'name':'id2', 'id':8},{'name':'idnumber', 'id':9},{'name':'lastnames', 'id':10}, \
          {'name':'firstnames', 'id':11}]

    detected_roi_names = []

    for class_idx in classes:
        for idx, item in enumerate(labels):
            if labels[idx]['id'] == class_idx:
                detected_roi_names.append(labels[idx]['name'])

    with open('card_information.txt', 'w') as file:
        for idx, text in enumerate(ocr_result):
            file.write(detected_roi_names[idx] +" - "+ text['text'][-1] + "\n")
    print(f'\nResults saved at {os.path.join(os.getcwd())}' )
     

def create_directories():
    print("\nCreating directories...")
    for dir in  ['input', 'output']:
        full_dir = os.path.join(os.getcwd(), dir)

        if os.path.isdir(full_dir):
            continue
        else:
            os.mkdir(full_dir)


def choose_image():
    input_dir = os.path.join(os.getcwd(), 'input')
    files = os.listdir(input_dir)

    print("\nloading files from 'input' folder...\n")
    time.sleep(2)

    for idx, item in enumerate(files):
        print(f'{idx + 1} - {item}')
        time.sleep(0.4)

    chosen_file = input("\npick a file by typing in the file number position\n\n")
    image = ''

    # print(files[int(chosen_file) - 1])
    while(True):
        try:
            if chosen_file == 'q':
                print('Quiting program...')
                break
            else:
                image = files[int(chosen_file) - 1]
                image = os.path.join(input_dir, image)
                break
        except:
            chosen_file = input("Enter a correct number or 'q' to quit")

    return image
