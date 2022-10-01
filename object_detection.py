import numpy as np
import os
import cv2

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

assert tf.__version__.startswith("2")
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# tf.debugging.set_log_device_placement(True)
tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from absl import logging

logging.set_verbosity(logging.ERROR)


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output["output_0"]))
    scores = np.squeeze(output["output_1"])
    classes = np.squeeze(output["output_2"])
    boxes = np.squeeze(output["output_3"])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path, (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    print(results)
    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        print(ymin, xmin, ymax, xmax)
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj["class_id"])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id - 1]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id - 1], obj["score"] * 100)
        cv2.putText(
            original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


def train():

    spec = model_spec.get("efficientdet_lite0")
    csv_file = os.path.join(main_folder, "data", "dataset.csv")

    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
        csv_file
    )

    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=2,
        epochs=50,
        train_whole_model=True,
        validation_data=validation_data,
    )
    # model.evaluate(test_data)
    label_map = model.model_spec.config.label_map
    print(f"label_map: {label_map}")

    model.export(export_dir="/home/qiao/TUM_AI/grocery_object_detection/models")

    with open("myfile.txt", "w") as f:
        for key, value in label_map.items():
            f.write("%s:%s\n" % (key, value))

    model.summary()


def test():

    DETECTION_THRESHOLD = 0.3

    TEMP_FILE = os.path.join(main_folder, "test_images/1.jpg")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        TEMP_FILE, interpreter, threshold=DETECTION_THRESHOLD
    )
    detection_result_image = cv2.cvtColor(detection_result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg", detection_result_image)


if __name__ == "__main__":
    main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"
    class_file = os.path.join(main_folder, "classes.txt")
    # Load the labels into a list
    classes = []
    with open(class_file) as f:
        for line in f.readlines():
            classes.append(line.strip())

    # Define a list of colors for visualization
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

    model_path = os.path.join(main_folder, "models/model.tflite")

    train()
    test()
