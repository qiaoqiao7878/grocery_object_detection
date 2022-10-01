# Useful imports
import os

import numpy as np
import cv2

import tensorflow as tf
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader

# Import the same libs that TFLiteModelMaker interally uses
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import (
    train_lib,
)

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
    # img = tf.image.convert_image_dtype(img, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.float32)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    # resized_img = tf.cast(resized_img, dtype=tf.uint8)
    resized_img = tf.cast(resized_img, dtype=tf.float32)
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


def training(csv_file):
    # Create whichever object detector's spec you want
    spec = object_detector.EfficientDetLite4Spec(
        model_name="efficientdet-lite0",
        uri="https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1",
        hparams="",  # enable grad_checkpoint=True if you want
        model_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_execution=1,
        moving_average_decay=0,
        var_freeze_expr="(efficientnet|fpn_cells|resample_p6)",
        tflite_max_detections=25,
        strategy=None,
        tpu=None,
        gcp_project=None,
        tpu_zone=None,
        use_xla=False,
        profile=False,
        debug=False,
        tf_random_seed=111111,
        verbose=1,
    )

    # Load you datasets
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
        csv_file
    )

    # Create the object detector
    detector = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=batch_size,
        train_whole_model=True,
        validation_data=validation_data,
        epochs=epochs,
        do_train=False,
    )

    """
    From here on we use internal/"private" functions of the API,
    you can tell because the methods's names begin with an underscore
    """

    # Convert the datasets for training
    train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(
        train_data, batch_size, is_training=True
    )
    validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(
        validation_data, batch_size, is_training=False
    )

    # Get the interal keras model
    model = detector.create_model()

    # Copy what the API interally does as setup
    config = spec.config
    config.update(
        dict(
            steps_per_epoch=steps_per_epoch,
            eval_samples=batch_size * validation_steps,
            val_json_file=val_json_file,
            batch_size=batch_size,
        )
    )
    train.setup_model(model, config)  # This is the model.compile call basically
    model.summary()

    completed_epochs = 0

    """
    Here we restore the weights
    """

    # Load the weights from the latest checkpoint.
    # In my case:
    # checkpoint_dir = "/content/drive/My Drive/Colab Notebooks/checkpoints_heavy/"
    # specific_checkpoint_dir = "/content/drive/My Drive/Colab Notebooks/checkpoints_heavy/ckpt-35"
    try:
        # Option A:
        # load the weights from the last successfully completed epoch
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        # Option B:
        # load the weights from a specific checkpoint
        # latest = specific_checkpoint_dir

        completed_epochs = int(
            latest.split("/")[-1].split("-")[1]
        )  # the epoch the training was at when the training was last interrupted
        model.load_weights(latest)

        print("Checkpoint found {}".format(latest))
    except Exception as e:
        print("Checkpoint not found: ", e)

    """
    Optional step.
    Add callbacks that get executed at the end of every N 
    epochs: in this case I want to log the training results to tensorboard.
    """
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
    # callbacks = train_lib.get_callbacks(config.as_dict(), validation_ds)
    # callbacks.append(tensorboard_callback)

    """
    Train the model 
    """
    model.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=completed_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps,
        callbacks=train_lib.get_callbacks(
            config.as_dict(), validation_ds
        ),  # This is for saving checkpoints at the end of every epoch
    )

    model.save_weights(os.path.join(main_folder, "checkpoints/my_checkpoint"))

    """
    Save/export the trained model
    Tip: for integer quantization you simply have to NOT SPECIFY 
    the quantization_config parameter of the detector.export method
    """
    export_dir = os.path.join(
        main_folder, "models"
    )  # save the tflite wherever you want
    quant_config = QuantizationConfig.for_float16()  # or whatever quantization you want
    detector.model = model  # inject our trained model into the object detector
    detector.export(
        export_dir=export_dir,
        tflite_filename="model.tflite",
        quantization_config=quant_config,
    )


def test(model_path):

    DETECTION_THRESHOLD = 0.3
    test_folder = os.path.join(main_folder, "test_images")
    test_images = os.listdir(test_folder)
    for test_image in test_images:
        TEMP_FILE = os.path.join(main_folder, "test_images", test_image)

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Run inference and draw detection result on the local copy of the original file
        detection_result_image = run_odt_and_draw_results(
            TEMP_FILE, interpreter, threshold=DETECTION_THRESHOLD
        )
        detection_result_image = cv2.cvtColor(detection_result_image, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(main_folder, "test_result", test_image)
        cv2.imwrite(save_path, detection_result_image)


if __name__ == "__main__":

    main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"

    # Setup variables
    batch_size = 1  # or whatever batch size you want
    epochs = 2
    checkpoint_dir = os.path.join(
        main_folder, "checkpoints"
    )  # whatever your checkpoint directory is

    class_file = os.path.join(main_folder, "labels.txt")
    # Load the labels into a list
    classes = []
    with open(class_file) as f:
        for line in f.readlines():
            classes.append(line.strip())

    # Define a list of colors for visualization
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

    model_path = os.path.join(main_folder, "models/model.tflite")

    csv_file = os.path.join(main_folder, "data", "dataset.csv")

    # training(csv_file)
    test(model_path)
