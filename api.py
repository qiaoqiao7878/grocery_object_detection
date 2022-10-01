import cv2
import pandas as pd

from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Region,
)
from msrest.authentication import ApiKeyCredentials
import os, time, uuid


def draw(prediction, image):

    image_height, image_width = image.shape[:2]

    xmin = int(prediction.bounding_box.left * image_width)
    ymin = int(prediction.bounding_box.top * image_height)

    xmax = xmin + int(prediction.bounding_box.width * image_width)
    ymax = ymin + int(prediction.bounding_box.height * image_height)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(prediction.tag_name, prediction.probability * 100)
    cv2.putText(image, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


def set_up():

    # Replace with valid values
    ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
    prediction_key = "57f09654b7bc4f539808c518155ebd70"

    # Now there is a trained endpoint that can be used to make a prediction
    prediction_credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": prediction_key}
    )
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    # Now there is a trained endpoint that can be used to make a prediction
    return predictor


def predict(predictor, image_path):
    df = pd.read_csv(csv_file)

    project_id = "a0c66b57-344b-4846-ad63-4c0b749cc389"
    publish_iteration_name = "Iteration3"
    # Open the sample image and get back the prediction results.
    with open(image_path, mode="rb") as test_data:
        results = predictor.detect_image(project_id, publish_iteration_name, test_data)

    image = cv2.imread(image_path)

    # Display the results.
    for prediction in results.predictions:
        if prediction.probability < 0.3:
            continue
        image = draw(prediction, image)
        print(
            "\t"
            + prediction.tag_name
            + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(
                prediction.probability * 100,
                prediction.bounding_box.left,
                prediction.bounding_box.top,
                prediction.bounding_box.width,
                prediction.bounding_box.height,
            )
        )

        df = df.append(
            {
                "training": "TRAIN",
                "image_path": image_path,
                "label": prediction.tag_name,
                "xmin": prediction.bounding_box.left,
                "ymin": prediction.bounding_box.top,
                "xmax": prediction.bounding_box.left + prediction.bounding_box.width,
                "ymax": prediction.bounding_box.top + prediction.bounding_box.height,
            },
            ignore_index=True,
        )

    df.to_csv(csv_file, index=False)
    save_path = os.path.join(result_folder, image_path.split("/")[-1])
    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    folder = "own_dataset"
    result_folder = "own_dataset_result"

    main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"
    csv_file = os.path.join(main_folder, "data", "dataset.csv")

    file_names = os.listdir(folder)
    predictor = set_up()
    for file_name in file_names:
        image_path = os.path.join(folder, file_name)
        predict(predictor, image_path)
