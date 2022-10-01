import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd


def vis(image, xmin, ymin, xmax, ymax):
    new_image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imwrite("test.png", new_image)
    return new_image


def get_bndbox(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    bndboxs = []
    for object in root.iter("object"):
        for bndbox in object.iter("bndbox"):
            for bound in bndbox:
                if bound.tag == "xmin":
                    xmin = int(bound.text)
                elif bound.tag == "ymin":
                    ymin = int(bound.text)
                elif bound.tag == "xmax":
                    xmax = int(bound.text)
                elif bound.tag == "ymax":
                    ymax = int(bound.text)
        bndboxs.append((xmin, ymin, xmax, ymax))

    return bndboxs


if __name__ == "__main__":
    main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"
    class_file = os.path.join(main_folder, "classes.txt")
    csv_file = os.path.join(main_folder, "data", "dataset.csv")
    image_size = 224

    df = pd.read_csv(csv_file)

    image_folder = os.path.join(main_folder, "images")
    image_jpeg_folder = os.path.join(main_folder, "images_jpeg")
    annotation_folder = os.path.join(main_folder, "annotations")

    classes = []
    with open(class_file) as f:
        for line in f.readlines():
            classes.append(line.strip())

    # for food_class in classes:
    for food_class in ["chips", "chocolate", "fish", "cereal"]:
        subclass = os.path.join(image_folder, food_class)
        image_names = os.listdir(subclass)
        os.makedirs(os.path.join(image_jpeg_folder, food_class), exist_ok=True)
        for image_name in image_names[:200]:
            image = cv2.imread(os.path.join(subclass, image_name))
            image_name_without_suffix = image_name.split(".")[0]
            new_image_path = os.path.join(
                image_jpeg_folder,
                food_class,
                f"{image_name_without_suffix}.jpeg",
            )
            cv2.imwrite(new_image_path, image)
            xml = f"{image_name_without_suffix}.xml"
            xml_path = os.path.join(annotation_folder, food_class, xml)
            bndboxs = get_bndbox(xml_path)

            for bndbox in bndboxs:
                xmin, ymin, xmax, ymax = bndbox
                df = df.append(
                    {
                        "training": "TRAIN",
                        "image_path": new_image_path,
                        "label": food_class,
                        "xmin": xmin / image_size,
                        "ymin": ymin / image_size,
                        "xmax": xmax / image_size,
                        "ymax": ymax / image_size,
                    },
                    ignore_index=True,
                )

    df.to_csv(csv_file, index=False)
