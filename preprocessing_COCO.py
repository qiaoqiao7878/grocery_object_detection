import json
import pandas as pd
import shutil
import os


json_path = "data/instances_train2017.json"


def extract_json():
    # Opening JSON file
    f = open(json_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    with open("images_train.json", "w") as outfile:
        json.dump(data["images"], outfile)
    with open("annotations_train.json", "w") as outfile:
        json.dump(data["annotations"], outfile)

    # Closing file
    f.close()


def copy_food_images():

    df_categories = pd.read_json("categories.json")
    df_food = df_categories[df_categories["supercategory"] == "food"]
    print(df_food)

    df_food = df_food[:1]

    for _, row in df_food.iterrows():
        food_id = row["id"]
        food_name = row["name"]

        df_images = pd.read_json("images_train.json")
        df_images.rename(columns={"id": "image_id"}, inplace=True)

        df_annotations = pd.read_json("annotations_train.json")
        df_annotations_food = df_annotations[df_annotations["category_id"] == food_id]

        df = df_annotations_food.join(df_images.set_index("image_id"), on="image_id")
        file_names = df["file_name"].drop_duplicates()

        os.makedirs(f"/home/qiao/TUM_AI/COCO/food_train/{food_name}/", exist_ok=True)

        for file_name in file_names:
            shutil.copyfile(
                f"/home/qiao/TUM_AI/train2017/{file_name}",
                f"/home/qiao/TUM_AI/COCO/food_train/{food_name}/{file_name}",
            )


if __name__ == "__main__":
    extract_json()
