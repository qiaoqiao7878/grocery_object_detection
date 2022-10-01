import cv2
import os

main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"
# dir = os.path.join(main_folder, "own_dataset_ori")
dir = os.path.join(main_folder, "test_images_ori")
scale_percent = 10  # percent of original size

for file_name in os.listdir(dir):

    print("Processing %s" % file_name)
    image_path = os.path.join(dir, file_name)
    image = cv2.imread(image_path)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # save_path = os.path.join(main_folder, "own_dataset", file_name)
    save_path = os.path.join(main_folder, "test_images", file_name)
    cv2.imwrite(save_path, resized)

print("All done")
