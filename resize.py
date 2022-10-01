from PIL import Image
import os

main_folder = "/home/qiao/TUM_AI/grocery_object_detection/"
dir = os.path.join(main_folder, "test_images_ori")

for file_name in os.listdir(dir):

    print("Processing %s" % file_name)
    image = Image.open(os.path.join(dir, file_name))

    output = image.resize((224, 224), Image.ANTIALIAS)

    save_path = os.path.join(main_folder, "test_image", file_name)
    output.save(save_path, "JPEG", quality=95)

    print("All done")
