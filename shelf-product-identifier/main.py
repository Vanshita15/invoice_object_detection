import os
from ultralytics import YOLO
import glob
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image
from  sklearn.neighbors import NearestNeighbors
from collections import Counter
from pathlib import Path
import argparse

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'
N_NEIGHBORS = 5
model = "dino2"  # options: "resnet18" or "dino2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--input', help='Input file path')

    # Parse the command-line arguments
    args = parser.parse_args()

    yolo_model = YOLO(MODEL_PATH)


    PATH = Path(args.input).stem
    print(f"Expected PATH: {PATH}")

    results = yolo_model.predict(
        source=args.input,
        save=True,
        save_crop=True,
        conf=0.5,
        project="data",
        name=PATH,
    )
    
    # Check what directory was actually created
    import os
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        actual_path = data_dirs[-1]  # Get the latest one
        print(f"Actual save directory: {actual_path}")
        PATH = actual_path  # Update PATH to the actual directory name

    # Get a list of image file paths using glob
    list_imgs = glob.glob(f"{DATA_PATH}/knowledge_base/crops/object/**/*.jpg")

    # Create an instance of the Img2VecResnet18 model
    if model == "dino2":
        img2vec = Img2VecDino2()
    else:
        img2vec = Img2VecResnet18()

    # Create empty lists to store classes and embeddings
    classes = []
    embeddings = []

    # Iterate over each image file
    for filename in list_imgs:
        # Open the image file
        I = Image.open(filename)

        # Get the feature vector representation of the image using img2vec.getVec()
        vec = img2vec.getVec(I)

        # Close the image file
        I.close()

        # Extract the folder path and name of the image file
        folder_path = os.path.dirname(filename)
        folder_name = os.path.basename(folder_path)

        # Append the folder name (class) and feature vector to the lists
        classes.append(folder_name)
        embeddings.append(vec)

    # Create a NearestNeighbors model and fit it with the embeddings
    model_knn = NearestNeighbors(metric='cosine', n_neighbors=N_NEIGHBORS)
    model_knn.fit(embeddings)

    # Get a list of image file paths using glob
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")

    print("Classifying images from ", f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    # Dictionary to store all predictions for summary
    all_predictions = []
    
    for IMG_DIR in list_imgs:

        # Open the target image file
        I = Image.open(IMG_DIR)
        # Get the feature vector representation of the target image
        vec = img2vec.getVec(I)
        # print(vec)
        # Close the target image file
        I.close()

        # Find the nearest neighbors and distances to the target image
        dists, idx = model_knn.kneighbors([vec])

        # Get the class labels of the nearest neighbors
        brands_nearest_neighbors = [classes[i] for i in list(idx[0])]

        # Count the occurrences of each class label
        count = Counter(brands_nearest_neighbors)

        # Get the most common class and its count
        product, n = sorted(count.items(), key=lambda item: item[1])[-1]


        crop_name = Path(IMG_DIR).stem
        
        # Store prediction for summary
        all_predictions.append({
            'crop_name': crop_name,
            'product': product,
            'confidence': n/N_NEIGHBORS
        })

        # move crop image into the corresponding folder
        if not os.path.exists(f"{DATA_PATH}/{PATH}/crops/{product}"):
            os.makedirs(f"{DATA_PATH}/{PATH}/crops/{product}")
        os.replace(IMG_DIR, f"{DATA_PATH}/{PATH}/crops/{product}/{crop_name}.jpg")

        file_path = f"{DATA_PATH}/{PATH}/predictions.txt"
        with open(file_path, "a", newline="") as file:
            # Write a row to the file
            row = f"the crop image {crop_name} is predicted as {product} with a {n/N_NEIGHBORS:.0%} probability\n"
            file.write(row)
            
        # write csv file
        with open(f"{DATA_PATH}/{PATH}/predictions.csv", "a", newline="") as file:
            file.write(f"{crop_name},{product},{n/N_NEIGHBORS:.0%}\n")
    
    # Print summary with product counts
    print("\n=== PRODUCT DETECTION SUMMARY ===")
    product_counts = Counter([pred['product'] for pred in all_predictions])
    
    for product, count in sorted(product_counts.items()):
        print(f"{product}: {count} items detected")
    
    print(f"\nTotal objects detected: {len(all_predictions)}")