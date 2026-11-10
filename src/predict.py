import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf

from src.data import CLASS_NAMES, IMG_SIZE

MODEL_PATH = "models/best_model.keras"
TEST_DIR = "data/Test"
TEST_CSV = "data/test.csv"
OUTPUT_CSV = "submission.csv"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_df = pd.read_csv(TEST_CSV)
    image_ids = test_df['ID'].astype(str).tolist()
    print(f"Test images to predict: {len(image_ids)}")

    idx_to_cls = {0: "cow", 1: "elephant", 2: "horse", 3: "spider"}

    predictions = []
    for image_id in image_ids:
        path = os.path.join(TEST_DIR, image_id)
        if not os.path.exists(path):
            for ext in ['.jpg', '.jpeg', '.png']:
                potential = os.path.join(TEST_DIR, image_id + ext)
                if os.path.exists(potential):
                    path = potential
                    break
            else:
                raise FileNotFoundError(f"Image {image_id} not found")

        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)

        probs = model.predict(img, verbose=0)[0]
        predicted_class = CLASS_NAMES[int(tf.argmax(probs))]
        predictions.append(predicted_class)

    submission = pd.DataFrame({"ID": image_ids, "CLASS": predictions})
    submission.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {OUTPUT_CSV} with {len(submission)} rows")
    print(submission.head(10))


if __name__ == "__main__":
    main()