import os
import pandas as pd
import cv2

annotations_base_path = 'C:/TrafficLight/Annotations/dayTrain'
images_base_path = 'C:/TrafficLight/dayTrain'

def load_daytrain_annotations():
    all_annotations = []

    # (dayClip1 to dayClip13)
    for subfolder in os.listdir(annotations_base_path):
        subfolder_path = os.path.join(annotations_base_path, subfolder)
        
        if os.path.isdir(subfolder_path):
            bulb_file = os.path.join(subfolder_path, 'frameAnnotationsBULB.csv')
            
            if os.path.exists(bulb_file):
                df = pd.read_csv(bulb_file, sep=';')
                df['clip_folder'] = subfolder
                df['Filename'] = df['Filename'].apply(lambda x: os.path.basename(x))
                all_annotations.append(df)

    full_annotations = pd.concat(all_annotations, ignore_index=True)
    return full_annotations

def visualize_annotations(base_image_path, annotations):
    for index, row in annotations.iterrows():

        image_path = os.path.join(base_image_path, row['clip_folder'], row['Filename'])
        
        if not os.path.exists(image_path):
            print(f"Warning: File {image_path} not found.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}.")
            continue
        
        # Drawing the bounding box on the image
        x1, y1 = int(row['Upper left corner X']), int(row['Upper left corner Y'])
        x2, y2 = int(row['Lower right corner X']), int(row['Lower right corner Y'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, row['Annotation tag'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            print("Exit key pressed. Closing window.")
            break

    cv2.destroyAllWindows()

# Load annotations
annotations = load_daytrain_annotations()
filtered_annotations = annotations[annotations['clip_folder'] == 'dayClip5']
visualize_annotations(images_base_path, filtered_annotations)
print(f"Loaded {len(annotations)} annotations from dayTrain clips.")

# Visualize 
visualize_annotations(images_base_path, filtered_annotations)
