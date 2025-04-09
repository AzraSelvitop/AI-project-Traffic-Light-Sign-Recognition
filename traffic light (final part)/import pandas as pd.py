import os
import pandas as pd
import cv2

annotations_base_path = 'C:/TrafficLight/Annotations/dayTrain'
images_base_path = 'C:/TrafficLight/dayTrain'


def load_daytrain_annotations():
    all_annotations = []

    #(dayClip1 to dayClip13)
    for subfolder in os.listdir(annotations_base_path):
        subfolder_path = os.path.join(annotations_base_path, subfolder)
        
        if os.path.isdir(subfolder_path):
            bulb_file = os.path.join(subfolder_path, 'frameAnnotationsBULB.csv')
            
            if os.path.exists(bulb_file):

                df = pd.read_csv(bulb_file, sep=';')
                
                df['clip_folder'] = subfolder
                all_annotations.append(df)

   
    full_annotations = pd.concat(all_annotations, ignore_index=True)
    return full_annotations

annotations = load_daytrain_annotations()
print(f"Loaded {len(annotations)} annotations from dayTrain clips.")

