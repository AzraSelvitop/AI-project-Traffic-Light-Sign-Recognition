import os
import shutil
import pandas as pd

# Paths
test_dir = 'C:/GTSRB/Test/Images/GTSRB/Final_Test/Images'  # Current folder with all test images
output_dir = 'C:/GTSRB/Test/Organized_Images'  # Where images will be organized
csv_file = 'C:/GTSRB/Test/GTSRB_Final_Test/GT-final_test.csv'  # Ground truth file

# Read ground truth CSV
df = pd.read_csv(csv_file, sep=';')  # Adjust delimiter if needed

# Create output directories for each class
for class_id in df['ClassId'].unique():
    os.makedirs(os.path.join(output_dir, str(class_id).zfill(5)), exist_ok=True)

# Move images to respective class folders
for _, row in df.iterrows():
    src_path = os.path.join(test_dir, row['Filename'])  # Image path
    dest_path = os.path.join(output_dir, str(row['ClassId']).zfill(5), row['Filename'])
    if os.path.exists(src_path):  # Only move if the file exists
        shutil.move(src_path, dest_path)

print("Test images organized successfully!")
