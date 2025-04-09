import os

folder_path = "C:/TrafficLight/dayTrain/dayClip9"
missing_files = []

for i in range(0,959 ): 
    file_name = f"dayClip9--{i:05d}.jpg".lower()
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("Missing files:")
    for f in missing_files:
        print(f)
else:
    print("All files found!")
print(f"Checking folder: {folder_path}")
