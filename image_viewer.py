import os
from PIL import Image
import matplotlib.pyplot as plt

# image counts in each directory
image_counts = {}

def view_image(image_path):
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show()
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

def rename_image(root, file, count):
    # Get file extension
    file_ext = os.path.splitext(file)[1]
    # Create new file name with sequential numbering
    new_file_name = f"image_2{count:03d}{file_ext}"
    old_file_path = os.path.join(root, file)
    new_file_path = os.path.join(root, new_file_name)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} to {new_file_path}")

def iterate_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):

        # Get the top-level directory
        top_level_folder = os.path.relpath(root, root_dir).split(os.sep)[0]
        if top_level_folder not in image_counts:
            image_counts[top_level_folder] = 0
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                # view_image(image_path)  # View the image
                
                # Increment the image count for the class
                image_counts[top_level_folder] += 1
                
                # Rename the image file in sequential order
                rename_image(root, file, image_counts[top_level_folder])

if __name__ == "__main__":
    # directory = input("Enter the root directory to search for images: ")
    directory = "Images_individual"
    iterate_directory(directory)
    
    print("\nImage counts per top-level folder:")
    for folder, count in image_counts.items():
        print(f"{folder}: {count} images")
