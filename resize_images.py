import cv2
import sys
import os

def main():
    old_folder_name = str(sys.argv[1])
    new_folder_name = str(sys.argv[2])
    
    width = 256
    height = 256
    image_dim = (width, height)
    
    # delete existing files from new_directory
    os.mkdir(new_folder_name)
    
    #opening and reading all images from old folder
    for filename in os.listdir(old_folder_name):
        f = os.path.join(old_folder_name, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, image_dim, interpolation = cv2.INTER_AREA)
            
            new_pathname = os.path.join(new_folder_name, filename)
            cv2.imwrite(new_pathname, resized)
            
            assert (resized.shape[0] == width and resized.shape[1] == height)


if __name__ == "__main__":
    main()