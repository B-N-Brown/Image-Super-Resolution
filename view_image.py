import cv2
import sys


def main():
    img_name = str(sys.argv[1])

    img = cv2.imread('images_raw/' + img_name + '.jpg')

    cv2.imshow('sample image', img)

    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


if __name__ == "__main__":
    main()
