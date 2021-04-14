import argparse
import csv
import cv2 as cv
import os

parser = argparse.ArgumentParser()
parser.add_argument('-csv', '--annotation_csv_path', help='Path of the annotations csv file.')
parser.add_argument('-i', '--images_path', help='Directory path of the images.')

BACKSPACE = 8
ESC = 27
SPACE = 32


# Loads the annotations from the csv file.
def load_annotations(filepath):
    annotations_dict = dict()

    with open(filepath, 'r', encoding='utf-8') as annotations_file:
        reader = csv.reader(annotations_file)
        next(reader)

        for annotation in reader:
            image_path = annotation[0]
            class_name = annotation[3]
            xmin = int(annotation[4])
            ymin = int(annotation[5])
            xmax = int(annotation[6])
            ymax = int(annotation[7])
            annotation_tuple = (image_path, xmin, ymin, xmax, ymax)

            if class_name in annotations_dict:
                annotations_dict[class_name].append(annotation_tuple)
            else:
                annotations_dict[class_name] = [annotation_tuple]

    return annotations_dict


def show_image(image_path, class_name, xmin, ymin, xmax, ymax):
    image = cv.imread(image_path)

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    masked_image = cv.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=5)
    cv.imshow(class_name, masked_image)
    key = cv.waitKey(1)
    return key

def main():
    args = parser.parse_args()
    annotation_csv_path = args.annotation_csv_path
    images_path = args.images_path

    annotations_dict = load_annotations(annotation_csv_path)

    print('-- INSTRUCTIONS --')
    print('Enter a valid Class Name to enter preview mode.')
    print('Enter ESC to exit preview mode.')
    print('Enter SPACE to view next image.')
    print('Enter Backspace to view previous image.')

    while True:
        class_name = input('\nEnter a class name or <E> to exit: ')
        if class_name == 'E' or class_name == 'e':
            break

        if class_name not in annotations_dict:
            print('ERROR: Wrong class name. Try again.')
            continue

        annotation_tuples = annotations_dict[class_name]
        i = 0
        total = len(annotation_tuples)

        while True:
            t = annotation_tuples[i]
            key = show_image(
                os.path.join(images_path, t[0]),
                class_name,
                t[1],
                t[2],
                t[3],
                t[4]
            )
            if key == ESC:
                break
            elif key == SPACE and i < total:
                i += 1
            elif key == BACKSPACE and i > 0:
                i -= 1
        cv.destroyAllWindows()

main()
