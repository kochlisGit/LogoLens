import argparse
import glob
import pandas as pd
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--annotation_dir', help='Annotation Directory.')
parser.add_argument('-o', '--output_path', help='Output Path of csv file.')


# Converts masks from XML format to CSV.
# Including: Filename | Width | Height | Class | Xmin | Ymin | Xmax | Ymax
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    args = parser.parse_args()

    xml_df = xml_to_csv(args.annotation_dir)
    xml_df.to_csv(args.output_path, index=False)
    print('Successfully converted xml to csv.')


main()
