import argparse
import csv
import os


parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-csv converter")

parser.add_argument("-p",
                    "--path_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)

args = parser.parse_args()


if __name__ == '__main__':
    xml_path = args.path_dir
    fields = ["class", "name", "relative path"]
    if not os.path.exists(xml_path):
        raise FileExistsError
    with open(xml_path + '/train_csv.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                     lineterminator="\n")
        employee_writer.writerow(fields)
        for i, f in enumerate(os.listdir(xml_path)):
            if os.path.isdir(xml_path + "/" + f):
                for img_path in os.listdir(xml_path + "/" + f):
                    if img_path.endswith(('.tif', '.jpg')):
                        employee_writer.writerow([f, img_path, f + '/' + img_path])
    print('Successfully converted xml to csv.')
