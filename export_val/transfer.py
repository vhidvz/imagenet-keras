import os
import re
import sys
import glob

import argparse


def get_lables(labels_path):
    if not os.path.exists(labels_path):
        print("Labels directory not found.", file=sys.stderr)
        exit(1)

    labels = glob.glob(labels_path + '/*.xml')
    print('Found {} labels.'.format(len(labels)))

    for label in labels:
        with open(label) as f:
            data = f.read().replace('\n', '').replace('\t', '')
            found = re.findall('<name>(n[0-9]+)</name>', data)[0]
            name = label.split('/')[-1].split('.')[0]
            yield name, found


def move_to_dataval(output_path, target_path, name, label):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(output_path + '/{}'.format(label)):
        os.mkdir(output_path + '/{}'.format(label))

    if not os.path.exists(target_path):
        print("Target directory not found.", file=sys.stderr)
        exit(1)

    if not os.path.exists(target_path + '/{}.JPEG'.format(name)):
        return

    os.rename(target_path + '/{}.JPEG'.format(name),
              output_path + '/{}/{}.JPEG'.format(label, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # TODO: Change description for program
        description="""
    description goes here...
    """)
    parser.add_argument("-l", "--labels", type=str,  # required=True,
                        default=None, help="labels directory.")
    parser.add_argument("-t", "--target", type=str,  # required=True,
                        default=None, help="target directory.")
    parser.add_argument("-o", "--output", type=str,
                        default='./dataval', help="output directory.")

    args = parser.parse_args()

    if args.labels and args.target:
        for name, label in get_lables(args.labels):
            move_to_dataval(args.output, args.target, name, label)
    else:
        parser.print_help()
        exit(1)
