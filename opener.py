# show ppm images
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="ppm file to show")
    args = parser.parse_args()

    img = Image.open(args.file)
    img.show()
