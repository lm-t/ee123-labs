import argparse, sys
import compression
from PIL import Image
import numpy as np
import pywt

def main():
    # define the program description
    text = "This is the EE123 Final Project. It can send and recieve images"
    # initiate the parser
    parser = argparse.ArgumentParser(description = text)
    # add arguments
    parser.add_argument("--send", "-s", help="set image to send")
    parser.add_argument("--recieve", "-r", help="recieve image")
    parser.add_argument("--frequency", "-f", nargs='?', const=443670000, type=int, help="set frequency to transmit. (default= 443670000)")
    # read arguments from the command line
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("no options selected. use --help for usage")
        sys.exit()

    elif args.send and not args.recieve:
        try:
            image =  Image.open(args.send)
        except IOError:
            image = None

        if image == None:
            print("Error: could not find file", args.send)
            sys.exit()

        #compressing image to it's Discrete Wavelet Transform
        #dwt_coeff = compression.extract_rgb_coeff(image)
        width, height = image.size
        #dwt_image = compression.img_from_dwt_coeff(dwt_coeff, width, height)
        #dwt_image.save('test.png')

        down = compression.downsample(image, 3)
        down.save('test_down.jpg')
        print("sending %s" % args.send)

    elif args.recieve:
        print("recieving ...")

if __name__ == '__main__':
    main()
