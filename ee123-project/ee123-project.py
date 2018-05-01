#!/usr/bin/python
import argparse, sys

def main():
    # define the program description
    text = "This is the EE123 Final Project. It can send and recieve images"
    # initiate the parser
    parser = argparse.ArgumentParser(description = text)
    # add arguments
    parser.add_argument("--send", "-s", help="set image to send")
    parser.add_argument("--recieve", "-r", help="recieve image")
    parser.add_argument("--frequency", "-f", help="set frequency to transmit. default= 443670000")
    # read arguments from the command line
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        print("no options selected. use --help for usage")
        sys.exit()
    elif args.send and not args.recieve:
        print("sending %s" % args.send)
    elif args.recieve:
        print("recieving ...")

if __name__ == '__main__':
    main()
