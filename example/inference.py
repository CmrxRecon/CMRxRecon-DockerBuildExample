import sys
import argparse

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    # TODO Load input file from input_dir and make your prediction,
    #  then output the predictions to output_dir
