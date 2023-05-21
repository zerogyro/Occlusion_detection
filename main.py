import argparse






def parse_args_and_config():

    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # file_config 
    parser.add_argument('--file_name', type=str, required=False, default = 'aaa',help='filename for path to processing file')

    args = parser.parse_args()

    return args
