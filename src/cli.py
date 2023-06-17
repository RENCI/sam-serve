import yaml
import json
import argparse


from logging import getLogger

log = getLogger("")


def json_line_iterator(file_path):
    """
    Iterates over jsonline formatted file
    :param file_path:
    :return: dict of a line
    """
    with open(file_path) as stream:
        for line in stream:
            yield json.loads(line)


def append_json_to_file(data, output_path):
    """
    Appends a json line to an output file
    :param data: data to append
    :param output_path: file to append to
    :return: None
    """
    with open(output_path, 'a') as stream:
        stream.write(json.dumps(data) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotate long text offline.")
    parser.add_argument("-i", "--input-file", help="Input file path", default=None)
    parser.add_argument("-o", "--output-file", help="Output file path", default=None)
    parser.add_argument("-c", "--config-file", help="Config file path", default=None)
    parser.add_argument("-m", "--model", help="Model to run", default=None)
    args = parser.parse_args()

    config = args.config_file
    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model