import argparse
import math
import random
import sys

_CONTENT_TYPE_CSV = 'text/csv'
_CONTENT_TYPE_JSON = 'application/json'
_CONTENT_TYPE_JSONLINES = 'application/jsonlines'

_VALID_CONTENT_TYPES = [_CONTENT_TYPE_JSON, _CONTENT_TYPE_CSV, _CONTENT_TYPE_JSONLINES]
_UNIT_FOR_PAYLOAD_SIZE = {'B': 1, 'KB': 1024, 'MB': 1048576}

def generate_json(shape, payload_size):
    one_record = _generate_json_recursively(shape)
    if payload_size:
        per_record_size = len(str(one_record))
        num_records = _get_num_records_for_json_payload(payload_size, per_record_size)
        records = []
        for record in range(0, num_records):
            records.append(one_record)
        return str(records)
    else:
        return str(one_record)


def _generate_json_recursively(shape):
    if len(shape) == 1:
        input = list(_random_input(shape[0]))
        return input
    else:
        inner_list = _generate_json_recursively(shape[1:])
        return [inner_list for _ in range(0, shape[0])]


def generate_jsonlines(shape, payload_size):
    one_row = _generate_json_recursively(shape)
    if payload_size:
        one_row_string = str(one_row)
        num_records = _get_num_records_for_json_payload(payload_size, len(one_row_string))
        return '\n'.join([one_row_string for _ in range(0, num_records)])
    else:
        return one_row


def _get_num_records_for_json_payload(payload_size, one_record_size):
    return max(int(round(float(payload_size) / one_record_size)), 1)


def generate_csv(shape, payload_size):
    # First, calculate overhead from commas.
    try:
        rows, columns = shape
    except ValueError:
        rows = 1
        columns = shape[0]
    # Override number of rows.
    if payload_size:
        # Divide by two given the assumption is that inputs are single-digit integers separate by commas and lines
        # end in newline characters.
        rows = int(math.ceil((float(payload_size) / columns / 2.0)))

    row = ','.join(map(lambda x: str(x), _random_input(columns)))
    return '\n'.join([row for _ in range(0, rows)])


def _random_input(n):
    for i in range(0, n):
        yield random.randint(0, 9)


def _map_payload_size_given_unit(payload_size, unit_of_payload):
    return payload_size * _UNIT_FOR_PAYLOAD_SIZE[unit_of_payload]


def generate_data(content_type, shape, payload_size, unit_of_payload='B'):
    assert unit_of_payload in _UNIT_FOR_PAYLOAD_SIZE.keys()
    payload_size = _map_payload_size_given_unit(payload_size, unit_of_payload)
    if content_type == _CONTENT_TYPE_JSONLINES:
        return generate_jsonlines(shape, payload_size)
    elif content_type == _CONTENT_TYPE_JSON:
        return generate_json(shape, payload_size)
    elif content_type == _CONTENT_TYPE_CSV:
        return generate_csv(shape, payload_size)
    else:
        raise ValueError('Content-type {} must be in {}'.format(content_type, _VALID_CONTENT_TYPES))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates test data of different shapes and content types.')
    parser.set_defaults(func=lambda x: parser.print_usage())

    parser.add_argument('-c', '--content-type', help='Content type to generate data for.', type=str, required=True,
                        choices=_VALID_CONTENT_TYPES)
    parser.add_argument('-s', '--shape', help='Shape of the output data. Behavior depends on content-type. '
                                              'For text/csv, "50,32" generates 50 lines with 32 columns. '
                                              'More than two values are invalid for CSV shape. '
                                              'For application/json, "32,32,3" generates a JSON object with shape. '
                                              '32,32,3. For application/jsonlines, the shape given represents a '
                                              'single instance as one nested JSON list.',
                        type=str, required=True)
    parser.add_argument('-p', '--payload-size', help='Approximate payload size. If supplied with text/csv, '
                                                     'ignores the number of rows given in shape and repeats a '
                                                     'single row until the payload is full. If supplied with '
                                                     'application/json or application/jsonlines, repeats the '
                                                     'payload with the given shape until the payload is full. '
                                                     'application/json payload is always one line. '
                                                     'Unit size defaults to the number of bytes unless --unit is set.',
                                                     type=int)
    parser.add_argument('-u', '--unit', help='Unit for payload size. One of "B", "KB", or "MB" for bytes, kilobytes, '
                                             'and megabytes, respectively', type=str,
                                             choices=_UNIT_FOR_PAYLOAD_SIZE.keys())

    args = parser.parse_args(sys.argv[1:])
    parsed_shape = list(map(lambda x: int(x), args.shape.split(',')))
    print(generate_data(args.content_type, parsed_shape, args.payload_size, args.unit))
