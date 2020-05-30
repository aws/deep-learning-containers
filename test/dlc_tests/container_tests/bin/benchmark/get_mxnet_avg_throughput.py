# usage: python <script_name>.py <log_file> 4 Training & Inference GPU
# usage: python <script_name>.py <log_file> 1 Inference CPU
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filepath")
parser.add_argument("index", type=int)
args = parser.parse_args()


def get_avg_speed(filepath, index):
    total = 0.0
    n = 0
    with open(filepath) as f:
        for line in f:
            if "Speed" in line:
                try:
                    total += float(line.split()[index])
                except ValueError as e:
                    raise RuntimeError("LINE: {} split {} ERROR: {}".format(line, line.split()[index], e))
                n += 1
    if total and n:
        return total/n
    else:
        raise ValueError("total: {}; n: {} -- something went wrong".format(total, n))


if __name__ == '__main__':
    filepath = args.filepath
    index = args.index
    avg_speed = get_avg_speed(filepath, index)
    print("Throughput: {} samples/sec".format(avg_speed))
