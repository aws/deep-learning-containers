# usage: python <script_name>.py <log_file> 4 Training & Inference GPU
# usage: python <script_name>.py <log_file> 1 Inference CPU
import sys


def get_avg_speed(filepath, index):
    total = 0.0
    n = 0
    with open(filepath) as f:
        for line in f:
            if "Speed" in line:
                try:
                    total += float(line.split()[index])
                except ValueError as e:
                    raise RuntimeError(f"LINE: {line} split {line.split()[index]} ERROR: {e}")
                n += 1
    if total and n:
        return total/n
    else:
        raise ValueError(f"total: {total}; n: {n} -- something went wrong")


if __name__ == '__main__':
    filepath = sys.argv[1]
    index = int(sys.argv[2])
    avg_speed = get_avg_speed(filepath, index)
    print(f"Speed: {avg_speed} samples/sec")
