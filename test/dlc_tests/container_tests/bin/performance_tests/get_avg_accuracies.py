import sys


def get_avg_speed(filepath):
    total = 0.0
    n = 0
    with open(filepath) as f:
        for line in f:
            if "Batch" in line:
                total += float(line.split()[4])
                n += 1
    if total and n:
        return total/n
    else:
        raise ValueError("total: {}; n: {} -- something went wrong".format(total, n))


if __name__ == '__main__':
    filepath = sys.argv[1]
    avg_speed = get_avg_speed(filepath)
    print("Speed: {} samples/sec".format(avg_speed))
