import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()
    filepath = args.filepath
    for line in reversed(list(open(filepath))):
        if "took time" in line:
            print(line)
            break
