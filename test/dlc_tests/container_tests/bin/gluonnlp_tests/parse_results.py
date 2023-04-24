import sys
import re

THRESHOLD = 0.75


if __name__ == '__main__':
    logfile = sys.argv[1]

    with open(logfile, 'r') as f:
        log = f.read()

    results = re.search(r'test acc ((?:\d*\.\d+)|\d+)', log)
    if results is not None:
        accuracy = float(results.groups()[0])

        if accuracy < THRESHOLD:
            sys.exit('Failed accuracy test -- Accuracies below threshold: {}'.
                     format(results))
        else:
            sys.stdout.write(
                'Accuracies are above threshold: {}'.format(results[0]))
    else:
        sys.exit('ERROR: Could not find accuracies.')
