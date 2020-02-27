import sys
import re

THRESHOLD = 75.0

def get_accuracies(log_file):
    # Instantiate lists for accuracies above threshold and below, respectively
    above = []
    below = []

    with open(log_file, 'r') as lf:
        for line in lf:
            if "Test accuracy " in line:
                accuracy = re.search('(\d*\.\d+)|\d+', line).group()
                accuracy = float(accuracy)
                if accuracy >= THRESHOLD:
                    above.append(accuracy)
                else:
                    below.append(accuracy)
    return above, below


if __name__ == '__main__':
    logfile = sys.argv[1]
    above_threshold, below_threshold = get_accuracies(logfile)
    if below_threshold:
        sys.exit('Failed accuracy test -- Accuracies below threshold: {}'.format(below_threshold))
    elif above_threshold:
        sys.stdout.write('Accuracies are above threshold: {}'.format(above_threshold))
    else:
        sys.exit('ERROR: Could not find accuracies.')
