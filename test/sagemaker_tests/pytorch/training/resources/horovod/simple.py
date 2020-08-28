"""Save the size, rank, and local rank to a JSON file."""
import json
import os

import horovod.torch as hvd

hvd.init()

ARTIFACT_DIRECTORY = '/opt/ml/model/'
FILENAME = 'local-rank-%s-rank-%s.json' % (hvd.local_rank(), hvd.rank())

with open(os.path.join(ARTIFACT_DIRECTORY, FILENAME), 'w+') as file:
    info = {'local-rank': hvd.local_rank(), 'rank': hvd.rank(), 'size': hvd.size()}
    json.dump(info, file)
    print(info)
