## Purpose
This directory house DLC license attribution generation process.

### Table of content

| Sub directory | Description |
|-----------|--------------|
| linux_packages | It is used to generate license attribution file for linux pacakages using dpkg command  |
| python_packages | It is used to generate license attribution file for python pacakages using pip-licenses command  |
| piplicenses | It is used to generate intermidiate JSON file to be used by generate_licenses. It scans package folders and their dist folders for their license attribution files |
| generate_licenses | It is used to generate license attribution text file from JSON file generated from piplicenses |
| datafiles | It has all data (manually added) needed if JSON file has UNKNOWN fields. Repositories are not always tagged as per version number and attribution information may change as per version so automation is not yet used to check out repository and get data using apis. |
