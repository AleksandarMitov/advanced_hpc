#! /usr/bin/env python
import subprocess

jobs = [
"jb1_1", "jb28_1", "jb28_2", "jb28_3", "jb_28_4",
"jm1_1", "jm28_1", "jm28_2", "jm28_3", "jm_28_4",
"js1_1", "js28_1", "js28_2", "js28_3", "js_28_4",
]

def hello_command(name, print_counter=False, repeat=10):
    """Print nice greetings."""
    for i in range(repeat):
        if print_counter:
            print i+1,
        print 'Hello, %s!' % name

if __name__ == '__main__':
    print("Starting job script:")
    for job in jobs:
        print("Submitting job: {}".format(job))
        print subprocess.check_output(["sbatch", job])
    print("DONE")
