import sys,os
import argparse
import time
IPYNB_FILENAME = 'elemem_behavioral_analysis.ipynb'
CONFIG_FILENAME = '.config_ipynb'

def main(argv):
    with open(CONFIG_FILENAME,'w') as f:
        f.write(' '.join(argv))
    
main(sys.argv)