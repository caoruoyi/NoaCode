import sys,os
import argparse
import time
IPYNB_FILENAME = 'elemem_behavioral_analysis.ipynb'
CONFIG_FILENAME = '.config_ipynb'

def main(argv):
    with open(CONFIG_FILENAME,'w') as f:
        f.write(' '.join(argv))
    time.sleep(1)
    
    if len(sys.argv) == 1:   
        file_name = 'all_subjects_report.html'
        print("This will time-out, so recommend running notebook to get output")
        return None
        
    else:
        # initialize parser
        parser = argparse.ArgumentParser()
        # adding optional argument
        parser.add_argument("-S", "--Subject", nargs="*", help = "type subject id that you want to generate report for")
        parser.add_argument("-s", "--Session", nargs="?", type=int, help = "type session number that you want to generate report for")
        args = parser.parse_args()
        
        # multiple subjects
        if type(args.Subject) != str:
            subjects = " ".join([sub for sub in args.Subject])
            file_name = "{}_report.html".format(subjects)

        # single subject
        else:
            # single session
            if args.Session:
                file_name = '{}_session_{}_report.html'.format(args.Subject, args.Session)
            # all sessions
            else:
                file_name = "{}_report.html".format(args.Subject)
        
        if not os.path.exists("./reports/"):
            os.makedirs("./reports/")
        
        os.system('jupyter nbconvert --execute --to html {} --no-input'.format(IPYNB_FILENAME))
        os.rename('elemem_behavioral_analysis.html', file_name)
        os.replace(file_name, os.path.join("./reports/", file_name))
        return None

main(sys.argv)