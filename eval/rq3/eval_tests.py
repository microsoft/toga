from joblib import Parallel, delayed
import glob, os, argparse
import subprocess
from tqdm import tqdm

def run(project, gen_dir, result_dir, timeout, args, i, tot):
    cmd=f'run_bug_detection.pl -p {project} -d {gen_dir} -o {result_dir}'
    if args.v:
        print(f'running ({i}/{tot}):', cmd)
    try:
        res = subprocess.run(cmd.split(), capture_output=True, timeout=timeout)
        stdout = res.stdout.decode('utf-8')
        stderr = res.stderr.decode('utf-8')
        if args.v:
            print(stdout)
            print(stderr)
        elif 'FAILED' in stdout or 'FAILED' in stderr:
            print('failed:', cmd)
            print(stdout)
            print(stderr)
    except subprocess.TimeoutExpired:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_test_dir')
    parser.add_argument('-o', dest='result_dir', default='results')
    parser.add_argument('-t', '--timeout', dest='timeout', type=int, default=180)
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()

    gen_test_dir = args.gen_test_dir
    result_dir = args.result_dir

    if os.path.exists(result_dir):
        subprocess.run(f'rm -r {result_dir}'.split())


    projects_dirs = glob.glob(gen_test_dir + '/*')

    task_args = []
    for pd in projects_dirs:
        p = os.path.basename(pd)

        for gen_dir in glob.glob(f'{pd}/*/*'):
            if glob.glob(gen_dir+'/*.tar.bz2'):
                task_args += [(p, gen_dir, result_dir)]

    tot = len(task_args)
    tasks = [delayed(run)(*task_arg, args.timeout, args, i+1, tot) for i, task_arg in enumerate(task_args)]

    if not args.v:
        Parallel(n_jobs=-1, prefer='processes')(tqdm(tasks))
    else:
        Parallel(n_jobs=None, prefer='processes')(tqdm(tasks))


if __name__=='__main__':
    main()



