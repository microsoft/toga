import os, subprocess, re, argparse
from subprocess import run
from pathlib import Path
from collections import defaultdict
from glob import glob


def close_test_harnesses(test_base_dir):
    test_harness_fs = Path(test_base_dir).rglob('*ESTest.java')

    for test_harness_f in test_harness_fs:
        with open(test_harness_f, 'a') as f:
            f.write('\n}')


def get_imports(java_file):
    imports = []
    with open(java_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('import'):
                imports += [line]
            if line.startswith('@RunWith'):
                break

    return imports


def get_test_harness(test_case_dir, output_test_dir):
    cwd = os.getcwd()
    try:
        os.chdir(test_case_dir)
        # run(f'tar -xf test_case.tar.bz2'.split())

        test_file = str(list(Path('.').rglob('*ESTest.java'))[0])

        with open('./test.txt') as f:
            test_case = f.read()

        # if harness not created yet, make it
        if not os.path.isfile(output_test_dir+'/'+test_file):
            test_dir = test_file.split('/')[0]
            run(f'cp -r {test_dir} {output_test_dir}'.split())

            with open(test_file) as f:
                test_file_txt = f.read()

            test_harness = test_file_txt.split(test_case)

            # write first half of test harness (needs '}' at end later)
            with open(output_test_dir+'/'+test_file, 'w') as f:
                f.write(test_harness[0])

        # if harness exists, check imports match
        else:
            existing_imports = set(get_imports(output_test_dir+'/'+test_file))
            new_imports = set(get_imports(test_file))

            assert new_imports == existing_imports

    except AssertionError as e:
        raise e
    except Exception as e:
        print('ERROR:', e, test_case_dir)
        raise e
    finally:
        os.chdir(cwd)
        pass

    return test_file, test_case


def aggregate_bug_tests(project, bug, bug_dir, output_bug_dir):
    cwd = os.getcwd()

    try:
        if os.path.isdir(output_bug_dir):
            run(f'rm -r {output_bug_dir}'.split())
        os.makedirs(output_bug_dir)
        os.chdir(output_bug_dir)
        
        test_name_re = re.compile(r'(\s*@Test.*public void )(\w+)(\(\).+)', re.DOTALL)

        test_names = defaultdict(lambda: defaultdict(lambda: 0))

        test_file = None
        ntests_collected = 0

        for test_case_dir in glob(bug_dir+'/*'):
            test_file, test_case = get_test_harness(test_case_dir, os.getcwd())
            
            # get test case name
            match = test_name_re.match(test_case)
            if match is None:
                print('ERROR could not match test:')
                print(test_case)
                continue
            test_pre, test_name, test_post = match.groups()
            
            # add to dict with number
            test_names[test_file][test_name] += 1
            
            # get test with unique name
            test_name_uniq = test_name + str(test_names[test_file][test_name])
            test_uniq = test_pre + test_name_uniq + test_post
            
            # append to aggregate test file
            with open(test_file, 'a') as f:
                f.write('\n'+test_uniq+'\n')

            ntests_collected += 1
                
        if test_file:
            test_base_dir = test_file.split('/')[0]
            close_test_harnesses(test_base_dir)

            tarball_name = f'{project}-{bug}b-toga.{bug}.tar.bz2'
            run(f'tar cjf {tarball_name} {test_base_dir}'.split())

            print(f'collected {ntests_collected} for {bug_dir}')

        else:
            print(f'no tests found for {bug_dir}')

    except AssertionError as e:
        raise e
    except Exception as e:
        print('ERROR:', e, bug_dir)
        raise e
    finally:
        os.chdir(cwd)


def aggregate_all_project_tests(all_projects_dir, output_dir):

    for project_dir in glob(all_projects_dir+'/*'):
        project = os.path.basename(project_dir)
        print(project_dir)

        for bug_dir in glob(project_dir+'/toga/*'):
            bug = os.path.basename(bug_dir)
            print(bug)

            output_bug_dir = f'{output_dir}/{project}/toga/{bug}'

            aggregate_bug_tests(project, bug, bug_dir, output_bug_dir)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path')
    args = parser.parse_args()

    CORPUS_DIR = os.getcwd() + '/'+args.corpus_path 
    all_projects_dir = CORPUS_DIR + '/generated_d4j_tests'
    output_dir = CORPUS_DIR + '/aggregated_d4j_tests'

    aggregate_all_project_tests(all_projects_dir, output_dir)
