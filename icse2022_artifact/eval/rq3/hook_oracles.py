from glob import glob
import json
import os
import csv
import re

'''
Given: A dataset of (method, oracle) pairs and an evosuite generated test suite.
Output: A modified evosuite test suite with the given oracle replacing the original behavior
'''

DATA_DIR = "../data/"
whitespace_re = re.compile(r'\s+')
path_re = re.compile(r"\S+\/generated\/(\S*)\/evosuite\/([0-9]+)\/(\S*)_ESTest.java")
fail_catch_extract_re = re.compile(r'try\s*{(.*;).*fail\(.*\)\s*;\s*}\s*catch', re.DOTALL)
assert_re = re.compile("assert\w*\s*\((.*)\)")

def get_prefix(test):
    m_try_catch = fail_catch_extract_re.search(test)
    m_assert = assert_re.search(test)
    loc = len(test)
    if m_try_catch:
        loc = m_try_catch.span()[0] - 32 #the 32 is for the // Undeclared exception! comment
        #print("there's a try catch at loc", loc)
        try_content = " " + m_try_catch.group(1).strip()
        #print("with try content", try_content)

        return test[0:loc] + try_content + "\n}"
    elif m_assert:
        try:
            assert m_assert #If there isn't a try catch, there should be an assertion!
        except AssertionError: 
            print("no assertion or try catch in", test) 
            sys.exit(1)
        loc = m_assert.span()[0]
        return test[0:loc] 
    else:
        return test[0:loc]

    '''
    if theres a try catch.. grab the body of the try until the "fail" call.
    Then,  add that to the prefix before the try catch and return that.

    OTHERWISE - no try. Find the line of the first assertion and return everything until then.
    '''

    #sys.exit(1) 

def eq(m1, m2):
    return whitespace_re.sub(' ', m1).strip() == whitespace_re.sub(' ', m2).strip()

def load_oracles(oracles):
    for f_oracle in glob("oracles/*.json"):
        oracles.append(json.load(open(f_oracle)))

def insert_assertion(method, assertion):
    lines = method.split("\n")
    return "\n".join(lines[0:-1] + ["      " + assertion] + ["}"])

def insert_try_catch(method):
    lines = method.split("\n")
    last_line = lines[-2]

    return "\n".join(lines[0:-2] + ["      try { "] + ["\t" + last_line] +  ["\t      fail(\"Expecting exception\");"] +  ["      } catch (Exception e) { }"] + ["}"])

def get_pairs(proj, bug_num):
    csv_file = os.path.join(DATA_DIR, f"{proj}_{bug_num}_prefix_fm_pairs.csv")
    if not os.path.exists(csv_file): return None
    pairs = []
    with open(csv_file) as f:
        reader = csv.reader(f)
        c = -1
        for row in reader:
            c += 1
            if c == 0: continue
            pairs.append(row)
            
    print(len(pairs))
    return pairs

def get_oracle(fm, oracles):
    for oracle in oracles:
        if eq(oracle["fm"], fm):
            return oracle
def main():
    GEN_TESTS_DIR = DATA_DIR+"evosuite_regression_all/"

    oracles = []
    load_oracles(oracles)

    for root, dirs, files in os.walk(GEN_TESTS_DIR):
        for f in files:
            if not f.endswith("_ESTest.java"): continue            
            full_fname = os.path.join(root, f)

            match = path_re.search(full_fname)
            print(full_fname)
            if not match:
                continue
            project = match.group(1) 
            bug_num = match.group(2) 
            class_path = match.group(3)

            pairs = get_pairs(project, bug_num)
            if not pairs:
                print("No pair file for", project, bug_num)
                continue

            for test, fm in pairs:
                oracle = get_oracle(fm, oracles)

                if not oracle:
                    print("ERROR COULDNT FIND THE ORACLE FOR", fm)
                    sys.exit(1)
                assert not (oracle["exception"] and oracle["assertion"])


                prefix = get_prefix(test)
                print(prefix)
                #sys.exit(1)

                if oracle["exception"]:
                    print(insert_try_catch(prefix))
                else:
                    assert oracle["assertion"]
                    print(insert_assertion(prefix, oracle["assertion"]))

                print("-"*100)
                #sys.exit(1)
                    
        
if __name__ == "__main__":
    main()
