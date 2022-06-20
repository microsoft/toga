import json, re, csv, random, math, argparse, os, random
from collections import namedtuple
from glob import glob
import pandas as pd
import numpy as np

whitespace_re = re.compile(r'\s+')
assertThrows_re = re.compile(r".*\(\s*\)\s*->\s*(.*)\s*\)", re.MULTILINE)
expected_re = re.compile(r"@Test.*(expected\s*=.*Exception.class)", re.MULTILINE)
assert_re = re.compile(r"assert[a-zA-z]*\(.*\);")
prune_asserts_re = re.compile(r"assert[a-zA-z]*\([^\)]*\);")
fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE)
isThrownBy_re = re.compile(r"isThrownBy")
empty_test_re = re.compile(r"@Test (public )?void \S*\(.*\)\s*\{\s*}", re.MULTILINE)
fail_catch_extract_re = re.compile(r'try\s*{(.*;) fail\(.*\)\s*;\s*}\s*catch', re.DOTALL)

method_name_re = re.compile(r".*\s+([\w\d\$]+)\(.*\)\s*(throws.*)?$", re.DOTALL)

throws_re = re.compile(r'throws .*Exception\s*{')


def clean(code):
    return whitespace_re.sub(' ', code).strip()


def get_block(test, start_call):
    """
    Extracts block from structure: start_call( () -> <block> )
    or start_call( <obj>::<method> )
    Looks for unmatched closing parens, indicates end of inside
    """
    test_parts = test.split(start_call)
    test_part = test_parts[1]
    paren_level = 0
    for col, char in enumerate(test_part):
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
            if paren_level == 0:
                break
    if paren_level != 0:
        print('ERROR: unable to extract normalized test correctly from:')
        print(test_part)
        print()
    block = test_part[:col]
    if '->' in block:
        block = block.split('->')[1]
    elif m := re.search(r'(\w+)::(\w+)', block):
        block = m.groups()[0] + '.' + m.groups()[1] + '()'

    return block


def normalize_assertThrows(test):
    testsplit = test.split('assertThrows')
    prefix, assertstmt = testsplit[0], testsplit[1]

    match = assertThrows_re.match(assertstmt) #, re.MULTILINE)
    throwing_stmt = match.groups()[0] + ';'

    normalized_test = prefix + throwing_stmt + ' }\n'
    return normalized_test


def normalize_expected(test):
    m = expected_re.match(test)
    expects = m.groups()[0]
    normalized = test.replace(expects, '')
    return normalized


def normalize_assertThatThrownBy(test):
    block = get_block(test, 'assertThatThrownBy')
    test_parts = test.split('assertThatThrownBy')
    normalized = test_parts[0] + block +'; }\n'
    return normalized


def normalize_isThrownBy(test):
    block = get_block(test, 'isThrownBy')
    test_parts = test.split('isThrownBy')
    normalized = test_parts[0] + block +'; }\n'

    return normalized


def normalize_fail_catch(test):
    m = fail_catch_extract_re.search(test)

    test_parts = test.split(m[0])
    block = m[1].strip()
    normalized = test_parts[0] + block +' }\n'
    return normalized


def normalize_negative(test):
    m = assert_re.search(test)

    if m:
        assert_start = m.span()[0]
        end = m.span()[1]

        last_semi = test.rfind(";", 0, assert_start)
        last_close_brace = test.rfind("}", 0, assert_start)
        last_open_brace = test.rfind("{", 0, assert_start)

        start = max([last_semi, last_open_brace, last_close_brace])

        normalized_test = test[0:start+1] + test[end:]

    else: # no asserts, already normalized?
        normalized_test = test

    return remove_assignment_rhs(normalized_test)

def remove_assignment_rhs(normalized_test):
    stripped_test = normalized_test.strip().rstrip(";} ")

    last_semi = stripped_test.rfind(";")
    last_close_brace = stripped_test.rfind("}")
    last_open_brace = stripped_test.rfind("{")

    start = max([last_semi, last_open_brace, last_close_brace]) + 1

    last_stmt = normalized_test[start:]
    if last_stmt.count("=") == 0:
        return normalized_test

    equals_idx = last_stmt.find("=") + 1

    return normalized_test[0:start] + last_stmt[equals_idx:]


def prune_asserts(tests):
    clean_tests = []
    for test in tests:
        clean_test = []
        pruned = False

        for line in test.split(";"):
            if not line: continue
            # line = line + ";"

            if not assert_re.search(line):
                clean_test += [line]
                # if "assert" in line:
                    # print("not a match", line)
            else:
                pruned = True

        clean_test = ';'.join(clean_test)

        if pruned:
            if not clean_test.endswith("}"):
                #print("adding end bracket", clean_test[-1])
                clean_test += " }"

        #ADD CLOSING BRAKET
        #REMOVE EXTRA SEMI COLON
        clean_tests.append(clean_test)

    return clean_tests


def standardize_tests(tests, methods, labels, idxs):
    tests = [clean(item) for item in tests]
    methods = [(clean(method), docstring) for (method, docstring) in methods]

    standard_tests, standard_methods, std_lbls = [], [], []

    errs = 0
    idxs_out = []
    for idx, test, (method, docstring), label in zip(idxs, tests, methods, labels):
        try:
            # get fm name
            m = method_name_re.match(method.split('{')[0])
            fm_name = m[1]
            fm_name = fm_name.capitalize()[0] + fm_name[1:]

            # standardize start of test
            test_pre, test_post = test.split('{', maxsplit=1)

            standard_test = f'public void test'+fm_name+'() {'+test_post

            standard_tests += [standard_test]
            standard_methods += [(method, docstring)]
            std_lbls += [label]
            idxs_out += [idx]

        except Exception as e:
            errs += 1

    return standard_tests, standard_methods, std_lbls, errs, idxs_out


def get_model_inputs(tests, methods, verbose=False):

    errors = 0
    missed = 0
    empty_tests = 0

    idxs = []

    normalized_tests, kept_methods, labels = [], [], []
    for i, (test, (method, docstring)) in enumerate(zip(tests, methods)):

        test = clean(test)
        method = clean(method), docstring

        try:

            if 'assertThrows' in test:
                normalized = normalize_assertThrows(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

            elif (m:=expected_re.match(test)):
                normalized = normalize_expected(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]


            elif 'assertThatThrownBy' in test:
                normalized = normalize_assertThatThrownBy(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

            elif fail_catch_re.search(test):
                normalized = normalize_fail_catch(test)

                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

            elif 'isThrownBy' in test:
                normalized = normalize_isThrownBy(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

            else:
                if ('exception' in test or 'Exception' in test):
                    missed += 1

                normalized = normalize_negative(test)


                if empty_test_re.match(normalized):
                    empty_tests += 1
                    continue

                assert('verifyException' not in test)

                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [0]


        except Exception as e:
            errors += 1
            continue

        idxs += [i]


    # standard cleanup on all tests:
    normalized_tests = [test.replace("// Undeclared exception! ", "") for test in normalized_tests]
    normalized_tests = prune_asserts(normalized_tests)

    normalized_tests, kept_methods, labels, std_errs, idxs = standardize_tests(normalized_tests, kept_methods, labels, idxs)
    errors += std_errs

    tests, methods = normalized_tests, kept_methods

    positives = len([l for l in labels if l])
    negatives = len(labels) - positives

    if verbose:
        print(f'processed {len(tests)}, extracted {len(normalized_tests)/len(tests)} tests with {positives/len(tests)} excpetions, {negatives/len(tests)} negatives, {empty_tests/len(tests)} empty tests, {errors/len(tests)} errors, {missed/len(tests)} missed')

    return normalized_tests, kept_methods, labels, idxs


