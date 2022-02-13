#!/usr/bin/env python
# coding: utf-8

import javalang
import argparse
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from glob import glob

VALID_ASSERTS = set(['assertEquals', 'assertNull', 'assertNotNull', 'assertTrue', 'assertFalse'])


def check_assertions(assertions, v):
    parser_errs = []
    grammar = []
    non_grammar = []

    counts = defaultdict(int)

    for assertion in tqdm(assertions):
        tokens = javalang.tokenizer.tokenize(assertion)
        parser = javalang.parser.Parser(tokens)

        try:
            assert_obj = parser.parse_primary()
        except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration) as e:
            #raise e
            parser_errs += [assertion]
            if v:
                print('parser_err')
            counts['parser_err'] += 1
            continue

        if 'hamcrest' in assertion:
            counts['unsupported'] += 1
            continue

        if assert_obj.member not in VALID_ASSERTS:
            non_grammar += [assertion]
            if v:
                print('non member assert method')
            counts['wrong_assert_name'] += 1
            continue

        if (assert_obj.member == 'assertEquals'):
            if (len(assert_obj.arguments) != 2):
                non_grammar += [assertion]
                if v:
                    print('non member too many args')
                counts['wrong_arg_cnt'] += 1
                continue
            else:
                types = []
                for node in assert_obj.arguments:
                    types += [type(node)]
                if len(types) == 2:
                    if (javalang.tree.Literal not in types) and (javalang.tree.MemberReference not in types):
                        if v:
                            print('non member no literal/var in equals')
                        counts['equals_no_var'] += 1
                        non_grammar += [assertion]
                        continue
        
        grammar += [assertion]
        counts['grammar'] += 1

    return grammar, non_grammar, counts, len(parser_errs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help="verbose mode")
    args = parser.parse_args()

    if not "ATLAS_PATH" in os.environ:
        print("Set your ATLAS_PATH!")
        sys.exit(1)

    ATLAS_PATH = os.environ["ATLAS_PATH"]

    assertion_files = glob(os.path.join(ATLAS_PATH, "Datasets", "Raw_Dataset", "*", "assertLines.txt"))
    assert len(assertion_files) == 3 #There should be an assertLines for each split (Train, Eval, Test)

    total_assertions, cant_parse = 0, 0
    all_grammar, all_non_grammar, all_counts = [], [], []
    for assertion_file in assertion_files:
        with open(assertion_file) as f:
            assertions = f.read().split('\n')
     

        total_assertions += len(assertions)
        grammar, non_grammar, counts, parse_err = check_assertions(assertions, args.v)
        all_grammar += grammar
        all_non_grammar += non_grammar
        all_counts += counts
        cant_parse += parse_err


    print('total assertions', total_assertions)
    print('cant parse', cant_parse)
    print('grammar', len(all_grammar))
    print('non grammar', len(all_non_grammar))
    print('grammar ratio', len(all_grammar)/(total_assertions-cant_parse))
    print()
        
    if args.v:
        for k, v in all_counts.items():
            print(k, v)

if __name__=='__main__':
    main()

