import os, sys, re, tqdm, csv, random
import javalang
import numpy as np
import pandas as pd
from collections import defaultdict

TEST = True
DBG = True
TEMPLATE_CHECK = False


# global broken_tests_list = []

whitespace_re = re.compile(r'\s+')

split_re = re.compile("\"<AssertPlaceHolder>\" ;[ }]*")

long_re = re.compile(r"assertEquals\(([-]?[0-9]+)L(,.*)")
neg_re = re.compile(r"assertEquals\(\((-\d+)\)(.*)")
cast_re = re.compile("(assertEquals\()\(\w+\)(\S*, \w+\))")
paren_re = re.compile("(assertEquals\()\((\S+)\)(, \w+\))")
equals_bool_re = re.compile(r"assertEquals\((false|true), (.*)\)")
assert_fail_re = re.compile("expected:\<(.*)\> but was:\<(.*)\>")
test_name_re = re.compile("public void (test[0-9]*)\(\)")

var_re = re.compile(r"^\w+$")
method_call_re = re.compile(r"^\w+\.\w+\(\)$")

assert_re = re.compile("assert\w*\(.*\)")

arg_re = re.compile("assert\w* \( (.*) \)")
arg_re_EVO = re.compile("assert\w*\((.*)\)")
arg_re_generic = re.compile("assert\w*\s*\((.*)\)")

vocab = None

TEMPLATES = ("assertTrue", "assertFalse", "assertEquals", "assertNotNull", "assertNull")
ASSERT_TYPES = TEMPLATES

generated_templates = defaultdict(int)

errs = defaultdict(int)

def atlas_spacing(string):
    string = clean(string)
    tok_chars = '()[].@{}=+-*!|%<>,'
    
    atlas_string = []
    in_string = False
    for c in string:
        if c == '"':
            in_string = not in_string
            
        if not in_string and c in tok_chars:
            atlas_string += [' '+c+' ']
        else:
            atlas_string += [c]
            
    atlas_string = clean(''.join(atlas_string))
    
    return atlas_string


def is_int(string):
    if string[0] == '(' and string[-1] == ')':
        string = string[1:-1]
    if string[-1] == 'L':
        string = string[:-1]
    try:
        int(string)
    except:
        return False
    return True


def is_float(string):
    if string[0] == '(' and string[-1] == ')':
        string = string[1:-1]
    if string[-1] in ('F', 'D'):
        string = string[:-1]
    try:
        float(string)
    except:
        return False
    return True



def is_string(string):
    string = string.strip()
    return string[0] in ('"', '\'') and string[-1] in ('"', '\'')


def get_args(assertion):
    # strip outer parens
    args_str = assertion[assertion.find('(')+1:assertion.rfind(')')]

    # split on nonstring ,
    in_str = False
    args, prev = [], 0
    for i, c in enumerate(args_str):
        if c == '"':
            if not (i >= 1 and args_str[i-1] != '\\'):
                in_str = not in_str
        if c == ',':
            args += [ args_str[prev:i] ]
            prev = i+1
    args += [ args_str[prev:].strip() ]

    return args


def clean(code):
    return whitespace_re.sub(' ', code).strip()

def parser_type_to_java_type(t):
    try:
        t = t.value if "value" in dir(t) else t.member
    except AttributeError:
        return None


    if t == "true" or t == "false":
        return bool
    try:
        t = int(t)
        return int
    except ValueError:
        try:
            t = float(t)
            return float
        except ValueError:
            return str

def get_type_info_evo(assertion, focal_method, test_method, vocab=vocab):
    
    try:
        tokens = javalang.tokenizer.tokenize(focal_method)
        parser = javalang.parser.Parser(tokens)
        focal_method_node = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration,\
            javalang.tokenizer.LexerError):

        errs["cant-parse3"] += 1

        if DBG:
            broken_tests_list = [('parse2', test_method, focal_method, assertion)]
            for err, tm, fm, asrt in broken_tests_list:
                print('ERROR', err)
                print(fm)
                print()
                print(tm)
                print()
                print(asrt)
                print('-'*50)



        #print(focal_method, "couldn't be parsed")
        #print("-"*100)
        #print()
        #print("cant parse")
        return None

    start = test_method.find("\"<AssertPlaceHolder>\" ;") 
    clean_test = test_method[0:start] + " }"
    
    tokens = javalang.tokenizer.tokenize(clean_test)
    parser = javalang.parser.Parser(tokens)
    try:
        test_method_node = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration,\
            javalang.parser.LexerError):
        print(clean_test, "couldn't be parsed")
        print("-"*100)
        print()

        errs["cant-parse4"] += 1
        #print("cant parse")
        return None


    assert_args = get_args(assertion)

    # some sanity checks on assert args:
    if len(assert_args) == 1:
        assert_var = assert_args[0]
    else:
        assert_var = assert_args[1]

    if not (var_re.match(assert_var) or method_call_re.match(assert_var)):
        return None

    if assertion.split('(')[0] not in ASSERT_TYPES:
        return None

    # prefer assert type based on evosuite-generated assertion
    if assertion.startswith('assertTrue') or assertion.startswith('assertFalse'):
        return_type = 'boolean'

    elif is_int(assert_args[0]):
        return_type = 'int'

    elif is_float(assert_args[0]):
        return_type = 'float'

    elif is_string(assert_args[0]):
        return_type = 'String'

    elif isinstance(focal_method_node, javalang.tree.ConstructorDeclaration):
        return_type = focal_method_node.name

    elif focal_method_node and focal_method_node.return_type:
        return_type = focal_method_node.return_type.name

    else:
        #print(focal_method, "couldn't get a return type")
        #print("-"*100)
        #print()

        errs["no-return"] += 1
        if DBG:
            broken_tests_list = [('return', test_method, focal_method, assertion)]
            for err, tm, fm, asrt in broken_tests_list:
                print('ERROR', err)
                print(fm)
                print()
                print(tm)
                print()
                print(asrt)
                print('-'*50)

        #print("cant parse")
        return None

    # if assertion.endswith('.length)'):
        # return_type = 'int'

    # override type if there is a secondary getter on focal method return obj
    # TODO replace with parser
    # assert_args = get_args(assertion)
    # print('args', assert_args)
    # if is_int(assert_args[0]):
        # return_type = 'int'
    # if 'assertTrue' in assertion or 'assertFalse' in assertion:
        # return_type = 'boolean'
    # print(return_type)



    if return_type == "boolean":
        _type = bool
    # elif return_type == "String" or return_type == "char":
    elif return_type == "String":
        _type = str
    elif return_type == "double" or return_type == "float":
        _type = float
    # elif return_type == "int" or return_type == "long" or return_type == "short":
    elif return_type == "int":
        _type = int
    else:
        #print(return_type)
        _type = return_type
        # return None

    #print(focal_method[0:50], _type)

    matching_type_vars = []
    all_var_types = []
    all_vars = []
    for path, node in test_method_node:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            name = node.declarators[0].name
            all_var_types += [pretty_type(node.type)]
            all_vars += [node.declarators[0].name]

        elif isinstance(node, javalang.tree.Literal):
            all_var_types += [parser_type_to_java_type(node.value)]
            all_vars += [node.value]

    
    same_type_vars = []
    for var, t in zip(all_vars, all_var_types):
        if t == _type:
            #print("MATCHING TYPE",_type)
            same_type_vars += [var]


    #PARSE ASSERTION
    tokens = javalang.tokenizer.tokenize(assertion)
    parser = javalang.parser.Parser(tokens)

    try:
        assertion_obj = parser.parse_primary()
    except javalang.parser.JavaSyntaxError:
        print(assertion, "can't be parsed")
        print("-"*100)
        print()

        errs["cant-parse5"] += 1
        return False

    
    arg_ind = 0 if len(assertion_obj.arguments) == 1 else 1
    return _type, arg_ind, len(assertion_obj.arguments), same_type_vars
    


def get_type(assertion_type, assertion, arg, full_test):

    #print("ARG", arg)
    full_test = "public void " + full_test.replace("\"<AssertPlaceHolder>\" ;", assertion + " ; ")

    #print(full_test)
    tokens = javalang.tokenizer.tokenize(full_test)
    parser = javalang.parser.Parser(tokens)
    try:
        test_obj = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration):
        errs["cant-parse6"] += 1
        print(full_test, "\ncan't be parsed")
        print("-"*100)
        print()

        return None
    #print("TEST OBJECT")
    #print(test_obj)

    target_type = None

    if assertion_type == "assertTrue" or assertion_type == "assertFalse":
        target_type = bool

    elif assertion_type == "assertEquals":
        target_type =  parser_type_to_java_type(arg)

    if assertion_type == "assertNotNull" or assertion_type == "assertNull" and isinstance(arg, javalang.tree.MethodInvocation):
        return None

    all_var_types = []
    all_vars = []
    for p, node in test_obj:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            name = node.declarators[0].name
            '''
            if not "member" in dir(arg):
                errs["cant-parse"] += 1
                return None
            '''

            if "member" in dir(arg) and name == arg.member and not target_type:
                target_type = pretty_type(node.type)
            else:
                all_var_types += [pretty_type(node.type)]
                all_vars += [node.declarators[0].name]

        elif isinstance(node, javalang.tree.Literal):
            all_var_types += [parser_type_to_java_type(node.value)]
            all_vars += [node.value]


    if not target_type:
        errs["misc"] += 1
        return None

    same_type_vars = []
    for var, _type in zip(all_vars, all_var_types):
        if _type == target_type:
            #print("MATCHING TYPE",_type)
            same_type_vars += [var]

    #print("SAME TYPE VARS", same_type_vars) 
    return target_type, same_type_vars

def pretty_type(tree, full_type=""):
    if isinstance(tree, javalang.tree.BasicType):
        return tree.name

    if tree.sub_type:
        return tree.name + "." + pretty_type(tree.sub_type, full_type)
    return tree.name
        

def get_type_info(assertion, test_method):
    start = 0

    if not test_method: 
        errs["cant-parse1"] += 1
        return None

    #end = assertion.find(" (")
    end = assertion.find("(")


    assertion_type = assertion[start:end].strip()
    #print(assertion_type)
    if not assertion_type in TEMPLATES:
        #print(assertion_type, "not a template")
        errs["non-template-assertion-type"] += 1
        return False

    #assertion = assertion.replace("categories", "\"0\"")
    tokens = javalang.tokenizer.tokenize(assertion)
    parser = javalang.parser.Parser(tokens)

    try:
        assertion_obj = parser.parse_primary()
    except javalang.parser.JavaSyntaxError:
        print(assertion, "can't be parsed")
        errs["cant-parse2"] += 1
        return False

    if len(assertion_obj.arguments) > 2: 
        errs["more-than-2-args"] += 1
        return False

    #IF there is only 1 arg -> then use first arg
    #OTHERWISE, find the method invocation
     
    relevant_arg = None
    other_arg = None
    total_args = len(assertion_obj.arguments)
    arg_num = -1
    if len(assertion_obj.arguments) == 1:
        relevant_arg = assertion_obj.arguments[0]
    else:
        for arg_idx, arg in enumerate(assertion_obj.arguments):
            if isinstance(arg, javalang.tree.MethodInvocation):
                relevant_arg = arg
                arg_num = arg_idx
            else:
                other_arg = arg

    if not relevant_arg:
        errs["non-typeable-arg"] += 1
        return False

    #last_arg = assertion_obj.arguments[-1]
    #print(assertion)
    if not other_arg: other_arg = relevant_arg
    #print(relevant_arg)

    out = get_type(assertion_type, assertion, other_arg, test_method)
    if not out: return False

    _type, matching_type_vars = out

    #print("TYPE", _type)
    return _type, arg_num, total_args, matching_type_vars


def assertion_to_arg(assertion, arg_num, total_args):
    #m = arg_re.search(assertion)
    # m = arg_re_EVO.search(assertion)
    m = arg_re_generic.search(assertion)
    #m = arg_re.search(assertion)
    #m = arg_re_EVO.search(assertion)

    g = m.group(1)
    args = g.split(",")
    try:
        # assert len(args) == total_args and total_args <= 2 and len(args) > arg_num 
        assert len(args) == total_args and len(args) > arg_num 
    except AssertionError as e:
        if total_args == 1:
            return g
        else:
            errs["unable_to_extract_assert_args"] += 1

            # print()
            # print("unable_to_extract_assert_args")
            # print(assertion)
            # print()

            

            raise e

    return args[arg_num]


def is_bug_finding_assertion(trace):
    is_assertion_bug = False
    if 'AssertionFailedError' in trace:
        trace_lines = trace.split('\n')
        if len(trace_lines) < 2:
            return is_assertion_bug
        assertion_error = trace_lines[1]

        if 'junit.framework.AssertionFailedError:' in assertion_error:
            assertion_error = assertion_error.split(':', 1)[1]
        elif 'junit.framework.AssertionFailedError' == assertion_error.strip():
            assertion_error = trace_lines[3].strip().split('at org.junit.Assert.')[1]

        if 'expected:' in assertion_error:
            is_assertion_bug = True
        elif 'assertTrue' in assertion_error or 'assertFalse' in assertion_error:
            is_assertion_bug = True
        elif 'assertNull' in assertion_error or 'assertNotNull' in assertion_error:
            is_assertion_bug = True

    return is_assertion_bug


def gen_variants(_type, arg, matching_type_vars, vocab=vocab):
    
    out = []
    values = matching_type_vars 
    arg = arg.strip()
    if _type in vocab: 
        top_values = list(vocab[_type].keys())
        if _type == int:
            top_values = [int(x.replace("(", "").replace(")","").replace(" ","")) for x in top_values][:-1]

        elif _type == float:
            top_values = [0.0, 1.0]
        elif _type == str:
            top_values = [] # atlas strings not relevant for evosuite
            # top_values += ["'"+k+"'" for k in vocab[_type].keys()]
            #top_values = [float(x) for x in top_values]

        values = top_values + values

    for var in values:
        if var == arg:
            continue
        if TEST:
            out += ["assertEquals({}, {})".format(var, arg)]
        else:
            out += ["assertEquals ( {} , {} )".format(var, arg)]

    if _type == bool:
        if TEST:
            out +=  ["assertTrue({})".format(arg), "assertFalse({})".format(arg)]
        else:
            out +=  ["assertTrue ( {} )".format(arg), "assertFalse ( {} )".format(arg)]
    elif not _type == int and not _type == float:
        # TODO is it worth including assertNull for bug finding?
        if TEST:
            out += ["assertNotNull({})".format(arg)]
            # out += ["assertNotNull({})".format(arg), "assertNull({})".format(arg)]
        else:
            out += ["assertNotNull ( {} )".format(arg)]
            # out += ["assertNotNull ( {} )".format(arg), "assertNull ( {} )".format(arg)]
    
    return list(set(out))


def consolidate_vars(test_prefix, assertion):

    # TODO cleanup
    if not (assertion.startswith('assertEquals') and
        (assertion.endswith('int0)') or 
        assertion.endswith('int1)') or 
        assertion.endswith('long0)'))):
        return test_prefix, assertion

    args = get_args(assertion)
    if (len(args) == 1):
        return test_prefix, assertion

    arg = args[1]

    assgn_re = r'(int|long) '+arg+r' = (.*);'

    test_lines = test_prefix.strip().split('\n')
    print(assgn_re, test_lines[-1].strip())
    if m := re.match(assgn_re, test_lines[-1].strip()):
        assertion = 'assertEquals('+args[0]+', '+m[2]+')'
        test_prefix = '\n'.join(test_lines[:-1]) + '\n    '
        print('REPLACING')
        print(test_prefix)
        print(assertion)

    return test_prefix, assertion


def separate_assertions(tests, focal_methods):
    idxs, aligned_tests, aligned_methods, assertions = [], [], [], []
    for i, (test_method, fm) in enumerate(zip(tests, focal_methods)):
        m = assert_re.search(test_method)
        if not m:
            continue

        start = m.span()[0]
        end = m.span()[1]
        assertion = m.group(0)

        test_prefix = test_method[0:start]

        # test_prefix, assertion = consolidate_vars(test_prefix, assertion)

        clean_test = test_prefix + "\"<AssertPlaceHolder>\" ; \n }"

        idxs += [i]
        aligned_tests += [clean_test]
        aligned_methods += [fm]
        assertions += [assertion]

    return aligned_tests, aligned_methods, assertions, idxs


# def get_data(assertion_file, method_file):
def get_data(raw_tests, raw_methods, vocab, metadata):

    method_test_assert_data = []

    # assertions = open(assertion_file).read().split("\n")
    tests, methods, assertions, idxs = separate_assertions(raw_tests, raw_methods)
    # test_method_asserts = separate_assertions(tests, methods)

    
    aligned_idxs, template_matches = [], []
    for test_method, focal_method, assertion, idx in zip(tests, methods, assertions, idxs):
        # if metadata.id[idx] == 'Math81org.apache.commons.math.linear.EigenDecompositionImpl_ESTest::test39':
            # print('HERE')
            # print(metadata.id[idx])
            # print(raw_tests[idx])
            # print()
            # print(test_method)
            # print()
            # print(assertion)


        template_matches += [False]
        # if not idx == 2:

        # if idx not in [72]:
        # if idx not in [21,55,64,66,72,74,76,77]:
            # continue

        print(idx, metadata.project[idx], metadata.bug_num[idx], metadata.test_name[idx])
        print(raw_tests[idx])
        print(assertion)

        if not clean(focal_method):
            errs['unable_to_find_focal_method'] += 1
            print('MISSING FOCAL METHOD')
            continue

        start = len("org . junit . Assert . ")
        if assertion.startswith("org . junit . Assert . "):
            assertion = assertion[start:]

        # NORMALIZE ASSERTION 
        m = neg_re.match(assertion)
        if m:
            assertion = 'assertEquals('+m[1]+m[2]
        m = long_re.match(assertion)
        if m:
            assertion = 'assertEquals('+m[1]+m[2]

        m = equals_bool_re.match(assertion)
        if m:
            if m.group(1) == "true":
                assertion = "assertTrue({})".format(m.group(2))
            else:
                assertion = "assertFalse({})".format(m.group(2))

        if TEST:
            out = get_type_info_evo(assertion, focal_method, test_method, vocab=vocab)

            print('type info')
            print(out)
        else:
            out = get_type_info(assertion, test_method)

        if not out: 
            if DBG: 
                broken_tests_list =  [('parse1', test_method, focal_method, assertion)]
                for err, tm, fm, asrt in broken_tests_list:
                    print('ERROR', err)
                    print(fm)
                    print()
                    print(tm)
                    print()
                    print(asrt)
                    print('-'*50)

            continue

        _type, arg_num, total_args, matching_type_vars = out
        # print('type', out)

        if _type == float:
            continue

        
        try:
            arg_txt = assertion_to_arg(assertion, arg_num, total_args)
        except AssertionError:
            if DBG:
                broken_tests_list = [('args', test_method, focal_method, assertion)]
                for err, tm, fm, asrt in broken_tests_list:
                    print('ERROR', err)
                    print(fm)
                    print()
                    print(tm)
                    print()
                    print(asrt)
                    print('-'*50)

            continue
        
        try:
            template_asserts = gen_variants(_type, arg_txt, matching_type_vars, vocab=vocab)
        except Exception as e:
                broken_tests_list = [('gen_variants', test_method, focal_method, assertion)]
                for err, tm, fm, asrt in broken_tests_list:
                    print('ERROR', err)
                    print(fm)
                    print()
                    print(tm)
                    print()
                    print(asrt)
                    print('-'*50)
                raise e


        focal_method_clean = clean(focal_method)
        test_method_clean = clean(test_method)

        m = test_name_re.search(test_method_clean)
        test_method_name = m.group(1)

        # TEMPLATE BASED DATAGEN

        if assertion not in template_asserts:
            if m := paren_re.search(assertion):
                assertion = m.group(1) + m.group(2) + m.group(3)
            # if m := long_re.search(assertion):
                # assertion = m.group(1) + m.group(2)
            elif m := cast_re.search(assertion):
                assertion = m.group(1) + m.group(2)

        assertion_clean = clean(assertion)

        # TODO better fix
        # assertion_clean = assertion_clean.replace('assertEquals((-1)', 'assertEquals(-1')

        template_asserts = [clean(t) for t in template_asserts]

        print('CHECK')
        print(test_method_clean)
        print(template_asserts)


        template_matches[-1] = assertion_clean in template_asserts

        if assertion_clean not in template_asserts:
            # print('non template assert', assertion_clean)
            errs['non_template_assert'] += 1
            if DBG:
                broken_tests_list = [('template', test_method, focal_method, assertion_clean)]
                for err, tm, fm, asrt in broken_tests_list:
                    print('ERROR', err)
                    print(fm)
                    print()
                    print(tm)
                    print()
                    print(template_asserts)
                    print()
                    print(asrt)
                    print('-'*50)

            if TEMPLATE_CHECK:
                continue

        #template_asserts += new_asserts

        '''
        for a in bug_finding_asserts:
            if a[0] in template_asserts:
                print("POSSIBLE TO CATCH", a)
        '''

        if not focal_method_clean:
            errs['unable_to_find_focal_method'] += 1
            continue

        # pos_sample = (1, focal_method_clean, test_method_clean, assertion_clean)
        # neg_samples = []
        # for i in range(len(template_asserts)):
            # if not assertion_clean == clean(template_asserts[i]): #and\
                # #len(neg_samples) < 3: and random.random() > 0.5: # NOTE: seeded above
                # neg_samples += [(0, focal_method_clean, test_method_clean, clean(template_asserts[i]))]
                
        # assert len(neg_samples) > 0

        # method_test_assert_data += [pos_sample] + neg_samples


        samples = []
        for i in range(len(template_asserts)):
            samples += [(0, focal_method_clean, 
                             test_method_clean,
                             clean(template_asserts[i]))]

        method_test_assert_data += samples
        for i in range(len(samples)):
            # aligned_bug_meta += [tuple(bug_meta)]
            aligned_idxs += [idx]

        print([idx]*len(samples))


        #label = template_asserts.index(assertion)
        #data += [(label, assertion, template_asserts)]


    return method_test_assert_data, aligned_idxs, template_matches
    # return method_test_assert_data, aligned_bug_finding_asserts


if __name__ == "__main__":
    random.seed(0)


    TEST = True
    DATA_DIR = "../../../../atlas---deep-learning-assert-statements/Datasets/Raw_Dataset"

    vocab = np.load("vocab.npy", allow_pickle=True).item()

    K = 3 

    for k,v in vocab.items():
        vocab[k] = {k2: v2 for k2, v2 in list(reversed(sorted(v.items(), key=lambda item: item[1])))[0:K]}


    data = []
    method_test_assert_data = []
    bug_finding_asserts = []
    
    if TEST:
        assertion_file = "evo_assertions.txt"
        method_file = "evo_methods.txt"

        a, b, test_inputs, test_meta = get_data(assertion_file, method_file)
        method_test_assert_data += a
        bug_finding_asserts += b

    else:
        assertion_file = "assertLines.txt"
        method_file = "testMethods.txt"

        for split in tqdm.tqdm(["Testing", "Eval", "Training"]):
            f_assertion = os.path.join(DATA_DIR, split, assertion_file)
            f_method = os.path.join(DATA_DIR, split, method_file)
        
            #print(f_assertion)
            method_test_assert_data += get_data(f_assertion, f_method)


    method_test_assert_data = [["label","fm","test","assert"]] + method_test_assert_data
    with open("test.txt", "w") as f:
        w = csv.writer(f) 
        for d in method_test_assert_data:
            w.writerow(d)


    # test_inputs = [s for s in method_test_assert_data if s[0] == 1]
    input_data = [["focal_method","test_prefix"]] + test_inputs
    with open("assert_inputs.csv", "w") as f:
        w = csv.writer(f) 
        for d in input_data:
            w.writerow(d)

    input_data = [["project", "bug_num", 'test_name', 'exception_bug', 'assertion_bug', 'exception_lbl', 'assertion_lbl']] + test_meta
    with open("assert_meta.csv", "w") as f:
        w = csv.writer(f) 
        for d in input_data:
            w.writerow(d)


    with open("bug_finding_asserts.txt", "w") as f:
        writer = csv.writer(f)
        for row in bug_finding_asserts:
            writer.writerow(row)
        #f.write("\n".join([",".join(b) for b in bug_finding_asserts]))

    # with open("assertion_data.csv", "w") as f:
        # f.write("\n".join([",".join(map(str, d)) for d in data]))

    n_pos = sum(map(lambda r: r[0]==1, method_test_assert_data))
    n_neg = sum(map(lambda r: r[0]==0, method_test_assert_data))

    # print('processed:', len(methods))
    print('ASSERT SAMPLES:', len(method_test_assert_data), 'total,', n_pos, 'positive,', n_neg, 'negative,')
    print('TEMPLATE MATCH', len(method_test_assert_data),'collected, ERRS:', errs)

    for sample in method_test_assert_data:
        label, fm, test, assertion = sample
        if label == 1:
            for template in TEMPLATES:
                generated_templates[template] += template in assertion

    print('MATCHED TEMPLATES')
    print(generated_templates)

    # for err, tm, fm, asrt in broken_tests_list:
        # print('ERROR', err)
        # print(fm)
        # print()
        # print(tm)
        # print()
        # print(asrt)
        # print('-'*50)

    # total = len(data) + sum([v for k,v in errs.items()]
