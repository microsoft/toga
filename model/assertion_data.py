import os, sys, re, tqdm, csv, random
import javalang
import numpy as np
import pandas as pd
from collections import defaultdict

whitespace_re = re.compile(r'\s+')

long_re = re.compile(r"assertEquals\(([-]?[0-9]+)L(,.*)")
neg_re = re.compile(r"assertEquals\(\((-\d+)\)(.*)")
cast_re = re.compile("(assertEquals\()\(\w+\)(\S*, \w+\))")
paren_re = re.compile("(assertEquals\()\((\S+)\)(, \w+\))")
equals_bool_re = re.compile(r"assertEquals\((false|true), (.*)\)")
test_name_re = re.compile("public void (test[0-9]*)\(\)")

var_re = re.compile(r"^\w+$")
method_call_re = re.compile(r"^\w+\.\w+\(\)$")

assert_re = re.compile("assert\w*\(.*\)")

arg_re_generic = re.compile("assert\w*\s*\((.*)\)")

vocab = None

ASSERT_TYPES = ("assertTrue", "assertFalse", "assertEquals", "assertNotNull", "assertNull")

errs = defaultdict(int)


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


def pretty_type(tree, full_type=""):
    if isinstance(tree, javalang.tree.BasicType):
        return tree.name

    if tree.sub_type:
        return tree.name + "." + pretty_type(tree.sub_type, full_type)
    return tree.name


def get_type_info_evo(assertion, focal_method, test_method, vocab=vocab):
    
    try:
        tokens = javalang.tokenizer.tokenize(focal_method)
        parser = javalang.parser.Parser(tokens)
        focal_method_node = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration,\
            javalang.tokenizer.LexerError):
        errs["cant-parse"] += 1
        return None

    start = test_method.find("\"<AssertPlaceHolder>\" ;") 
    clean_test = test_method[0:start] + " }"
    
    tokens = javalang.tokenizer.tokenize(clean_test)
    parser = javalang.parser.Parser(tokens)
    try:
        test_method_node = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration,\
            javalang.parser.LexerError):
        errs["cant-parse4"] += 1
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
        errs["no-return"] += 1
        return None

    if return_type == "boolean":
        _type = bool
    elif return_type == "String":
        _type = str
    elif return_type == "double" or return_type == "float":
        _type = float
    elif return_type == "int":
        _type = int
    else:
        _type = return_type

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
            same_type_vars += [var]

    #PARSE ASSERTION
    try:
        tokens = javalang.tokenizer.tokenize(assertion)
        parser = javalang.parser.Parser(tokens)
        assertion_obj = parser.parse_primary()
    except javalang.parser.JavaSyntaxError:
        errs["cant-parse5"] += 1
        return None
    
    arg_ind = 0 if len(assertion_obj.arguments) == 1 else 1
    return _type, arg_ind, len(assertion_obj.arguments), same_type_vars
    
def assertion_to_arg(assertion, arg_num, total_args):
    m = arg_re_generic.search(assertion)

    g = m.group(1)
    args = g.split(",")
    try:
        assert len(args) == total_args and len(args) > arg_num 
    except AssertionError as e:
        if total_args == 1:
            return g
        else:
            errs["unable_to_extract_assert_args"] += 1
            raise e

    return args[arg_num]

def gen_variants(_type, arg, matching_type_vars, vocab=vocab):
    out = []
    values = matching_type_vars 
    arg = arg.strip()
    if _type in vocab: 
        top_values = vocab[_type]
        values = top_values + values

    for var in values:
        if var == arg:
            continue
        out += ["assertEquals({}, {})".format(var, arg)]

    if _type == bool:
        out +=  ["assertTrue({})".format(arg), "assertFalse({})".format(arg)]
    elif not _type == int and not _type == float:
        out += ["assertNotNull({})".format(arg)]
    
    return list(set(out))


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

        clean_test = test_prefix + "\"<AssertPlaceHolder>\" ; \n }"

        idxs += [i]
        aligned_tests += [clean_test]
        aligned_methods += [fm]
        assertions += [assertion]

    return aligned_tests, aligned_methods, assertions, idxs


def get_model_inputs(raw_tests, raw_methods, vocab):

    method_test_assert_data = []

    tests, methods, assertions, idxs = separate_assertions(raw_tests, raw_methods)
    
    aligned_idxs, template_matches = [], []
    for test_method, focal_method, assertion, idx in zip(tests, methods, assertions, idxs):

        template_matches += [False]

        if not clean(focal_method):
            errs['unable_to_find_focal_method'] += 1
            continue

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

        out = get_type_info_evo(assertion, focal_method, test_method, vocab=vocab)

        if not out: 
            continue

        _type, arg_num, total_args, matching_type_vars = out

        if _type == float:
            continue
        
        try:
            arg_txt = assertion_to_arg(assertion, arg_num, total_args)
        except AssertionError:
            continue
        
        try:
            template_asserts = gen_variants(_type, arg_txt, matching_type_vars, vocab=vocab)
        except Exception as e:
            raise e


        focal_method_clean = clean(focal_method)
        test_method_clean = clean(test_method)

        m = test_name_re.search(test_method_clean)
        test_method_name = m.group(1)

        if assertion not in template_asserts:
            if m := paren_re.search(assertion):
                assertion = m.group(1) + m.group(2) + m.group(3)
            elif m := cast_re.search(assertion):
                assertion = m.group(1) + m.group(2)

        assertion_clean = clean(assertion)

        template_asserts = [clean(t) for t in template_asserts]

        template_matches[-1] = assertion_clean in template_asserts

        if assertion_clean not in template_asserts:
            errs['non_template_assert'] += 1
            
        if not focal_method_clean:
            errs['unable_to_find_focal_method'] += 1
            continue

        samples = []
        for i in range(len(template_asserts)):
            samples += [(0, focal_method_clean, 
                             test_method_clean,
                             clean(template_asserts[i]))]

        method_test_assert_data += samples
        for i in range(len(samples)):
            aligned_idxs += [idx]

    return method_test_assert_data, aligned_idxs
