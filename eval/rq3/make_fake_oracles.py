from glob import glob
import csv
import json
import random

def get_pairs(fname):
    pairs = []
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            pairs.append(row)
            
    return pairs

def get_random_assertion():
    return "assertTrue(true)"

def write_oracle(oracle, oracle_num):
    out_file = f"oracles/{oracle_num}.json"
    print("writing to", out_file)
    with open(out_file, "w") as f:
        json.dump(oracle, f)

def get_random_oracle(fm):
    exception = random.randint(0,1)
    if exception:
        assertion = None
    else:
        assertion = get_random_assertion()

    oracle = {"exception": exception, "assertion": assertion, "fm": fm}
    return oracle

oracle_num = 0
for data_pair_file in glob("../data/*fm_pairs.csv"):
    pairs = get_pairs(data_pair_file)
    for evotest, fm in pairs:
        rand_oracle = get_random_oracle(fm)

        write_oracle(rand_oracle, oracle_num)
        oracle_num += 1
