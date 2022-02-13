import csv, re, tqdm, sys, os
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

fm_re = re.compile("(public|private|static)? ?[^(]* (\S*)\(")

print('loading doc data...')
# doc_data = [] #(class, method name, docstring)
doc_db = defaultdict(list)
with open("docstring_data.txt") as f:
    r = csv.reader(f)
    for fc, fm_name, docstring in r:
        doc_db[fm_name] += [ (fc, docstring) ]
        # doc_data += [row]

# for fc, fm_name, docstring in doc_data:


for split in ["train", "test", "eval"]:
    print('matching', split+'.csv')

    train_data = [] #(label, test, fm)
    with open(split+".csv") as f:
        r = csv.reader(f)
        for row in r:
            train_data += [row]

    
    out_data = []
    count = 0
    for label, test, fm in tqdm.tqdm(train_data[1:]):
        my_docstring = ""

        m = fm_re.search(fm)
        if not m:
            continue

        parsed_name = m.group(2)

        for fc, doc_sample in doc_db[parsed_name]:
            if fc in test or fc in fm:
                my_docstring = doc_sample
                count += 1
                break
        
        # if len(out_data) % 1000 == 0:
            # print("got", count, "docstrings out of", len(out_data), "samples...")
        out_data.append([label, test, fm, my_docstring])

    print("got", count, "docstrings out of", len(out_data), "samples")
    outdir = 'methods2test_star'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(outdir+'/'+split+".csv", "w") as f:
        w = csv.writer(f)
        w.writerow("label,test,fm,docstring".split(','))
        for sample in out_data:
            w.writerow(sample)
