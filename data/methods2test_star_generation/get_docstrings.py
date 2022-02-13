import os
import re
import csv
import numpy as np
from tree_hugger.core import JavaParser

repo_re = re.compile("methods2test-projects\/([^\/]*)")
proj_name_re = re.compile("https:\/\/github\.com\/\S*\/(\S*)")
repo_to_classes = np.load("repo_to_classes.npy", allow_pickle=True).item()


def get_classes(proj_name):
    count = 0
    values = None
    for k, v in repo_to_classes.items():
        k_proj_name = proj_name_re.match(k).group(1)
        if k_proj_name == proj_name:
            values = v
            count += 1
    try:
        assert count == 1 and values
    except Exception as e:
        return None

    return values

#if __name__ == "__main___":

with open("docstring_data.txt","w") as f:
    w = csv.writer(f)
    w.writerow(["fc","fm","docstring"])
    '''
    for point in data:
        w.writerow(point)
    '''

data = []
done = False
jp = JavaParser("/tmp/tree-sitter-repos/my-languages.so")
for root, dirs, files in os.walk("methods2test-projects/"):
    if done: break
    for f in files:
        if done: break
        if not f.endswith(".java"): continue

        full_fname = os.path.join(root, f)
        repo_name = repo_re.match(full_fname).group(1)
        classes = get_classes(repo_name)
        class_name = f[0:f.find(".java")]

        if not classes or not class_name in classes: continue
        jp.parse_file(full_fname)
        docstrings = jp.get_all_method_documentations()
        methods = jp.get_all_method_bodies()
        for k, v in docstrings.items():
            data += [(class_name, k, v)]
            #print(len(data))

            #if len(data) > 100:
            #    done = True

            if len(data) % 10000 == 0:
                print("writing data from", len(data)-10000, "to", len(data))
                with open("docstring_data.txt","a") as f:
                    w = csv.writer(f)
                    #w.writerow(["fc","fm","docstring"])
                    for point in data[len(data)-10000:]:
                        w.writerow(point)

        '''
        if docstrings:
            print(jp.get_all_method_bodies())
            print(docstrings.keys())
            break
        '''

with open("docstring_data.txt","a") as f:
    w = csv.writer(f)
    idx = len(data) % 10000
    print("writing data from", len(data)-idx, "to", len(data))
    for point in data[len(data)-idx:]:
        w.writerow(point)
