# ToGA Artifact

This repository contains the replication artifact for TOGA: A Neural Method for Test Oracle Generation to appear in ICSE 2022.

Testing is widely recognized as an important stage of the softwaredevelopment lifecycle. Effective software testing can provide benefits such as documentation, bug finding, and preventing regressions. In particular, unit tests document a unit’s intended functionality. A test oracle, typically expressed as an condition, documents the intended behavior of the unit under a given test prefix. Synthesizing a functional test oracle is a challenging problem, as it has to capture the intended functionality and not the implemented functionality. In our paper, we propose TOGA (Test Oracle GenerAtion), a unified transformer-based neural approach to infer both exceptional and assertion test oracles based on the context of the focal method.

Our artifact reproduces the results for all RQs in the paper's evaluation.  The artifact includes source code and download links for datasets and models produced in the paper, fulfilling the requirements for reproduced, resuable, and available badges. We assume basic unix familiarity and ability to run python. Our artifact is optionally given as a docker image for linux.

## Docker Setup
For an easy setup, we recommend our docker container that includes all data, pretrained models, and source. Otherwise, follow the setup instructions in the next section.

First, pull the docker image:    
`docker pull edinella/toga-artifact`      

Connect to it: 
`docker run -i -t edinella/toga-artifact`

Then, setup some environment variables:    
`export PATH=$PATH:/home/defects4j/framework/bin`    
`export ATLAS_PATH=/home/icse2022_artifact/data/atlas---deep-learning-assert-statements/`

## Setup

Requirements: `python3.9`, `git lfs`

First, clone this repo and install the dependencies:    
```
cd toga/    
pip install -r requirements.txt
git lfs pull
git clone https://gitlab.com/cawatson/atlas---deep-learning-assert-statements.git
export ATLAS_PATH=<path_to_atlas...>/atlas---deep-learning-assert-statements/  
```


## Models    
If you're using our docker image, the pretrained models are in:
```
icse2022_artifact/model/assertions/pretrained/    
icse2022_artifact/model/exceptions/pretrained/    
```

Otherwise, to install our exception and assertion pretrained models, download from: https://drive.google.com/drive/folders/1dZDxu92rZzB_LEwnAkkiy3DblxMJ6nUT?usp=sharing
Put them in `model/exceptions/pretrained/pytorch_model.bin` and `model/assertions/pretrained/pytorch_model.bin` respectively.

To train your own model, run 
```
cd model/exceptions/
bash run_train.sh

cd model/assertions/
bash run_train.sh
```

### Defects4j setup (optional):
If you want to build and execute defects4j tests, [defects4j](https://github.com/rjust/defects4j) must be installed.

Requirements:
```
sudo apt install libdbi-perl
sudo apt install openjdk-8-jdk
sudo apt install libdbd-csv-perl
```


## Datasets - Preprocessed
Our approach is trained and evaluated on three datasets. The first, **Atlas\*** is an adaption of the [Atlas](https://gitlab.com/cawatson/atlas---deep-learning-assert-statements/) dataset. The preprocessed datasets **Atlas\*** and **Methods2Test\*** are included as `data/<DATASET>_star.tar.gz` files and be accessed by:
```
cd data
tar xzf atlas_star.tar.gz
tar xzf methods2test_star.tar.gz
```

The third dataset is our test dataset generated from Evosuite tests. This is located in `data/evosuite_tests.tar.gz`. Since the dataset size is very large (>400k tests), we provide two smaller sample datasets that can be used to reproduce the bug counts result and false positive rate result respectively from Table 3 for `Our Approach` in `data/evosuite_reaching_tests.tar.gz` and `data/evosuite_5project_tests.tar.gz`. `reaching_tests` contains bug-reaching tests only, while `5project_tests` contains the tests generated for the same 5 defects4j projects used in [1]'s evaluation.

To access the evosuite test datasets run:
```
cd data
tar xzf evosuite_reaching_tests.tar.gz
tar xzf evosuite_5project_tests.tar.gz
tar xzf evosuite_tests.tar.gz
```

## OPTIONAL Datasets - Generate from base datasets

```
cd data
python atlas_star_datagen.py
```

The resulting dataset will be stored in ``data/atlas_star``

Methods2Test\* Generation:
To generate **Methods2Test\***, first download the methods2test dataset).

```
git clone git@github.com:microsoft/methods2test.git
cd methods2test
export METHODS2TEST_PATH=`pwd`

cd dataset
for f in *.bz2; do tar xf $f; done
cd ../corpus/raw/fm
for f in *.bz2; do tar xf $f; done
```

Navigate to the methods2test_star_gen directory and execute scripts in order. Note that these scripts can take several hours to run because they must download all projects from the methods2test dataset and parse them for docstrings, so we recommend using the provided preprocessed dataset.

```
cd data/methods2test_star_generation
python gen_exception_dataset.py
bash download_projects.sh
python get_docstrings.py
python match_docstrings.py
```

## Evaluation
In our paper, we evaluate three research questions (RQs). 

The following commands assume you are in the root of this directory.


1. **RQ1: Is our grammar representative of most developer-written assertions?**
   To evaluate this research question:    
   ``cd eval/rq1 && python rq1.py``

2. **RQ2: Can we infer assertions and exceptional behavior with high accuracy?**



**Exception Inference:**

To reproduce the exception results shown in table 1 for `TOGA Model`, run:
```
cd eval/rq2/exception_inference
bash rq2.sh
```

This script uses the pretrained exception model to predict whether a test is expected to trigger an exception or not, evaluated for accuracy and f1 score on the `methods2test_star` dataset. Note that the model used in the artifact has been retrained, so the results are slightly different from the submission (accuracy=85\% instead of 86\%, f1 score is 0.40 instead of 0.39). 

To reproduce the weighted coin experiment in table 1, run:
```
cd eval/rq2/exception_inference
python coin.py
```

**Assertion Inference:**
To reproduce the exception results shown in table 2 for `TOGA Model`, run:
```
cd eval/rq2/assertion_inference
bash rq2.sh
```

This script uses the pretrained assertion model to predict an assertion given a test prefix and method under test's signature. we evaluate for accuracy and f1 score on the `atlas_star` dataset. 

3. **RQ3: Can we catch bugs with low false alarms?**

To reproduce the results shown in Table 3 for `Our Approach`, run `toga.py` on either the bug-reaching inputs (for bug results) or 5 project sample (for false positive rate). By default, the `toga` tool will use the test metadata labels to evaluate oracles predicted by the models and print results. The tool also generates a `predicted_oracles.csv` file that can be used to generate executable test suites.

Note that we have improved our implementation since the submission and now find 4 additional bugs (58 total instead of 54) with a lower FP rate (22\% instead of 25\%).

To faciliate faster evaluation, the `toga.py` will automatically check its predicted oracles against labels included in its metadata input. This can save time since generating and executing test suites is potentially very time consuming. Note that `toga` will overestimate the `FP rate` when checking against labels, so the False Positive rate on generated tests will be lower.

To validate the results, use the `eval/rq3/rq3.sh` to generate and run test suites from the toga generated oracles.



To reproduce table 3 bug result by running only bug-reaching tests (this will not reproduce the FP rate, which requires running on all of the tests):
```
python toga.py data/evosuite_reaching_tests/inputs.csv data/evosuite_reaching_tests/meta.csv
```

To reproduce table 3 false positive rate result on a 5 project sample (2+ hour runtime):
```
python toga.py data/evosuite_5project_tests/inputs.csv data/evosuite_5project_tests/meta.csv
```


Both bug and FP rate results on the entire dataset (potentially 12+ hour runtime):
```
python toga.py data/evosuite_5project_tests/inputs.csv data/evosuite_5project_tests/meta.csv
```




## References

1. Tufano, Michele, et al. "Unit Test Case Generation with Transformers." arXiv preprint arXiv:2009.05617 (2020). 

