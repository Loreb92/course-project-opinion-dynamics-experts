# course-project-opinion-dynamics-experts
This repository contains the scripts used for the final project of the Introduction to Computational Social Science course.

The ``config``folder contains the parameter grids used for the different experiments, the ``src`` folder contains the scripts needed to run the model, the ``notebook`` folder contains the notebook used to read the results. 

The Facebook friendship network from the The Copenhagen Networks Study can be downloaded at: https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1
Then, move it to the ``folder`` folder and unzip it

To run the experiments, run the following command on the terminal:

```
python run_experiment.py --input-params-file config/{CONFIG-FILE}.json --results-folder results/ --n-workers {N-WORKERS} --log-fold logs/ --initial-seed 1 --n-runs-per-param 100 
```
