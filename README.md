## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Steps to run code

1. Clone starter kit repo to setup environment https://github.com/udacity/CarND-Term1-Starter-Kit
2. Follow these directions: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md and create the gpu environment
3. Activate the `carnd-term1` environment

To run all the experiments you can take the following steps (Note: I would not recommend running the experiments because it is training a model for 4 epochs 430 different times with different hyperparameter combinations):  

1. Run `python fetch_data.py` 
2. Run `python run_experiments.py`
3. Wait 5 hours
4. Run `python run_top_five.py`  

The Jupyter notebook and writeup both contain explanations for the experiments being ran. Additionally the Jupyter notebook contains a model that can be trained with the set of hyperparmeters that I found had the highest accuracy rate from the experiments.


## Writeup

[Traffic Sign Classifier Writeup](writeup/writeup.md)

