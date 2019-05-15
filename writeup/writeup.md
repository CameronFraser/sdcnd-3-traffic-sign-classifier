
# Project: Build a Traffic Sign Recognition Classifier


In this project I will train a traffic sign classifier on the [German Traffic Sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) data set with an aim for model performance on test data to have an accuracy of 93% or higher. During this project I also took the time to try to develop deeper intuitions for the hyperparameters and their effect on the accuracy of the model. I came up with a list of hyperparameters that I wanted to experiment with and defined a range of values each could take. I then trained the model with each distinct set of hyperparameters for 4 epochs and recorded the results in a CSV file and found my top performing models based on their validation accuracy. The second part of this notebook and corresponding writeup will be analyzing the results from the data I collected.

---
## Load Data  

We first need to load the dataset which is done with a python script. The images are already cropped and resized so there is minimal preprocessing that needs to be done.

We are using the following libraries:
* pickle - for loading data  
* numpy - array library and linear algebra operations  
* tensorflow - machine learning framework  

After importing we load the training, validation, and test data sets which have already been split for us and assigned to their respective feature set and feature label variables

## Dataset Summary & Exploration

Exploring the data set first is important so we can start asking the following questions:  
* Is there anything wrong with the data set that needs to be fixed? Examples: classes with really high frequencies relative to other classes, missing data, corrupted data, possible adversial agents manipulating training data, etc.
* What type of data are we working with? Is it images? If so, is the object in question in full view? Do the images need to be resized? 

Below we use pandas to show a table with some stats about our data set. Although the data sets were already split up if we were working with another data set we would have similar proportions for the percentage split given the same amount of training data. A common strategy for splitting up the data set is to do an 80/20 split for train/test data and then on the train set do another 80/20 split for train/valid. This puts the train set at 64%, validation set at 16%, and test set at 20%. It is worth nothing that with very large datasets you can begin reducing the percentage given to the validation and test set. 


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Examples</th>
      <th>Percent of total data</th>
      <th>Shape</th>
      <th># Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TRAIN</th>
      <td>34799</td>
      <td>67.128996</td>
      <td>(32, 32, 3)</td>
      <td>43</td>
    </tr>
    <tr>
      <th>VALIDATION</th>
      <td>4410</td>
      <td>8.507109</td>
      <td>(32, 32, 3)</td>
      <td>43</td>
    </tr>
    <tr>
      <th>TEST</th>
      <td>12630</td>
      <td>24.363896</td>
      <td>(32, 32, 3)</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



Our goal is classify different traffic signs correctly so we should take a look at our classes and see what we are dealing with. You can see the classes with an example image from each class plotted out below. It is worth nothing that many of these signs are similar to one another. There are just a few groups of signs that are distinct geometrically. Additionally these signs have a pretty wide variance in lighting conditions which might help it generalize better.


![png](output_11_0.png)



![png](output_12_0.png)


Above is a histogram of the class distribution. There are quite a few classes with really high frequency than others, some by factors of 12. Image augmentation is a method meant to address the unbalanced frequencies.

### Pre-process the Data Set  

We will begin by pre-processing the data set by converting the image to grayscale and then using mean normalization. Additionally we will use an image augmentation process to try to even out the frequency of the class distribution. We can augment images by changing the image to make it be perceived as a new image with operations such as flipping the image, zooming in, adding salt and pepper noise, rotating, and much more. In this particular implementation I use the roll, flip, and rotate operations which should be able to increase any given class by a factor of 9 at most.


Let's look at the frequency of classes in the augmented dataset:


![png](output_18_0.png)


It looks a little bit better but my algorithm could have used some improvement. Additionally it was only at the time of writing this that I realized the image augmentation was returning floats for the numeric values which may explain why the unaugmented data set outperformed the augmented one in the experiments I ran. I did something else that could be dangerous which was accidentally turned a few classes that once had the lowest frequency to classes that now have the highest frequency due to algorithm. Without really understanding your data this could lead to the model generalizing towards the class when  maybe there was a good reason for it having a low frequency. 

### Model Architecture  

We will only train one model in this jupyter notebook, but I trained 430 models with different hyperparameter sets to gain a better intuition of how hyperparameters affected the model. The hyperparameters we're using for this model was the winner from the top 5 showdown between the highest performing models.  

### Hyperparameters

* Learning Rate - 0.001 (The size of the steps we take to reach a minimum)
* Epochs - 30 (An epoch is one full pass of the training data through the neural network)
* Batch Size - 32 (Processing a small batch size instead of the entire training set at once will let us save memory and train faster)
* Augmentation - True (Whether or not training data is augmented)
* Architecture - CONV -> MAX_POOL -> RELU -> CONV -> MAX_POOL -> RELU -> FLATTEN -> FC -> DROPOUT -> RELU -> FC  

#### Intuition For Choice in Hyperparameters

The learning rate was the first hyperparameter I experimented with and found that either 1e-3 or 1e-4 was a good option based on trying out different values. Andrej Karpathy recommends 3e-4 for Adam as a starting point, but I have not read any justification for that exact number.  

The number of epochs was also a trial and error process I did before running the experiments. 

The stride of convolution layers, max pooling, and output size of fully connected layers were just picked and not changed during this learning process. I kept those values constant along with the learning rate and epochs. (*Note: I only learned after I ran the experiments that choosing even numbers for kernel size is not recommended and I would have gotten more interesting data by testing 1, 3, 5, and 7 instead*)

| Hyperparameter                 | Range                          |
|--------------------------------|--------------------------------|
| Data Augmented?                | True or False                  |
| Batch Size                     | 32, 64, 128                    |
| Dropout Keep Prob              | 0.4, 0.5, 0.6                  |
| Layer Architecture             | 1 Conv Layer or 2 Conv Layer   |
| Filters (1st layer, 2nd layer) | (16, 32), (32, 64), (64, 128)  |
| Conv Kernel Size               | (1, 1), (2, 2), (3, 3), (4, 4) |


### Train the Model

With training we want to avoid overfitting by doing things like trainin the model for too long. Overfitting means that our network is beginning to memorize the training data which implies it is not making a good generalization about the data. Underfitting is just the opposite, but it seems that overfitting is an easier trap to fall into. For the optimizer we are using the Adam (adaptive moment estimation) optimizer whch is similar to stochastic gradient descent (SGD). One of the differences is that in SGD the optimizer maintains a single learning rate for all weights (the learning rate is a function of the momentum), whereas Adam will maintain a unique variable learning rate for each weight.


From the visualization below we can see that after about the 10th epoch the model's accuracy began fluctuating around a similar range of accuracy.


![png](output_25_0.png)


### Test the Model

Finally we can test our model against new training data and see how well it does.


    Test Accuracy = 0.964


Our accuracy on the test set is 96.4% which is a little bit lower than what we were getting on the validation set, but it is still pretty good. I did not go back and try to tune results from here as I felt that would sacrifice the integrity of the result and possibly appeal to test data.

---

## Test a Model on New Images

While we already tested the model on unseen data from the provided dataset, it would be interesting to see how well it perform on German traffic signs found outside of the dataset.

### Load and Output the Images


![png](output_31_0.png)


### Predict the Sign Type for Each Image

We need to remember to pre process these new images since that is what the model is used to. We use the argmax function to get the index of the max value which and map it to the values in the signnames.csv file. After that all there is left to do is check the accuracy of the results.


![png](output_33_0.png)


### Analyze Performance

The classifier classified all 10 of the traffic signs correctly.

---

## Analysis of Experiment Data

Below is the results of the experiments along with some analysis of the data collected.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>architecture</th>
      <th>batch_size</th>
      <th>conv_filters</th>
      <th>conv_ksize</th>
      <th>data_augmented</th>
      <th>dropout_keep_probability</th>
      <th>elapsed_time_to_train</th>
      <th>validation_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>64.0</td>
      <td>[64, 128]</td>
      <td>[4, 4]</td>
      <td>False</td>
      <td>0.5</td>
      <td>31.183523</td>
      <td>[0.71927437593066501, 0.85442176870748299, 0.9...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>128.0</td>
      <td>[64, 128]</td>
      <td>[2, 2]</td>
      <td>True</td>
      <td>0.4</td>
      <td>42.758209</td>
      <td>[0.60748299341353162, 0.74580498849994203, 0.8...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>128.0</td>
      <td>[32, 64]</td>
      <td>[2, 2]</td>
      <td>False</td>
      <td>0.5</td>
      <td>8.770622</td>
      <td>[0.73537414974095872, 0.82448979570211467, 0.8...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>128.0</td>
      <td>[16, 32]</td>
      <td>[3, 3]</td>
      <td>False</td>
      <td>0.6</td>
      <td>6.978182</td>
      <td>[0.76553288014297316, 0.85351473849917214, 0.8...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>64.0</td>
      <td>[16, 32]</td>
      <td>[2, 2]</td>
      <td>True</td>
      <td>0.5</td>
      <td>22.562105</td>
      <td>[0.72426303865687913, 0.78321995508103148, 0.8...</td>
    </tr>
  </tbody>
</table>
</div>



We can see in the data above that the changing hyperparameters during the experiment were documented along with the time to train and the validation accuracies from every epoch. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>architecture</th>
      <th>batch_size</th>
      <th>conv_filters</th>
      <th>conv_ksize</th>
      <th>data_augmented</th>
      <th>dropout_keep_probability</th>
      <th>elapsed_time_to_train</th>
      <th>validation_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>64.0</td>
      <td>[64, 128]</td>
      <td>[4, 4]</td>
      <td>False</td>
      <td>0.5</td>
      <td>31.183523</td>
      <td>0.872744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>128.0</td>
      <td>[64, 128]</td>
      <td>[2, 2]</td>
      <td>True</td>
      <td>0.4</td>
      <td>42.758209</td>
      <td>0.771383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>128.0</td>
      <td>[32, 64]</td>
      <td>[2, 2]</td>
      <td>False</td>
      <td>0.5</td>
      <td>8.770622</td>
      <td>0.842721</td>
    </tr>
    <tr>
      <th>3</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>128.0</td>
      <td>[16, 32]</td>
      <td>[3, 3]</td>
      <td>False</td>
      <td>0.6</td>
      <td>6.978182</td>
      <td>0.856871</td>
    </tr>
    <tr>
      <th>4</th>
      <td>conv --&gt; relu --&gt; max_pool --&gt; flatten --&gt; fc ...</td>
      <td>64.0</td>
      <td>[16, 32]</td>
      <td>[2, 2]</td>
      <td>True</td>
      <td>0.5</td>
      <td>22.562105</td>
      <td>0.795737</td>
    </tr>
  </tbody>
</table>
</div>



Converting the validation accuracy field to an average of its values makes it easier to sort the content. This is probably not a great idea in practice, but I ran out of time and didn't research different ways of measuring model performance other than accuracy.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>architecture</th>
      <th>batch_size</th>
      <th>conv_filters</th>
      <th>conv_ksize</th>
      <th>data_augmented</th>
      <th>dropout_keep_probability</th>
      <th>elapsed_time_to_train</th>
      <th>validation_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>333</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>32.0</td>
      <td>[64, 128]</td>
      <td>[3, 3]</td>
      <td>False</td>
      <td>0.6</td>
      <td>106.978316</td>
      <td>0.940907</td>
    </tr>
    <tr>
      <th>381</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>32.0</td>
      <td>[32, 64]</td>
      <td>[3, 3]</td>
      <td>False</td>
      <td>0.5</td>
      <td>425.330492</td>
      <td>0.940000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>32.0</td>
      <td>[16, 32]</td>
      <td>[4, 4]</td>
      <td>False</td>
      <td>0.5</td>
      <td>30.320489</td>
      <td>0.935465</td>
    </tr>
    <tr>
      <th>418</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>32.0</td>
      <td>[32, 64]</td>
      <td>[4, 4]</td>
      <td>False</td>
      <td>0.5</td>
      <td>28.004069</td>
      <td>0.935193</td>
    </tr>
    <tr>
      <th>397</th>
      <td>conv --&gt; max_pool --&gt; relu --&gt; conv --&gt; max_po...</td>
      <td>32.0</td>
      <td>[32, 64]</td>
      <td>[3, 3]</td>
      <td>False</td>
      <td>0.4</td>
      <td>714.426502</td>
      <td>0.934921</td>
    </tr>
  </tbody>
</table>
</div>



A few things should stand out right away. The first is that the 2 layer convolutional net dominated the top 5 architecture parameter. Second, a batch size of 32 was also consistent with models with high validation accuracy. Lastly, the data augmented parameter is false on all of them which brings me back to my earlier comments about being uncertain about my augmentation strategy. I believe these results prove that my choice was not a good one.

------

## Conclusion

The experiments I ran have started to give me a decent foundation for my intuition in solving learning problems. The first thing I would have done differently is spent more time with the image augmentation and construct a better algorithm for it or rely on existing software to do it for me such as Keras ImageDataGenerator class. The second improvement I could make is to use a non brute force solution for hyperparameter searching. There are many different approaches such as random search and/or Bayesian optimization. The third improvement would be to use Keras and tensorflow with eager execution. Using tensorflow directly was unnecessary for this project and I ended up just building a Keras-like api over tensorflow. Lastly, I would focus on experimenting more with the layer parameters that I left constant and the layer architectures.

