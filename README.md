# SNS_Covid
TensorFlow Keras has been used to train univariate and multivariate convolutional neural networks and long short term
 memory networks which aim to predict the number of COVID deaths reported in the UK up to 7 days in advance.
   Two data sources have been used: Our World in Data and Google's community mobility reports which are downloaded 
   and pre-processed.  The univariate convolutional neural network (which is the simplest neural network) had the best performance.
     This suggests that there is not a clear pattern in the data used which would allow the more complex models
      implemented to operate better as you would expect them to.
      
 
  How to run:
  1. Install the required packages using pip install -r requirements.txt
  2. Set up config.py as you want
  3. Run main.py
  4. From the menu download the data (or don't but expect it to be downloaded when a model is run)
  5. From the menu select a model
  6. Graphs will be plotted for the model which are shown and saved
  
  * More model structures can be added in model_structures.py
  * Country can be changed in config
  * Data columns used can be changed in config
  * Different data sources can be used by creating a loud_country function which is similar to the others