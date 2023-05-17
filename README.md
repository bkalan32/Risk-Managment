# Venture Funding with Deep Learning

Alphabet Soup, a venture capital firm, receives many funding applications from startups every day. The firm's business team needs help to create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

A CSV file was provided containing data on more than 34,000 organizations that have received funding from Alphabet Soup over the years. This dataset includes various information about these businesses, such as whether they ultimately became successful.

### Challenge
The challenge is to preprocess the dataset and create a binary classification model using a deep neural network that can predict whether an applicant will become a successful business based on the features in the dataset.

### Process
The process is divided into three sections:

Prepare the data for use on a neural network model.
Compile and evaluate a binary classification model using a neural network.
Optimize the neural network model.

### Data Preparation
The applicants_data.csv file is read into a Pandas DataFrame and preprocessed for use with a neural network model. The preprocessing steps include:

Dropping irrelevant columns such as "EIN" and "NAME".
Encoding categorical variables using OneHotEncoder.
Combining the original numerical data with the encoded variables.
Defining the target (y) as the "IS_SUCCESSFUL" column and the features (X) as the remaining columns.
Splitting the data into training and testing datasets.
Scaling the features data using StandardScaler.

### Model Compilation and Evaluation
A binary classification deep neural network model is created and compiled using TensorFlow's Keras. This model uses the relu activation function for the hidden layers and is fit using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric. The model is then evaluated using the test data to determine the model's loss and accuracy.

### Model Optimization
To optimize the model and improve accuracy, at least two alternative models are created. Various optimization techniques are used, such as adjusting the input data, adding more neurons or hidden layers, changing the activation functions, and altering the number of epochs in the training regimen. Each model's accuracy score is compared to determine the most effective model.

### Result
The model is saved and exported to an HDF5 file, AlphabetSoup.h5, for further use. The optimization process may not always yield a high-accuracy model; however, it provides valuable insights into tuning neural networks for specific use cases.

The goal is to strive for a predictive accuracy as close to 1 as possible. Even if this level of accuracy is not achieved, the process of creating and optimizing a deep learning model is a valuable exercise in applying machine learning techniques to real-world data.
