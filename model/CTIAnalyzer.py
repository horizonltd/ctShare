import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve


class CTIAnalyser:
    """
        THIS CLASS ANALYZE CTI DATA AND SHOWCASING DEPICTION OF THE
        MODEL PERFORMANCE USING DIFFERENT MATRIX

        NOTE: THIS CLASS CAN BE INHERITED BY ANY OF THE CTI-sHARE MODELS
    """
    # read in the data
    data = pd.read_csv('datasets.csv')  # DATASET WILL BE CONFIGURED TO PULL DATA FROM A REMOTE DATASOURCE

    # preprocess the data
    data['typeofattack'] = pd.Categorical(data['typeofattack'])
    data['typeofattack'] = data['typeofattack'].cat.codes
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['description'])
    y = data['typeofattack']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

    # train the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # evaluate the model
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # predict the type of attack for new data
    new_data = ['A new type of attack is detected in Los Angeles']
    new_data_transformed = vectorizer.transform(new_data)
    prediction = lr.predict(new_data_transformed)
    print("Prediction:", prediction)

    # plot the actual type of attack and predicted type of attack
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Type of Attack')
    ax.set_ylabel('Predicted Type of Attack')
    plt.show()

    # create a label encoder object
    le = LabelEncoder()

    # encode the type of attack column
    data['typeofattack'] = le.fit_transform(data['typeofattack'])

    # print the encoded labels
    print(le.classes_)

    # Define the metrics
    precision = 0.064
    recall = 0.2
    f1_score = 0.09696969696969697

    # Create a bar chart
    labels = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1_score]
    plt.bar(labels, values)

    # Set the title and axis labels
    plt.title('Model Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')

    # Show the chart
    plt.show()

    """
        =======  STARTING =====
        THIS SECTION SERVES AS AREA WHERE MODEL CAN BE TEST EVEN WITHOUT ANY DATASET
        THIS WILL GENERATE RANDOM DATASET FOR TRAINING AND TEST PURPOSES
    """

    # Generate some random data
    X = np.random.rand(100, 1)
    y = X ** 2

    # Create a Linear Regression model
    model = LinearRegression()

    # Calculate the learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

    # Calculate the mean and standard deviation for the training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Create a learning curve chart
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()

    """
        =======  ENDING ======
        THIS SECTION SERVES AS AREA WHERE MODEL CAN BE TEST EVEN WITHOUT ANY DATASET
        THIS WILL GENERATE RANDOM DATASET FOR TRAINING AND TEST PURPOSES
    """