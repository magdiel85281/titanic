# Kaggle Titanic Challenge
I'm revisiting this challenge because my first submission was pre-my-understanding-of-GitHub. Also, I've learned a lot since that time, some of which I am practicing and demonstrating here.  


## Fetching Data
Initialize a TitanicData object and set attributes to begin working with the data. 

`t = TitanicData()` <br>
`t.fetch_data()` <br>



## Feature Engineering: First Impressions
Exploratory Data Analysis and intuition suggested some immediate opportunities for transformations in the data:

- Fill null values for Age with the average by gender and embarkation location. 
- There are finite values for Pclass, Sex, SibSp, and Parch, so convert to categorical.
- Cabin flag. My thought here is that passengers were somewhat, if not entirely, segregated. A value for Cabin may be an indicator of this.
- Many values in Cabin lead with an alphabetic character. The character may be an identifier of a Section of the ship. Once Section is broken out, add this to the partition for filling null Age with average.
- Break out LastName from Name.
- Apply MinMaxScaler (after train-test split)


The ```set``` method tranforms the data according to all but the last of the above. It also applies all transformations to the unseen test set used for submission. 

`t.set()` <br>


## Build Input for Model
Select the features to be used in a Random Forest Classifier model by first initializing a ModelInput object with the training and validation data along with the unseen test set. Then, execute the methods that set the features, split the data into the training and test sets, and apply MinMaxScaler.


`mi = ModelInput(t.data, t.test_data)` <br>
`mi.set_features()` <br>
`mi.train_test_split(test_size=0.2)` <br>
`mi.scale()` <br>

