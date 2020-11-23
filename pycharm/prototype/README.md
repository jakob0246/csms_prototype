## Installation
Run project with PyCharm and Python 3.6. For that, create a new PyCharm project for the _csms_prototype_ repository. Then install the required libraries, as well, using the _requirements.txt_.

## Usage
Execute _Main.py_ if you want to run the Clustering Selection Management System (CSMS). Results will be written to a report inside the _reports_ folder. The usage of the program is also determined by the utilization of the config.txt. Here, _system_parameters_ and _system_parameter_preferences_distance_ represent the user-configuration-parameters of the system.

## Datasets
CSV & ARFF files are supported. Use the following formatting for the header in the case of CSVs per column: _"\<column-name\> \<data-type\>"_. Where the data type can be _"numerical"_ or _"categorical"_.
  
Exemplary datasets are obtained from the UCI machine learning repository.
###### Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.