# Classification Models for Human Resources Analytics
This project contains a script responsible for implementing SVM, Neural Networks and Naïve Bayes with three different datasets: the original one, the one which contains only features selected by RELIEF algorithm and the one that has PCA components. For each run, the script prints average error, recall and precision for comparison. 
P.S.:The original dataset was downloaded from kaggle at https://www.kaggle.com/ludobenistant/hr-analytics/downloads/human-resources-analytics.zip. 

#### How to run

Clone the repository:

`git@github.com:tuliorc/hr-analysis.git`


Go into your new local repository:

`cd hr-analysis`


Make sure you have R installed in your computer:

`R --version`


In case you don't, install it:

`sudo apt-get update`
`sudo apt-get install r-base`


For other Operational Systems, follow the instructions found in the links below:

- https://cran.r-project.org/bin/macosx/
- https://cran.r-project.org/bin/windows/base/

Then, execute the main file script.R, either by running it in RStudio or terminal. 


#### Tips
* Try to run, at first, the first block of code independently. You may need to run install.packages() specifying the packages that you still don't have in your computer. Both RStudio and Terminal are going to notify you about what is missing.
* Then, run the second and third blocks of code. You may again be asked to install some missing packages.
* After that, you can run the fourth to sixth blocks of code. These will only create the model functions and no trouble should be caused.
* Now, you're able to run the seventh (and last) block of code. Here you can make some experimentations! Feel free to change and adapt parameters to what you are curious about.
By chunking the script as above, it will be easier to follow and understand what's going on!
