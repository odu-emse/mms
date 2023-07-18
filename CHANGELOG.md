
<a name="v0.3.0"></a>
## [v0.3.0](https://github.com/odu-emse/mms/compare/v0.2.0...v0.3.0)

> 2023-07-18

### Bug Fixes

* figures were overlapping in exported png
* boolean cmd line arguments parsed correctly
* cmd line args do not get parsed properly
* changed type definition breaking docs generation
* addressed bugs as a result of separate testing dataset

### Code Refactoring

* logging to cmd line and log file logic
* adjusted visualizations showing and saving logic
* array of objects with ID property for JSON LP output
* worked on text output formatting
* added type safety to class members
* configured separate test dataset input to work with KNN
* switched to KNN from K-means classifier
* started reworking architecture to be supervised
* added visualization cmd line flag
* turned vectorized data into class members
* created shared class members
* worked on improving the accuracy of the model
* added more visualizations to clustering
* improved clustering accuracy
* separated data transformer from model creator method
* renamed static methods
* added output path arg for processed data save function

### Continuous Integration Changes

* fixing docs pipeline
* fixing docs pipeline
* added docs to requirements.txt
* updated transformers version
* resolving required dependency installation errors
* resolving nltk stopwords download
* fixing broken pipeline

### Features

* multiple training files can be parsed and merged
* added ability to use separate test dataset
* worked on similarity calculations between documents
* comparison of documents using cosine similarity
* created method to run clustering on whole dataset

### Maintenance

* addressed linter errors and configured workspace
* added documentation and changed viz configurations
* created debug configuration for cmd line args
* created classification log file
* cleaned up Orange board
* created vscode settings
* create base CHANGELOG file
* added stop words to list
* improved readability of cmd line prints
* removed unused code
* added more comments for docs
* added vscode settings
* generate learning path output
* upgraded sklearn package
* resumed input directory ignore
* added input directory to git tracking
* optimized GH-actions to use less compute time
* removed nltk download due to breaking deploy
* fixing docs deploy
* replaced deprecated package
* added nltk as a dep
* updated docs site
* formatting changes


<a name="v0.2.0"></a>
## v0.2.0

> 2023-03-02

