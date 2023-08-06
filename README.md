# Overview of Project

This is an assignment for Week 7 in the Northwestern Masters in Data Science [MSDS-431 Data Engineering with Go](https://msdsgo.netlify.app/data-engineering-with-go/) course.

The purpose of this work is to figure out whether there is a suitable implementation of an Isolation Forest algorithm in Go, to the point where it would produce relatively consistent results compared to similar algorithm implementations in Python and R.  In effect, this is kind of like a real-life stress test of being a data scientist with a job to do where your manager tells you the company is switching to Go and if you have a problem with it, holler.

### Key findings

The tl;dr is that there are multiple implementations of the isolation forest algorithm in Go that produce results similar to other - more popular - implementations.  I tested the consistency of these algorithms to identify outliers in the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database of handwritten digits against R's isotree and solitude implementations and Python's sklearn.ensemble IsolationForest method.  All work was performed on the 60k-item training dataset, and evaluation takes the form of correlations in outlier score.

Correlation results:

| Correl         | Python SKLearn | R Isotree | R Solitude | Go IForest | Go RForest |
| -------------- | -------------- | --------- | ---------- | ---------- | ---------- |
| Python SKLearn | 1              |           |            |            |            |
| R Isotree      | 0.72           | 1         |            |            |            |
| R Solitude     | 0.58           | 0.74      | 1          |            |            |
| **Go IForest**     | \-0.94         | \-0.67    | \-0.52     | 1          |            |
| **Go RForest**     | \-0.94         | \-0.76    | \-0.64     | 0.91       | 1          |

Note: Go IForest and Go RForest are different packages, enumerated below.

### Specific Answers to Specific Questions

#### What led to the selection of one package over others?
See below for additional detail.  Two different packages were tested: 
1. [e-XpertSolutions/go-iforest](https://github.com/e-XpertSolutions/go-iforest), which is the "Go IForest" above
2. [malaschitz/randomForest](https://github.com/malaschitz/randomForest), which is the "Go RForest" above.

An initial list of packages which purported to implement isolation forests was created, and then each package was reviewed for popularity, recency of activity, and apparent ease of use via documentation.

The Go IForest package was my clear favorite since the package was focused on isolation forests and documentation on use was transparent and apparently easy to follow.  I was unable to crown this the victor without researching alternatives, however, because something in my initial implementation was generating results materially inconsistent with Python and R.  These issues were eventually resolved and the correlation scores above identify the algorithm is working.  However, a bug that magically "fixes itself" isn't really fixed, so until I have more confidence in the algorithm I'd hesitate to only use this package.

The Go RForest package was my backup.  Most of the code that had been written to use the Go IForest package was reusable to run this incarnation.  I wasn't thrilled with this package because its focus was on random forests and then the isolation forest implementation appeared to be an afterthought.  Further, there were two approaches for generating isolation forests and the package author stated they would produce different results.  Note that this package's implementation of a direct isolation forest doesn't return an anomaly score, but returns two values instead: one representing the number of trees where the data point appears and one representing the cumulative depth of the point in all the trees.  I had to first calculate each item's ratio (depth / # trees) and then normalize these ratios to produce a 0-1 anomaly score.  I'm certain I'm missing some logic in this application, and the incompleteness of this package is a ding against its long-term reliability.

See below for other alternatives considered.

#### How easy was it to clone the package repository, implement the code, and set up the tests?

Overall, relatively difficult. There is an immaturity to most of the golang packages, so many of the points of friction that have been worn down through millions of touches in Python or R still exist in go.  For example, multiple methods needed to be written to read in the MNIST database, validate the read-in happened correctly, and then pass these results to multiple modeling methodologies.  Straightforward, but not easy.

If this were being implemented in a live production environment I think there's a lot of work that can be done to better define structures and types that would simplify the code.  Unfortunately, this assignment was done on an aggressively abbreviated timeframe so it's the literal definition of "if I had more time I would have written a shorter note [program]."

Fortunately, we were provided the R and Python results, as well as extensive starter code to accelerate the analysis of our work.  Once the scores were produced, comparing to other sources was streamlined.

#### Were the tests successful?  To what extent were Go results similar to results from R and/or Python?

Eventually, yes.  As noted above, the correlations ranged from 0.5 - 0.95 (on an absolute basis).  More research is necessary to really identify why there were some instances with surprisingly low performance, but it's a very promising start.

#### How difficult will it be for the firm to implement Go-based isolation forests as part of its data processing pipeline?

It will be straightforward.  I am concerned, though, at the lack of dynamism around hyperparameter tuning compared to alternative systems.  The solutions I worked with provide an out-of-the-box set of results that look intuitive *prima facie*, but I'd want to ensure I have the appropriate level of control if I were to turn over the keys to my business kingdom to this algorithm.

I was also surprised at how slow the code was.  I benefitted from not having to run the Python or R code to produce the scores myself, but it was not a fast process to run these two go approaches.  I made some half-hearted attempts to speed it up using concurrency but I worried about maintaining appropriate order in my data structure for downstream comparison to other systems' outputs.


# Program Structure and Use

The resultant exe from this work produces a CSV file with anomaly scores for each of the 60,000 images in the MNIST training database.  File paths within the program are hard-coded so please don't change the file structure.

In any case, run:

```bash
TKTKTK
```

the R file results/analyzeResults.R was largely provided by our professor as an example of how to quickly collect and compare different scores.  This code was enriched based on what he shared to also include multiple Go instances and to produce png versions of graphs for inclusion in this document.

As always, this program was developed on a Mac although both Mac and Windows executables are provided.  This is because the Canvas website which manages assignments will only accept a .exe and won't accept a Mac executable.  The Mac executable has been tested and works; the Windows executable has not been tested.

# Full Set of Packages Researched

These were all of the packages that we, as a class, took a look at to see if they would work for isolation forest applications:

* [e-XpertSolutions/go-iforest](https://github.com/e-XpertSolutions/go-iforest).  This is my preferred: it was last active in November 2022 and has good documentation for how to use it.  Also, we don't know yet how the python code looks but this has built-in functionality to run this on the training set and apply the model to the test set
* [bn4t/go_iforest](https://github.com/bn4t/go-iforest).  Looks like a valid option.  Output is the anomaly score and that's it.  Last updated in Jan 2021
* [malaschitz/randomForest](https://github.com/malaschitz/randomForest).  Main focus of the package is random forests while isolation forests appears to be a specific (afterthought) application once all of the infrastructure was built.  Package was last updated in July 2022, but documentation is poor: two options are provided for how to generate the isolation forest but as the documentation states, the results are different although why they'd be different isn't made clear.
* [golearn](https://github.com/sjwhitworth/golearn).  This package gets a lot of support because it has a very large breadth of topics being covered.  I never really got into the details here but some classmates said it was difficult to get the data structured correctly for ingestion.

### Additional Hat Tips

* Hat tip to [petar/GoMNIST](https://github.com/petar/GoMNIST) for writing a set of functions to make reading the MNIST data a little less painful.
* Hat tip to [Prof Miller](https://github.com/ThomasWMiller/jump-start-mnist-iforest/tree/main) for providing a bunch of work we could use as a starting point.

# Distribution and Correlation Graphs

### Distribution Graphs



# FYI - assignment details motivating this work

### Management Problem

Managers of a technology startup are keen on limiting the number of computer languages supported by the company. They would like software engineers and data scientists to work together using the same language for backend research and product development. In particular, they want to see employees using Go as their primary programming language. 

Critical to the work of data science is preparing data for analysis and modeling. It is not uncommon for data scientists to spend 70 to 80 percent of their time in data cleaning, with outlier detection and data deduplication comprising essential processes (Ilyas and Chu 2019). Considering outlier/anomaly detection in particular, managers of the startup wonder if a Go package for isolation forests could replace R and Python routines. 

Isolation forests have proven to be effective in detecting outliers in multivariate data analysis (Emmott et al. 2016). Extensive research has been conducted comparing isolation forests with other methods of unsupervised outlier/anomaly detection (Koren, Koren, and Peretz 2023).

Recent searches of go.dev Links to an external site. and Awesome Go Links to an external site. show a number of open-source Go packages for isolation forests and random forests in general. Perhaps one of these packages could serve the firm's needs for outlier/anomaly detection in the data science pipeline.

### Assignment Requirements 

Take on the role of a company data scientist looking for a Go package for isolation forests. Begin by reviewing the literature on isolation forests, as described under Outlier Detection: Isolation Forests. Then complete the following assignment tasks:

* Review open-source package options for isolation forests from the Go community, as identified from searches of go.dev Links to an external site. and Awesome Go Links to an external site. . Select one of the packages for testing. (It is a good idea to choose a package that has numerous contributors and users, as well as recent activity, such as postings in 2023. Finding an isolation forests or random forests package that satisfies these criterial may be difficult, however.  So, we may need to settle for a package that has adequate documentation and well-worked code examples.)
* Under Programs, Data, and Documentation below, we will provide a link to a GitHub repository with jump-start programs for gathering the MNIST data and for running isolation forest programs in R and Python. Try running the R and/or Python programs to see what to expect from a Go isolation forest program for MNIST outlier/anomaly detection.
* Working with the selected Go package for isolation forests, run tests against the MNIST dataset. Compare the Go results with R and/or Python results.
* In the README.md file, summarize the work on Go package selection, implementation, and testing. What led to the selection of one package over others? How easy was it to clone the package repository, implement the code, and set up the tests?  Were the tests successful?  To what extent were Go results similar to results from R and/or Python? How difficult will it be for the firm to implement Go-based isolation forests as part of its data processing pipeline?

### Grading Guidelines (100 Total Points)

* Coding rules, organization, and aesthetics (20 points). Effective use of Go modules and idiomatic Go. Code should be readable, easy to understand. Variable and function names should be meaningful, specific rather than abstract. They should not be too long or too short. Avoid useless temporary variables and intermediate results. Code blocks and line breaks should be clean and consistent. Break large code blocks into smaller blocks that accomplish one task at a time. Utilize readable and easy-to-follow control flow (if/else blocks and for loops). Distribute the not rather than the switch (and/or) in complex Boolean expressions. Programs should be self-documenting, with comments explaining the logic behind the code (McConnell 2004, 777–817).
* Testing and software metrics (20 points). Employ unit tests of critical components, generating synthetic test data when appropriate. Generate program logs and profiles when appropriate. Monitor memory and processing requirements of code components and the entire program. If noted in the requirements definition, conduct a Monte Carlo performance benchmark.
* Design and development (20 points). Employ a clean, efficient, and easy-to-understand design that meets all aspects of the requirements definition and serves the use case. When possible, develop general-purpose code modules that can be reused in other programming projects.
* Documentation (20 points). Effective use of Git/GitHub, including a README.md Markdown file for each repository, noting the roles of programs and data and explaining how to test and use the application.
* Application (20 points). Delivery of an executable load module or application (.exe file for Windows or .app file for MacOS). The application should run to completion without issues. If user input is required, the application should check for valid/usable input and should provide appropriate explanation to users who provide incorrect input. The application should employ clean design for the user experience and user interface (UX/UI).

### Assignment Deliverables

* Text string showing the link (URL) to the GitHub repository for the assignment
* README.md Markdown text file documentation for the assignment
* Zip compressed file with executable load module for the program/application (.exe for Windows or .app for MacOS)