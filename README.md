#Data Science Interview Questions Some common questions about machine learning to answer at a data science interview.

 

Define machine learning. What is machine learning?
There are a few different definitions of machine learning are; the field of study that gives computers the ability to learn without being explicitly programmed, this is the 1959 definition by Arthur Samuel. Another by Tom Mitchell in 1977 goes by a computer program is said to learn from experience E with respect to some task T and some performance measure P, if it’s performance on T, as measured by P, improves with experience E. I think about its applications, one example I come to is virtual cellphone keyboards that predict a pressed letter when the word misfits that letter. Usually people ‘fat finger’ a key and the predictive nature of machine learning helps approximate nearby appropriate letters in machine translation. Machine learning is systems built to learn from data and to get better at some task measured by some performance metric. 

Name four types of problems where it shines.
One use of machine learning is an application of spam detection, where a long list of rules becomes hard to manage. Problems that require a lot of hand-tuning or long lists of rules, machine learning makes shorter code that is more simplified and performs better. Writing a traditional spam filtering program would start by looking at spam emails and picking out sender names, particular phrases or other patterns in the email and then you would start adding rules to filter those by one. Machine learning automatically detects these patterns, uses them and judges if they are good predictors in an automated cycle. Human language is another area where machine learning is better used than programming extremely complicated applications. Complex problems with no good solution using traditional approaches are better done with machine learning. Trying to write software that recognizes pitch, tone, and augmentation of sounds in recordings of a conversation would require every letter, and it uses’ sound patterns to be coded line by line. Scaling to every known language would be much made simpler by giving recordings of words and programming an algorithm to detect the patterns from them. Machine learning algorithms much better do fluctuating environments where a machine learning algorithm can scale and adapt to new data. Another use would allow a person to learn from a machine learning algorithm to find trends and patterns by inspecting what it has detected like a spam filter would detect patterns and create a list, this is often referred to as data mining. Machine learning makes getting insight into complex problems and copious amounts of data more accessible. 

Can you name four common unsupervised tasks?
A common unsupervised aim is called clustering which tries to detect groups of similar visitors and firing these into subgroups. Hierarchical clustering algorithms (HCA). Visualization algorithms which output 2D or 3D plots preserving structure to understand interpret for patterns. Dimensionality reduction simplified data, (e.g., merging correlation features), AKA feature extraction. Another unsupervised algorithm is anomaly detection, by training on a normal instance it can detect unusual, defective, or instances that are outliers. Finally another common unsupervised in association rule learning which discovers relationships between attributes. 

What is a labeled training set?
Labeled refers to the data to a machine learning algorithm that has the desired solutions used to compare to instances with unknown classifications. 

What are the two most common supervised? 
Machine learning is commonly used in classification like choosing a class good value or bad value. Machine learning is also used to predict a target numeric value given a set of feature calls predictors known as regression, for example, Francis Galton introduced the term regression to the mean because tall people tend to have shorter children, since tall people have predicting feature, (them being tall), their children are predicted to regress to a standard mean average lower than tall. You could also choose to turn many statistical techniques similar to Multivariate Analysis of Variance into features for prediction and predict instances from the data 

What type of machine learning algorithm would use to allow a robot to walk in various unknown terrains?
Reinforcement learning uses an agent to observe environment, select and perform actions, and get rewards or penalties which it uses to develop the policy to get the most rewards. Deepmind’s Alpha Go & robots learning to walk are examples. 

What type of algorithm would you use to segment your customers into multiple groups?
To segment customers into multiple groups without knowing the groups then clustering is useful if groups are known or labeled then classification algorithms can be used. 

Would you frame the problem of spam detection as a supervised learning problem or an unsupervised problem?
Typically spam email is marked as spam or not and so it is a supervised learning problem. 

What is an online learning system?
Online learning systems are sequentially fed implementations of algorithms that take data in small batches. Training is computationally inexpensive because the batches are small in size, and the online learning system is able to learn about new data when it arrives. Stock prices are a couple that needs to adapt to change rapidly for autonomously, Oline learning systems are useful for limited space or low resources because learned instances can be discarded when instances are aggregated. When a dataset is huge and only a subset can fit into memory online learning systems can load part of the data this is called out-of-core learning. In general online learning, systems can also be referred to as incremental learning. The learning rate is a parameter in online learning systems that when set high will cause quick adaptation new data and vise vrsa quickly forget the old data; a low learning rate will also have more inertia and will learn slowly which will be less sensitive to noise or sequences of non-representative data. Bad data are cautious reasons to set low learning rates, malfunctioning sensors or spamming to rank higher on a system. Machine learning techniques to alert for performance drops anomaly detection algorithms may be used. 

What type of learning algorithm relies on a similarity measure to make predictions?
Instance-based learning takes a basic measure of similarity between two emails, e.g., the number of words that have in common and measure new instances by previous ones by generalizing the distances.

What is the difference between a model parameter and a learning algorithm’s hyperparameter?
A parameter of a model like a linear model with y=theta0 + theta1X, theta0 are the parameters where theta0 adjusts hights and theta1 adjust the slope. In a learning algorithm, the hyperparameters are a parameter of where theta0 adjusts height and theta1 adjusts the slope. In a learning algorithm, the hyperparameters are a parameter of the learning algorithm like the amount of regularization or learning, it is set before training and remains constant to training.

What do model-based learning algorithms search for?
Model-based algorithms use examples to search for parameters that generalize to predictions. Like whether or not money makes people hungry if a model can learn that more money trends with higher hunger in populations across wealthier and wealthier countries it could predict based on income across a countries citizens how hungry they are. 

What is the most common strategy they use to succeed?
Models are tested for high accuracy using a fitness function or cost function, a maximization or minimization. For example, linear regression uses a cost function to measure and minimize the distance between points from the training set and the model's predictions.

How does a model make predictions?
You feed it training examples, and it finds parameters that make the model fit the data. 

Can you name four challenges in machine learning?
Machine learning models perform better depending on the amount of data, and often sufficient data amounts are a problem, in every model the model data it has the better it will perform regardless of the model. Nonrepresentative training data can also be a challenge in machine learning; sampling bias is always a problem. Poor-quality at that is full of noise errors or outliers where instances are out of the norm or instances all together are missing a few features. Irrelevant features are also a challenge in machine learning which can be dressed by feature extraction, feature engineering, feature selection, or creating new features by gathering new data. 

If your model performs great on the training data but generalizes poorly to new instances, what is happening? 
Can you name three possible solutions?
Performing well on training data but poorly on new data is a sign of overfitting aka high bias simplifying the model by reducing the number of attributes in the data or constraints the mode may help. Gather more training data or reducing the noise in training by fixing errors and removing outliers.

What is a test set and why would you want to use it?
A test set is a portion of the data held out, normally 80/20% ratio of data.
