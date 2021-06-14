# Twitter political views recogniser
Python command line program for recognising political views of Polish twitter accounts. Draws a 3-axis political compass for a given twitter account based on last 1000 tweets of that account. Training data based on public <a href="https://mypolitics.pl/" title="myPolitics">myPolitics</a> test results and public Twitter accounts. Loading tweets thanks to <a href="https://github.com/JustAnotherArchivist/snscrape" title="snscrape">snscrape</a>. Compass template based on <a href="https://sapplyvalues.github.io/" title="SapplyValues">SapplyValues</a>. 

## Requirements
* joblib	1.0.1
* matplotlib	3.3.4
* nltk	3.6.2
* numpy	1.20.2
* pandas	1.2.4
* pillow	8.2.0
* pip	21.0.1
* pl-core-news-sm	2.3.0	
* python	3.9.2
* regex	2021.4.4
* scikit-learn	0.24.2
* scipy	1.6.2
* snscrape	0.3.5.dev99+gf44b397	
* spacy	2.3.5
* stop-words	2018.7.23	
* urllib3	1.26.4

## Running

### Load tweets for training
> python twitter_scraper.py

### Train the models (one for each axis)
> python political_views_recogniser.py -train

### Get the most influential words in each model
> python political_views_recogniser.py -most

### Recognise political views of a twitter user
> python political_views_recogniser.py [twitter_username]
