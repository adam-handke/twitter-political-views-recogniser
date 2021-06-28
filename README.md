# Twitter political views recogniser
Python command line program for recognising political views of Polish Twitter accounts. Draws a 3-axis political compass for a given Twitter account based on last 1000 tweets of that account. Training data is based on public <a href="https://mypolitics.pl/" title="myPolitics">myPolitics</a> test results and public Twitter accounts (last 1000 tweets of each user from the dataset). Loading tweets thanks to <a href="https://github.com/JustAnotherArchivist/snscrape" title="snscrape">snscrape</a>. Compass template based on <a href="https://sapplyvalues.github.io/" title="SapplyValues">SapplyValues</a>. 

## myPolitics dataset
As we can see on the picture below, the data distribution is not balanced. Authoritarian and Conservative political views are underrepresented. It may be a result of less interest of people with such views in publishing their political compasses. It should be taken into account when analysing results.

![Dataset compass](https://github.com/adam-handke/twitter-political-views-recogniser/blob/main/dataset_compass.png?raw=true)

## Requirements
Listed in file *requirements.txt*.

## Running

### Load tweets for training
```bash
python twitter_scraper.py
```

### Train the models (one for each axis)
```bash
python political_views_recogniser.py -train
```

### Get the most influential words in each model
```bash
python political_views_recogniser.py -most
```

### Recognise political views of a twitter user
```bash
python political_views_recogniser.py [twitter_username]
```

## Example
```bash
python political_views_recogniser.py szymon_holownia
```
![Example compass](https://github.com/adam-handke/twitter-political-views-recogniser/blob/main/example.png?raw=true)
