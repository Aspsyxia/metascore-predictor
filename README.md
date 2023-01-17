# Metascore predictor
A simple model that predicts metascore of video games given by the user based on their review using linear regression and neautral language processing methods.

## Goal
For a final assigment for my neautral language processing course, I decided to prepeare something connected with my area of interest. Metacritic is my go-to site
when it comes to any major game release, and despite having no time right now, I often visit it to know what is worth checking out in the future.

I thought it would be interesting to analyze reviews and based on that analysis, train a model that would be able to roughly guess the metascore given by the author
of the review. It's also worth mentioning, that I decided to focus on user reviews, as they are likely to be more random and misleading, so our model will be somewhat
immune to some of the less obvious reviews.

## How it works
For now, I decided to split the whole predictor into 3 python scripts - one that would setup my data (reviews for each different game) into one .csv file, another one
that would perform semantic analysis on each of the review and produce another .csv file with those data and finally, the last one that takes the analyzed data
and with their use trains predicting model.

#### NLP part
For semantic analysis I used Vader, a very well performing semantic analyzer from NLTK library. From output of it's work done on each of the reviews I took compound value,
but due to some of the reviews being sort of misleading, I had to do a slight correction.

For example : someone writes a very positive review, but forgets that the scale goes from 0 - 10, and gives game 4, thinking it was 4/5. Whatever reason, it leads
to very problematic situation, in which model gets confused in learning process. It receives a positive compound vaule, but the score is rather low.

That's why I simply multiplied each misleading compound value by -1. It's not a perfect solution, but it worked really well for now.

#### ML part
Machine learning part is rather straight forward - it's a simple linear regression model built with use of sklearn library, in which I use compound values to determine the score.
Use of polynomial features improved results by a considerable amount.

I think it's worth mentioning the presence of "dumb model" in my code. It's just used to compare my model and make sure it doesn't get worse results from "model" that doesn't
rely on any ML techniques. It was most useful in first moments of training my model, but I thought it wouldn't hurt to leave it for comparasion.

## Data
As for data, I used beautifly crafted datasets gathered by Ellie Lockhart (https://github.com/EllieLockhart/Metacritic---Rotten-Tomatoes-Controversial-Reviews-Dataset.git).
Out of all datasets provided, I decided to use the ones that weren't affected by review bombing, cause they would produce most accurate results for avarage game.

I also had to remove some of the good reviews, cause my model was fed with too many good scores and couldn't handle lower ones.

## Future improvments
I'd like to make it more user friendly and add procedures that could count overall user metascore based on reviews given, but those are all simple improvements and changes.

A bit more challanging thing I thought of would be searching for review bombing interviews and tagging them correctly, so they can be treated diffrently in final metascore result.
