import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te
from transformers import pipeline


sentences = [
    "Very very nice place - the one you instantly want to come back to! Me, my wife and our teenage kids really loved it. Rooms are super nice, the sea view is outstanding. Breakfast is one of the best we have ever experienced, great choice and everything was super fresh and tasty (including delicious coffee!). The pools and jacuzzi are superb, too. Overall, we were very happy with our stay - and this will be our #1 choice next time we're in this area!",
    "Toilet was full of spider webs and big gigantic spiders, nobody helped to get it clean and a lady came after 5th call to clean the toilet, sadly the big spider got vanished somewhere in the towels and when we requested to have a new room nobody answered and pretended that they dont understand the english anymore. even the bedding was full of dust , as if the room was not opened since a month or something, even though we were made to wait 2 hours as our room was getting prepred... they took 60 zloty for wifi for 2 nights without informing us in advance, the roof top pool was closed with no information beforehand. yes its a big chain so they dont care what guest thinks, but for us its a big no in future."
]

print("\nVader:\n")
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(f"{sentence}\n")
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print("\n")

print("\nText2Emotion:")
for sentence in sentences:
    emotions = te.get_emotion(sentence)
    print(f"\n{sentence}\n")
    for emotion, score in emotions.items():
        print(f"{emotion}: {score}")

print("\nTransformers:\n")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

for sentence in sentences:
    result = classifier(sentence)[0]
    print(f"\n{sentence}\n")
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")
