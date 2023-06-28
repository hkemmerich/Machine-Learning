import pandas as pd
import snscrape.modules.twitter as sntwitter


# query = 'bitcoin lang:en  -filter:replies'

print('Example: bitcoin lang:en since:2010-01-01 -filter:replies')

query= str(input('Which tweet do you want to search? '))

tweets = []
# limit =  500000
limit = int(input('How much tweet do you want: '))
i = 0

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        print(f'Chegou ao limite {limit}')
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent])
        i = i + 1
        print(f'já foi adicionado {i} tweet')
    
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

df['Date'] = df['Date'].dt.tz_localize(None)

df.to_excel(f'base_de_dados_{query}.xlsx')

print(df.head(20))