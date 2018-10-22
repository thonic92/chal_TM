import pandas as pd

def echantionnage_donnees(path,size,replace = False):
  data = pd.read_csv(path,sep = '\t',low_memory=False)
  data['tweet_creation_dt'] = pd.to_datetime(data['tweet_creation_dt'])
  data['tweet_creation_dt'] = data['tweet_creation_dt'].dt.date
  #print(data.head())
  count = data.groupby('tweet_creation_dt')['tweet_creation_dt'].count().reset_index(name = 'count')
  count.plot()
  data = pd.merge(data, count, how='left', on="tweet_creation_dt")
  tmp_sup100 = data[data['count']>size]
  fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
  tmp_sup100_ech = tmp_sup100.groupby('tweet_creation_dt', as_index=False).apply(fn)
  data = data[~data['tweet_creation_dt'].isin(tmp_sup100['tweet_creation_dt'])]
  data = data.append(tmp_sup100_ech)
  count = data.groupby('tweet_creation_dt')['tweet_creation_dt'].count().reset_index(name = 'count')
  count.plot()
  return(data)
