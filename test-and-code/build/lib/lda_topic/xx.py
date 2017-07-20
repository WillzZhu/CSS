def test(min_df, max_df):
    cv = CountVectorizer(stop_words='english', min_df=min_df, max_df=max_df)
    cv.fit_transform(np.array(['I am happy.', 'I am strong and happy', 'I was sad.']))
    return cv
