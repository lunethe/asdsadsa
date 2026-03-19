from train.reward_copyleaks import _load_proxies, score_texts_sync

proxies = _load_proxies('./proxies.txt')
print(f'Loaded {len(proxies)} proxies:')
for p in proxies[:5]:
    print(f'  {p}')

results = score_texts_sync(
    ['Artificial intelligence has significantly transformed various industries and continues to reshape how organizations operate and deliver value to their customers around the world today in many ways.'],
    num_workers=1,
    proxy_file='./proxies.txt',
)
print(f'Test score: AI={results[0].ai_probability:.2%} reward={results[0].reward:.3f} error={results[0].error}')
