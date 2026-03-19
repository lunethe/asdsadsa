import os
from train.reward_copyleaks import score_texts_sync

proxy_file = os.getenv('PROXY_FILE', '')
print(f'Proxy file: {proxy_file!r}')

texts = [
    'Artificial intelligence has significantly transformed various industries '
    'and continues to reshape how organizations operate and deliver value to '
    'their customers around the world today in many ways.',
    'I think AI is pretty cool honestly. It changed a lot of stuff we do '
    'every single day, and honestly my phone autocorrects way better now.',
]

results = score_texts_sync(texts, num_workers=1, proxy_file=proxy_file or None)
for r in results:
    print(f'AI={r.ai_probability:.2%} reward={r.reward:.3f} error={r.error} | {r.text_preview!r}')
