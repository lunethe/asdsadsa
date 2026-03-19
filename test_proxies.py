import os, asyncio, json
from train.reward_copyleaks import _load_proxies, CopyleaksWorker

proxy_file = os.getenv('PROXY_FILE', '')
print(f'Proxy file: {proxy_file!r}')
proxies = _load_proxies(proxy_file)
print(f'Proxies loaded: {len(proxies)}')
if proxies:
    print(f'First proxy: {proxies[0]}')

async def main():
    worker = CopyleaksWorker(worker_id=0, proxy=proxies[0] if proxies else None)
    await worker.start()

    # Step 1: check what IP the browser is using
    ip_result = await worker._page.evaluate(
        "fetch('https://api.ipify.org?format=json').then(r=>r.text())"
    )
    print(f'Browser IP: {ip_result}')

    # Step 2: try the API call
    result = await worker.score(
        'Artificial intelligence has significantly transformed various industries '
        'and continues to reshape how organizations operate and deliver value to '
        'their customers around the world today in many ways.'
    )
    print(f'AI={result.ai_probability:.2%} reward={result.reward:.3f} error={result.error}')
    await worker.stop()

asyncio.run(main())
