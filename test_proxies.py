import os, asyncio, json, httpx
from train.reward_copyleaks import _load_proxies, CopyleaksWorker

proxy_file = os.getenv('PROXY_FILE', '')
print(f'Proxy file: {proxy_file!r}')
proxies = _load_proxies(proxy_file)
print(f'Proxies loaded: {len(proxies)}')
if proxies:
    print(f'First proxy: {proxies[0]}')

async def main():
    proxy = proxies[0] if proxies else None
    worker = CopyleaksWorker(worker_id=0, proxy=proxy)

    # Step 1: check proxy IP via httpx (outside the browser CSP)
    if proxy:
        try:
            async with httpx.AsyncClient(proxies=proxy, timeout=10) as client:
                r = await client.get('https://api.ipify.org?format=json')
                print(f'Proxy IP: {r.text}')
        except Exception as e:
            print(f'IP check failed: {e}')
    else:
        print('No proxy — using direct IP')

    await worker.start()

    # Step 2: score a text via UI automation
    result = await worker.score(
        'Artificial intelligence has significantly transformed various industries '
        'and continues to reshape how organizations operate and deliver value to '
        'their customers around the world today in many ways. The rapid advancement '
        'of machine learning and deep neural networks has enabled computers to perform '
        'tasks previously requiring human intelligence across many domains and sectors.'
    )
    print(f'AI={result.ai_probability:.2%} reward={result.reward:.3f} error={result.error}')
    await worker.stop()

asyncio.run(main())
