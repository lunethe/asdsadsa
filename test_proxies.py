import httpx, json

TEXT = ('Artificial intelligence has significantly transformed various industries '
        'and continues to reshape how organizations operate and deliver value to '
        'their customers around the world today in many ways.')

HEADERS = {
    'Content-Type': 'application/json',
    'Referer': 'https://app.copyleaks.com/v1/scan/ai/embedded',
    'Origin': 'https://app.copyleaks.com',
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
}

print('--- Test 1: direct (no proxy) ---')
try:
    r = httpx.post(
        'https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text',
        headers=HEADERS,
        content=json.dumps({'text': TEXT}),
        timeout=30,
        follow_redirects=True,
    )
    print(f'Status: {r.status_code}')
    print(f'Response: {r.text[:300]}')
except Exception as e:
    print(f'Error: {e}')
