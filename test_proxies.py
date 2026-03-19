import json
from curl_cffi.requests import Session

TEXT = ('Artificial intelligence has significantly transformed various industries '
        'and continues to reshape how organizations operate and deliver value to '
        'their customers around the world today in many ways.')

EMBEDDED_URL = 'https://app.copyleaks.com/v1/scan/ai/embedded'
API_URL = 'https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text'

print('--- Test: curl_cffi Chrome impersonation ---')
try:
    with Session(impersonate='chrome120') as s:
        # Step 1: visit embedded page to get Cloudflare cookies
        print('Visiting embedded page to obtain CF cookies...')
        r = s.get(EMBEDDED_URL, timeout=30)
        print(f'  Page status: {r.status_code}')
        cf_cookies = {k: v for k, v in s.cookies.items()}
        print(f'  Cookies set: {list(cf_cookies.keys())}')

        # Step 2: call the API (cookies sent automatically)
        print('Calling AI scan API...')
        r = s.post(
            API_URL,
            headers={
                'Content-Type': 'application/json',
                'Referer': EMBEDDED_URL,
                'Origin': 'https://app.copyleaks.com',
            },
            data=json.dumps({'text': TEXT}),
            timeout=30,
        )
        print(f'  Status: {r.status_code}')
        print(f'  Response: {r.text[:300]}')
        if r.status_code == 200:
            data = r.json()
            ai = data['summary']['ai']
            print(f'\nSUCCESS — AI={ai:.2%}  Human={1-ai:.2%}')
except Exception as e:
    print(f'Error: {e}')
