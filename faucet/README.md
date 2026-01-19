NilzCoin Faucet
================

This is a small FastAPI service that powers a web faucet for NilzCoin. It signs small payouts from a hot wallet and broadcasts them to your public relay node, with rate limiting and optional CAPTCHA.

Quick start

1) Create a faucet wallet and fund it

```bash
python -m wallet.wallet init --label faucet --passphrase "changeme"
# Note the faucet address. Send some NILZ to it so the faucet can pay out.
```

2) Run the public relay node (if not already)

```bash
export NILZ_NODE_UPSTREAM=http://127.0.0.1:5000
python node/edge_node.py
```

3) Run the faucet service

```bash
export NILZ_FAUCET_NODE_URL=http://127.0.0.1:5010
export NILZ_FAUCET_WALLET_FILE=$(pwd)/nilz.wallet
export NILZ_FAUCET_PASSPHRASE=changeme
export NILZ_FAUCET_FROM_LABEL=faucet
export NILZ_FAUCET_AMOUNT=1
export NILZ_FAUCET_INTERVAL_S=3600
export NILZ_FAUCET_DAILY_CAP=100
export NILZ_FAUCET_ALLOW_ORIGINS=https://your-website.example
python faucet/server.py
```

Endpoints

- GET `/health` → Faucet config and today usage summary.
- POST `/claim` → Body `{ "address": "nilz...", "captcha_token": "..." }` returns JSON with submission result.

Integrate on your website

Basic HTML + JS (no CAPTCHA):

```html
<form id="faucet-form">
  <input id="addr" placeholder="Your nilz address" required />
  <button type="submit">Claim</button>
  <div id="msg"></div>
  <script>
    document.getElementById('faucet-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const address = document.getElementById('addr').value.trim();
      const resp = await fetch('https://your-faucet.example/claim', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address })
      });
      const t = await resp.json();
      document.getElementById('msg').textContent = resp.ok ? 'Sent! Check mempool.' : `Error: ${JSON.stringify(t)}`;
    });
  </script>
</form>
```

Adding CAPTCHA (optional)

- For Google reCAPTCHA: set `RECAPTCHA_SECRET` in the faucet env and include the client site key on your page; send the `captcha_token` returned by the client to `/claim`.
- For hCaptcha: set `HCAPTCHA_SECRET` similarly.

Notes & Security

- Treat the faucet wallet as a hot wallet; keep amounts limited and consider running behind a reverse proxy with HTTPS.
- Use `NILZ_FAUCET_ALLOW_ORIGINS` to restrict CORS to your website origin.
- Use a CDN/WAF or reverse proxy (Nginx/Caddy) for rate limiting bursts at the edge if opening publicly.
