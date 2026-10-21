import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

test('Actual browser helper adds a single session token to every same-origin mutation only', async () => {
  const requests = [];
  const location = new URL('http://localhost:8792/review/index.html');
  let sessionGets = 0;
  const window = { fetch: async (input, options = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location);
    requests.push({ url, options });
    if (url.pathname === '/api/review-session') {
      sessionGets++;
      return { ok: true, status: 200, json: async () => ({ token: 'fixture-token' }) };
    }
    return { ok: true, status: 200 };
  } };
  vm.runInNewContext(readFileSync(new URL('../src/review_security.js', import.meta.url), 'utf8'), { window, location, URL, Headers });
  await Promise.all(['POST', 'PUT', 'PATCH', 'DELETE'].map(method => window.fetch('api/save', { method, headers: { 'Content-Type': 'application/json' }, body: '{"rows":{}}' })));
  assert.equal(sessionGets, 1);
  for (const { options } of requests.filter(r => r.url.pathname.endsWith('/save'))) {
    assert.equal(options.headers.get('X-Review-Token'), 'fixture-token');
    assert.equal(options.headers.get('Content-Type'), 'application/json');
    assert.equal(options.credentials, 'same-origin');
    assert.equal(options.body, '{"rows":{}}');
  }
  await window.fetch('/api/bootstrap');
  await window.fetch('https://external.example/api/save', { method: 'POST' });
  assert.equal(requests.at(-2).options.headers, undefined);
  assert.equal(requests.at(-1).options.headers, undefined);
});
