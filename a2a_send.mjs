#!/usr/bin/env node
// Sends ONE forecasting request to GUARD.IAN over A2A and prints his raw A2A task as JSON.
//
// It uses upstream's own client, A2AClient from @xpr-agents/sdk (XPRNetwork/xpr-agents sdk/src/A2AClient.ts), so the
// message shape, the EOSIO request signing (caller + timestamp + audience + chain id + body hash) and the
// "redirect: 'error'" rule are exactly upstream's. Nothing here signs or builds a request by hand.
//
// Input:  JSON on stdin, {"text": "<request text>"}
// Output: the A2A task JSON on stdout (exit 0)
// Errors: one line on stderr; exit 2 = configuration, exit 3 = the A2A call failed (timeout, HTTP, JSON-RPC error)
//
// Environment (see .env.template):
//   GUARD_IAN_A2A_ENDPOINT  base URL of GUARD.IAN's agent (the endpoint registered on chain); https only
//   GUARD_IAN_ACCOUNT       his XPR account: the caller AND the audience (he forecasts for his own bot)
//   A2A_SIGNING_KEY         private key of a dedicated `a2a` permission on that account, with no linkauth.
//                           Never the active key, never the `guard` key. Read here only, never printed.
//   GUARD_IAN_CHAIN         mainnet (default) or testnet; picks the chain id from the SDK's NETWORKS
//   GUARD_IAN_A2A_TIMEOUT_S request timeout in seconds (default 300; the SDK default of 30 s is too short for a brain run)
import sdk from '@xpr-agents/sdk';

const { A2AClient, NETWORKS } = sdk;

const signingKey = process.env.A2A_SIGNING_KEY || '';

function fail(code, message) {
  const safe = signingKey ? String(message).split(signingKey).join('[redacted]') : String(message);
  process.stderr.write(`a2a_send: ${safe}\n`);
  process.exit(code);
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

const endpoint = process.env.GUARD_IAN_A2A_ENDPOINT || '';
const account = process.env.GUARD_IAN_ACCOUNT || '';
const chain = (process.env.GUARD_IAN_CHAIN || 'mainnet').toUpperCase();
const timeoutMs = Number(process.env.GUARD_IAN_A2A_TIMEOUT_S || 300) * 1000;

if (!endpoint || !account || !signingKey) {
  fail(2, 'GUARD_IAN_A2A_ENDPOINT, GUARD_IAN_ACCOUNT and A2A_SIGNING_KEY must all be set');
}
let url;
try {
  url = new URL(endpoint);
} catch {
  fail(2, 'GUARD_IAN_A2A_ENDPOINT is not a valid URL');
}
const isLocalTest = url.protocol === 'http:' && (url.hostname === '127.0.0.1' || url.hostname === 'localhost');
if (url.protocol !== 'https:' && !isLocalTest) fail(2, 'GUARD_IAN_A2A_ENDPOINT must use https');
if (!/^[a-z1-5.]{1,12}$/.test(account)) fail(2, 'GUARD_IAN_ACCOUNT is not a valid XPR account name');
const network = NETWORKS[chain];
if (!network) fail(2, 'GUARD_IAN_CHAIN must be mainnet or testnet');
if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) fail(2, 'GUARD_IAN_A2A_TIMEOUT_S must be a positive number');

let text;
try {
  ({ text } = JSON.parse(await readStdin()));
} catch {
  fail(2, 'stdin must be JSON like {"text": "..."}');
}
if (typeof text !== 'string' || !text.trim()) fail(2, 'request text is empty');

const client = new A2AClient(endpoint, {
  callerAccount: account,
  signingKey,
  targetAccount: account,
  chainId: network.chainId,
  timeout: timeoutMs,
});

try {
  const task = await client.sendMessage({ role: 'user', parts: [{ type: 'text', text }] });
  process.stdout.write(JSON.stringify(task));
} catch (err) {
  const code = err && typeof err.code === 'number' ? ` (code ${err.code})` : '';
  fail(3, `${(err && err.name) || 'Error'}: ${(err && err.message) || err}${code}`);
}
