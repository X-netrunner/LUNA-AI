#!/usr/bin/env node
// luna-whapp — local WhatsApp bridge for Luna.
//
// Links your WhatsApp account once over QR (like WhatsApp Web) using the
// Baileys client, keeps the session alive, and exposes a tiny HTTP API on
// 127.0.0.1 so Luna (or anything local) can send messages:
//
//   GET  /health     → { ok, connected, loggedIn }
//   POST /send       → body { to, text }, header Authorization: Bearer <token>
//                      to = full international number, digits only is fine
//                      (e.g. 15551234567) — no leading + or spaces.
//
// Subcommands:
//   serve   keep socket connected + serve HTTP (normal mode, one process)
//   link    print the QR code and exit once linked (used by `luna --whatsapp-link`)
//   status  report session/link state (no network needed)
//   send    <to> <text>  direct send through a running bridge
//
// Data lives in ~/.local/share/whapp/  (override with $WHAPP_DIR):
//   config.json  { port, token }          token generated on first run
//   session/     Baileys auth state       linked via QR, persists across restarts
//   bridge.log   bridge lifecycle log

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import {
  makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
} from '@whiskeysockets/baileys';
import QRCode from 'qrcode-terminal';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HOME = process.env.HOME || process.env.USERPROFILE || '.';
const DATA_DIR = process.env.WHAPP_DIR || path.join(HOME, '.local', 'share', 'whapp');
const CONFIG_PATH = path.join(DATA_DIR, 'config.json');
const SESSION_DIR = path.join(DATA_DIR, 'session');
const LOG_PATH = process.env.WHAPP_LOG || path.join(DATA_DIR, 'bridge.log');

function log(...parts) {
  const line = `[${new Date().toISOString()}] ${parts.join(' ')}\n`;
  try {
    fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
    fs.appendFileSync(LOG_PATH, line);
  } catch {}
}

function loadConfig() {
  let cfg = { port: 7373, token: null };
  try {
    cfg = { ...cfg, ...JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) };
  } catch {}
  if (!cfg.token) {
    cfg.token = crypto.randomBytes(24).toString('hex');
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + '\n', { mode: 0o600 });
  }
  return cfg;
}

function toJid(raw) {
  const digits = String(raw).replace(/\D/g, '');
  if (digits.length < 11 || digits.length > 15) {
    throw new Error(
      'to must be a full international number (country code + number), e.g. 15551234567'
    );
  }
  return `${digits}@s.whatsapp.net`;
}

// ── Baileys socket (links via QR, reconnects, persists session) ───────────────

function createSocket({ onQr, onOpen, onLoggedOut }) {
  let sock = null;
  let connecting = false;

  async function connect() {
    if (connecting) return;
    connecting = true;
    const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
    const version = await fetchLatestBaileysVersion();
    sock = makeWASocket({
      auth: state,
      version: version[0],
      browser: ['Luna', 'Chrome', '23'],
      syncFullHistory: false,
      markOnlineOnConnect: false,
      logger: { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} },
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', async (update) => {
      const { connection, lastDisconnect, qr } = update;
      if (qr && onQr) onQr(qr);
      if (connection === 'open') {
        log('connected');
        if (onOpen) onOpen();
      } else if (connection === 'close') {
        connecting = false;
        const code = lastDisconnect?.error?.output?.statusCode;
        log(`connection closed, statusCode=${code}`);
        if (code === DisconnectReason.loggedOut) {
          try { fs.rmSync(SESSION_DIR, { recursive: true, force: true }); } catch {}
          log('logged out — session cleared');
          if (onLoggedOut) onLoggedOut();
        }
        setTimeout(connect, 3000); // reconnect after a pause
      }
    });
  }

  connect();
  return { get: () => sock };
}

// ── HTTP API ──────────────────────────────────────────────────────────────────

function startHttp(getSocket, cfg) {
  const server = http.createServer(async (req, res) => {
    const send = (code, body) => {
      res.writeHead(code, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(body));
    };
    const auth = req.headers.authorization || '';
    const provided = auth.replace(/^Bearer\s+/i, '');
    const okToken =
      typeof cfg.token === 'string' &&
      provided.length > 0 &&
      crypto.timingSafeEqual(Buffer.from(provided), Buffer.from(cfg.token));

    if (req.url === '/health' && req.method === 'GET') {
      const sock = getSocket();
      return send(200, {
        ok: true,
        connected: !!sock?.user,
        loggedIn: !!sock?.user,
        port: cfg.port,
      });
    }

    if (!okToken) return send(401, { ok: false, error: 'unauthorized' });

    if (req.url === '/send' && req.method === 'POST') {
      let body = '';
      for await (const chunk of req) body += chunk;
      let parsed;
      try { parsed = JSON.parse(body || '{}'); } catch { return send(400, { ok: false, error: 'invalid JSON' }); }
      const sock = getSocket();
      if (!sock?.user) return send(503, { ok: false, error: 'bridge not linked — run `luna --whatsapp-link` to scan the QR' });
      const text = String(parsed.text || '').trim();
      if (!text) return send(400, { ok: false, error: 'text is required' });
      let jid;
      try { jid = toJid(parsed.to ?? ''); } catch (e) { return send(400, { ok: false, error: e.message }); }
      try {
        const msg = await sock.sendMessage(jid, { text });
        log(`sent ${text.length} chars to ${jid}`);
        return send(200, { ok: true, id: msg?.key?.id || null });
      } catch (e) {
        log(`send failed: ${e.message}`);
        return send(500, { ok: false, error: `send failed: ${e.message}` });
      }
    }

    return send(404, { ok: false, error: 'not found' });
  });

  server.on('error', (e) => {
    if (e.code === 'EADDRINUSE') {
      console.error(`Port ${cfg.port} already in use — is another bridge running?`);
      process.exit(1);
    }
    throw e;
  });

  server.listen(cfg.port, '127.0.0.1', () => log(`listening on 127.0.0.1:${cfg.port}`));
}

// ── Subcommands ───────────────────────────────────────────────────────────────

const cmd = process.argv[2];

if (cmd === 'link') {
  const cfg = loadConfig();
  createSocket({
    onQr: (qr) => {
      console.log('\nScan this QR in WhatsApp → Settings → Linked devices:\n');
      QRCode.generate(qr, { small: true });
    },
    onOpen: () => {
      console.log('\nLinked. WhatsApp bridge ready.');
      log('linked via QR');
      process.exit(0);
    },
    onLoggedOut: () => console.log('\nSession was logged out — scanning a fresh QR next.\n'),
  });
} else if (cmd === 'serve') {
  const cfg = loadConfig();
  const { get } = createSocket({
    onQr: (qr) => {
      console.log('\nNot linked yet — scan this QR in WhatsApp → Linked devices:\n');
      QRCode.generate(qr, { small: true });
    },
    onLoggedOut: () => console.log('\nLogged out — a fresh QR will appear, rescan to relink.\n'),
  });
  startHttp(get, cfg);
  console.log(`Luna WhatsApp bridge listening on 127.0.0.1:${cfg.port} (health: /health)`);
} else if (cmd === 'status') {
  const cfg = loadConfig();
  const linked = fs.existsSync(path.join(SESSION_DIR, 'creds.json'));
  console.log(`port:        ${cfg.port}`);
  console.log(`token set:   ${!!cfg.token}`);
  console.log(`linked:      ${linked}`);
  if (linked) console.log('(see /health on the running bridge for live connection state)');
} else if (cmd === 'send') {
  const to = process.argv[3];
  const text = process.argv.slice(4).join(' ');
  if (!to || !text) {
    console.error('usage: node bridge.mjs send <to> <text>');
    process.exit(1);
  }
  const cfg = loadConfig();
  const body = JSON.stringify({ to, text });
  const req = http.request({
    host: '127.0.0.1',
    port: cfg.port,
    path: '/send',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.token}` },
  }, (res) => {
    let out = '';
    res.on('data', (c) => (out += c));
    res.on('end', () => {
      console.log(out);
      process.exit(res.statusCode === 200 ? 0 : 1);
    });
  });
  req.on('error', (e) => { console.error(`bridge unreachable: ${e.message}`); process.exit(1); });
  req.write(body);
  req.end();
} else {
  console.error(`usage: node ${path.relative('.', import.meta.url)} { serve | link | status | send }`);
  process.exit(1);
}