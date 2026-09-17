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
  let cfg = { port: 7373, token: null, my_phone: null };
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

function saveConfig(patch) {
  let cfg = { port: 7373, token: null, my_phone: null };
  try { cfg = { ...cfg, ...JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) }; } catch {}
  cfg = { ...cfg, ...patch };
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + '\n', { mode: 0o600 });
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

function extractPhone(jid) {
  if (!jid) return null;
  const m = jid.match(/^(\d+)/);
  return m ? m[1] : null;
}

// Search the aggregated contacts map for a name (case-insensitive substring
// match). Returns [{ jid, phone, name }] (max 5).
function resolveContact(contactsMap, name) {
  if (!contactsMap || !name) return [];
  const q = name.toLowerCase();
  return [...contactsMap.values()]
    .filter((c) => c.name && c.name.toLowerCase().includes(q))
    .slice(0, 5)
    .map((c) => ({
      jid: c.id,
      phone: extractPhone(c.id),
      name: c.name,
    }));
}

// Baileys expects a pino-style logger (with .child()); quietLogger mimics that
// interface and swallows everything so it never clutters our own logs.
function quietLogger() {
  const noop = () => {};
  const logger = {
    level: 'silent',
    trace: noop, debug: noop, info: noop, warn: noop, error: noop, fatal: noop,
    child: () => logger,
  };
  return logger;
}

const CONTACTS_PATH = path.join(DATA_DIR, 'contacts.json');

function loadContacts() {
  try {
    return JSON.parse(fs.readFileSync(CONTACTS_PATH, 'utf8'));
  } catch {
    return [];
  }
}

function persistContacts(contacts) {
  try {
    fs.writeFileSync(CONTACTS_PATH, JSON.stringify(contacts, null, 2) + '\n');
  } catch {}
}

// ── Baileys socket (links via QR, reconnects, persists session) ───────────────
// `giveUpAfter` (number of closed connections without ever opening) makes a
// one-shot caller like `link` stop retrying and exit with a diagnostic instead
// of hanging on a blank screen forever. `serve` keeps retrying indefinitely.

function createSocket({ onQr, onOpen, onLoggedOut, onContacts, giveUpAfter = null }) {
  let sock = null;
  let connecting = false;
  let failures = 0;
  // id (jid) → { id, name } aggregated from WhatsApp's contacts.upsert /
  // contacts.update pushes (their version of the contact book) so name-based
  // sends work without Baileys' removed in-memory store.
  const contacts = new Map(loadContacts().map((c) => [c.id, c]));

  function mergeContacts(incoming) {
    let changed = false;
    for (const c of incoming) {
      const prev = contacts.get(c.id) || {};
      const name = c.name || c.notify || prev.name || prev.notify;
      if (!name) continue;
      contacts.set(c.id, { id: c.id, name });
      changed = true;
    }
    if (changed) {
      persistContacts([...contacts.values()]);
      if (onContacts) onContacts(contacts.size);
    }
  }

  async function connect() {
    if (connecting) return;
    connecting = true;
    console.log('Connecting to WhatsApp servers…');
    const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
    // fetchLatestBaileysVersion returns { version: [major, minor, patch], isLatest }
    const { version } = await fetchLatestBaileysVersion();
    sock = makeWASocket({
      auth: state,
      version,
      browser: ['Luna', 'Chrome', '23'],
      syncFullHistory: false,
      markOnlineOnConnect: false,
      logger: quietLogger(),
    });

    sock.ev.on('creds.update', saveCreds);
    sock.ev.on('contacts.upsert', mergeContacts);
    sock.ev.on('contacts.update', mergeContacts);
    sock.ev.on('connection.update', async (update) => {
      const { connection, lastDisconnect, qr } = update;
      if (qr && onQr) onQr(qr);
      if (connection === 'open') {
        failures = 0;
        log('connected');
        // Persist the logged-in user's phone so Luna can resolve 'myself' later.
        const phone = extractPhone(sock?.user?.id);
        if (phone) {
          log(`logged-in phone: ${phone}`);
          saveConfig({ my_phone: phone });
        }
        if (onOpen) onOpen();
      } else if (connection === 'close') {
        connecting = false;
        const code = lastDisconnect?.error?.output?.statusCode;
        const reason = lastDisconnect?.error?.message || `statusCode=${code ?? 'unknown'}`;
        log(`connection closed, ${reason}`);
        if (code === DisconnectReason.loggedOut) {
          try { fs.rmSync(SESSION_DIR, { recursive: true, force: true }); } catch {}
          log('logged out — session cleared');
          if (onLoggedOut) onLoggedOut();
        } else {
          failures += 1;
          if (giveUpAfter !== null && failures >= giveUpAfter) {
            log(`giving up after ${failures} failed connections`);
            console.error(
              `\nCouldn't reach WhatsApp's servers (${reason}). ` +
                `Your current network may be blocking WhatsApp.\n` +
                `Try a phone hotspot or a different Wi-Fi, then run 'luna --whatsapp-link' again.`
            );
            process.exit(1);
          }
        }
        setTimeout(connect, 3000); // reconnect after a pause
      }
    });
  }

  connect();
  return { get: () => sock, contacts: () => contacts };
}

// ── HTTP API ──────────────────────────────────────────────────────────────────

function startHttp({ get: getSocket, contacts: getContacts }, cfg) {
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
        my_phone: cfg.my_phone || null,
      });
    }

    if (req.url === '/me' && req.method === 'GET') {
      const sock = getSocket();
      if (!sock?.user) return send(503, { ok: false, error: 'bridge not linked' });
      return send(200, {
        ok: true,
        phone: extractPhone(sock.user.id),
        jid: sock.user.id,
        name: sock.user.name || null,
      });
    }

    if (!okToken) return send(401, { ok: false, error: 'unauthorized' });

    if (req.url.startsWith('/resolve') && req.method === 'GET') {
      const sock = getSocket();
      if (!sock?.user) return send(503, { ok: false, error: 'bridge not linked' });
      const name = new URL(req.url, 'http://localhost').searchParams.get('name') || '';
      const contactsMap = getContacts();
      if (contactsMap.size === 0) {
        return send(404, {
          ok: false,
          error:
            "no contacts indexed yet — WhatsApp only pushes them on a device link or when they message you. " +
            "Re-pair once from the terminal (luna --whatsapp-link) to backfill your contact book now.",
        });
      }
      // More lenient match: exact first, then case-insensitive substring,
      // ignoring non-alphanumeric differences (handles "Vani :D" vs "Vani:D" etc.)
      const q = name.toLowerCase().replace(/[^a-z0-9]/g, '');
      const results = [...contactsMap.values()]
        .filter((c) => c.name) // name must exist
        .filter((c) => {
          const clean = c.name.toLowerCase().replace(/[^a-z0-9]/g, '');
          return clean === q || clean.includes(q) || q.includes(clean);
        })
        .slice(0, 5)
        .map((c) => ({
          jid: c.id,
          phone: extractPhone(c.id),
          name: c.name,
        }));
      if (results.length === 0) {
        return send(404, { ok: false, error: `no contact named "${name}" matched` });
      }
      return send(200, { ok: true, matches: results });
    }

    if (req.url.startsWith('/contacts') && req.method === 'GET') {
      const sock = getSocket();
      if (!sock?.user) return send(503, { ok: false, error: 'bridge not linked' });
      const q = new URL(req.url, 'http://localhost').searchParams.get('q') || '';
      const contactsMap = getContacts();
      if (contactsMap.size === 0) {
        // Empty contact book is a real (and recoverable) state, not an error.
        return send(200, { ok: true, contacts: [], total: 0, note: 'no contacts indexed yet' });
      }
      const list = [...contactsMap.values()]
        .filter((c) => c.name && (!q || c.name.toLowerCase().includes(q.toLowerCase())))
        .sort((a, b) => String(a.name).localeCompare(String(b.name)))
        .map((c) => ({ jid: c.id, phone: extractPhone(c.id), name: c.name }));
      return send(200, { ok: true, contacts: list, total: list.length });
    }

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
    giveUpAfter: 4, // don't hang blank if the network blocks WhatsApp
  });
} else if (cmd === 'serve') {
  const cfg = loadConfig();
  const ctx = createSocket({
    onQr: (qr) => {
      console.log('\nNot linked yet — scan this QR in WhatsApp → Linked devices:\n');
      QRCode.generate(qr, { small: true });
    },
    onLoggedOut: () => console.log('\nLogged out — a fresh QR will appear, rescan to relink.\n'),
    onContacts: (n) => log(`contacts indexed: ${n}`),
  });
  startHttp(ctx, cfg);
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