//! src/overlay/mod.rs — Luna's on-screen "she's listening" indicator
//!
//! A native Wayland overlay rendered by Luna herself via the wlr-layer-shell
//! protocol: a compact animated pill pinned to the bottom-center of the
//! screen, showing pulsing green rings while listening, amber while thinking,
//! and nothing at all while idle (the surface is unmapped so pointer clicks
//! pass through). No extra windows, no compositor hacks, no Python.
//!
//! The overlay lives inside the wake daemon process as two background
//! threads. Anything else — the headless voice loop or a spawned TUI — drives
//! it through a tiny unix socket at ~/.local/share/luna/overlay.sock with
//! one-line tags: "listening", "thinking", "idle" (see [`signal`]).

use fontdue::{Font, FontSettings};
use std::io::{Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_buffer::WlBuffer;
use wayland_client::protocol::wl_compositor::WlCompositor;
use wayland_client::protocol::wl_output::WlOutput;
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_shm::{Format, WlShm};
use wayland_client::protocol::wl_shm_pool::WlShmPool;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::{
    Layer, ZwlrLayerShellV1,
};
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::{
    Anchor, Event as LayerSurfaceEvent, KeyboardInteractivity, ZwlrLayerSurfaceV1,
};

// ── Surface geometry ──────────────────────────────────────────────────────────
const W: u32 = 380;
const H: u32 = 84;
const STRIDE: i32 = (W * 4) as i32;
const FRAME_BYTES: usize = (W * H * 4) as usize;

/// ARGB color (source alpha in the 4th slot).
type Rgba = (u8, u8, u8, u8);
/// Opaque RGB color.
type Rgb = (u8, u8, u8);

/// What the indicator is showing right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    /// Nothing on screen (surface unmapped).
    Idle,
    /// Green, pulsing — the microphone is open.
    Listening,
    /// Amber — Luna is thinking (LLM turn in flight).
    Thinking,
}

impl State {
    fn parse(tag: &str) -> Option<State> {
        match tag.trim() {
            "idle" => Some(State::Idle),
            "listening" => Some(State::Listening),
            "thinking" => Some(State::Thinking),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            State::Idle => "",
            State::Listening => "listening",
            State::Thinking => "thinking",
        }
    }

    /// Pill background + accent ring/dot color, by state.
    fn palette(self) -> (Rgba, Rgb) {
        match self {
            State::Listening => ((16, 22, 34, 235), (74, 222, 128)),
            State::Thinking => ((26, 16, 24, 235), (251, 191, 36)),
            State::Idle => ((0, 0, 0, 0), (0, 0, 0)),
        }
    }
}

// ── External signals (socket client) ──────────────────────────────────────────

pub fn socket_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".local/share/luna/overlay.sock"))
}

/// Best-effort tag push. No-op when the wake daemon (and thus the overlay
/// socket) isn't running — e.g. plain `luna --tui` sessions without the
/// wake daemon, or a desktop without a Wayland display.
pub fn signal(tag: &str) {
    let Some(path) = socket_path() else { return };
    let Ok(mut sock) = UnixStream::connect(&path) else { return };
    let _ = sock.write_all(tag.as_bytes());
    let _ = sock.write_all(b"\n");
    let _ = sock.shutdown(std::net::Shutdown::Both);
}

// ── Startup ───────────────────────────────────────────────────────────────────

/// Spawn the socket server + Wayland renderer threads. Returns false (with a
/// logged reason) when there is no Wayland display to draw on.
pub fn start() -> bool {
    if std::env::var("WAYLAND_DISPLAY").is_err() {
        tracing::warn!("overlay: no WAYLAND_DISPLAY — desktop indicator disabled");
        return false;
    }
    let Some(path) = socket_path() else { return false };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&path);

    let (tx, rx) = mpsc::unbounded_channel::<State>();
    let tx_sock = tx.clone();
    let sock_path = path.clone();
    std::thread::spawn(move || socket_server(sock_path, tx_sock));
    std::thread::spawn(move || run_wayland(rx));
    tracing::info!("overlay: listening indicator active ({})", path.display());
    true
}

/// Accepts one-line state tags on a unix socket and forwards them to the
/// renderer. Each client connects, writes a tag, and closes.
fn socket_server(path: PathBuf, tx: mpsc::UnboundedSender<State>) {
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("overlay: socket bind failed ({}): {}", path.display(), e);
            return;
        }
    };
    for stream in listener.incoming() {
        let mut sock = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = [0u8; 64];
        let n = match sock.read(&mut buf) {
            Ok(0) => continue, // client closed without writing
            Ok(n) => n,
            Err(_) => continue,
        };
        let tag = String::from_utf8_lossy(&buf[..n]);
        if let Some(st) = State::parse(&tag) {
            tracing::debug!("overlay: socket tag {:?}", tag.trim());
            let _ = tx.send(st);
        }
    }
}

// ── Wayland renderer ──────────────────────────────────────────────────────────

struct Host {
    /// A configure was received (and acked) on the layer surface.
    configured: bool,
}

impl Host {
    fn new() -> Self {
        Host { configured: false }
    }
}

// NOTE: `Dispatch::event` is declared as `fn event(state: &mut State, ...)`
// — a plain parameter (State defaults to Self), NOT a `self` receiver. That
// is why every impl below spells it `state: &mut Self`.
macro_rules! empty_dispatch {
    ($( $t:ty ),+ $(,)?) => {
        $(
            impl Dispatch<$t, ()> for Host {
                fn event(
                    state: &mut Self,
                    _proxy: &$t,
                    _event: <$t as Proxy>::Event,
                    _data: &(),
                    _conn: &Connection,
                    _qhandle: &QueueHandle<Self>,
                ) {
                    let _ = state;
                }
            }
        )+
    };
}

empty_dispatch!(
    ZwlrLayerShellV1,
    WlCompositor,
    WlShm,
    WlShmPool,
    WlBuffer,
    WlSurface,
    WlOutput,
);

// The registry's new-global bookkeeping (registry_queue_init) is an internal
// detail — nothing to do.
impl Dispatch<WlRegistry, GlobalListContents> for Host {
    fn event(
        state: &mut Self,
        _proxy: &WlRegistry,
        _event: <WlRegistry as Proxy>::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        let _ = state;
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for Host {
    fn event(
        state: &mut Self,
        proxy: &ZwlrLayerSurfaceV1,
        event: LayerSurfaceEvent,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            LayerSurfaceEvent::Configure { serial, .. } => {
                proxy.ack_configure(serial);
                state.configured = true;
            }
            LayerSurfaceEvent::Closed => {
                tracing::debug!("overlay: layer surface closed by compositor");
            }
            _ => {} // future/deprecated events
        }
    }
}

// Sends any buffered requests to the compositor. wayland-client buffers all
// requests client-side and only transmits them on flush()/roundtrip() —
// dispatch_pending() does NOT flush, so commits must be flushed explicitly.
fn flush_queue(eq: &EventQueue<Host>) {
    if let Err(e) = eq.flush() {
        tracing::warn!("overlay: flush failed: {e}");
    }
}

fn run_wayland(mut rx: mpsc::UnboundedReceiver<State>) {
    let conn = match Connection::connect_to_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("overlay: wayland connect failed: {}", e);
            return;
        }
    };
    let (globals, mut event_queue) = match registry_queue_init::<Host>(&conn) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("overlay: registry init failed: {}", e);
            return;
        }
    };
    let mut host = Host::new();
    let qh = event_queue.handle();

    let compositor = match globals.bind::<WlCompositor, Host, ()>(&qh, 1..=6, ()) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("overlay: no wl_compositor: {}", e);
            return;
        }
    };
    let shm = match globals.bind::<WlShm, Host, ()>(&qh, 1..=1, ()) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("overlay: no wl_shm: {}", e);
            return;
        }
    };
    let layer_shell = match globals.bind::<ZwlrLayerShellV1, Host, ()>(&qh, 1..=5, ()) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("overlay: compositor has no wlr-layer-shell (is Hyprland/sway running?): {}", e);
            return;
        }
    };
    // First output only — the indicator sits on the primary monitor. None is
    // fine too: wlroots still maps the surface to the "all outputs" fallback.
    let output = globals.bind::<WlOutput, Host, ()>(&qh, 1..=4, ()).ok();

    let surface = compositor.create_surface(&qh, ());
    let layer_surface = layer_shell.get_layer_surface(
        &surface,
        output.as_ref(),
        Layer::Overlay,
        "luna-overlay".to_string(),
        &qh,
        (),
    );
    layer_surface.set_size(W, H);
    layer_surface.set_anchor(Anchor::Bottom);
    layer_surface.set_exclusive_zone(-1); // never reserve space for it
    layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer_surface.set_margin(0, 0, 26, 0); // top, right, bottom, left
    surface.commit();

    // First roundtrip: the compositor replies with the initial configure.
    match event_queue.roundtrip(&mut host) {
        Ok(_) => tracing::debug!(
            "overlay: setup roundtrip ok (initial configure received: {})",
            host.configured
        ),
        Err(e) => {
            tracing::warn!("overlay: initial roundtrip failed: {e}");
            return;
        }
    }

    // Shared-memory pool with two buffers (double buffering).
    // NOTE: must be O_RDWR — mmap(PROT_READ|PROT_WRITE) fails with EACCES on
    // a write-only fd (File::create is O_WRONLY, so don't use it here).
    let shm_file = std::env::temp_dir().join(format!("luna-overlay-{}.shm", std::process::id()));
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&shm_file)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("overlay: shm file create failed: {}", e);
            return;
        }
    };
    let pool_size: i32 = (FRAME_BYTES * 2) as i32;
    if let Err(e) = file.set_len(pool_size as u64) {
        tracing::warn!("overlay: shm resize failed: {}", e);
        return;
    }
    let mut mmap = match unsafe { memmap2::MmapMut::map_mut(&file) } {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("overlay: shm mmap failed: {}", e);
            return;
        }
    };
    let pool = shm.create_pool(file.as_fd(), pool_size, &qh, ());
    let buf_a = pool.create_buffer(
        0,
        W as i32,
        H as i32,
        STRIDE,
        Format::Argb8888,
        &qh,
        (),
    );
    let buf_b = pool.create_buffer(
        FRAME_BYTES as i32,
        W as i32,
        H as i32,
        STRIDE,
        Format::Argb8888,
        &qh,
        (),
    );
    // The mmap stays valid after the file is dropped.
    drop(file);
    let _ = std::fs::remove_file(&shm_file);

    let font = load_font();

    let mut state = State::Idle;
    let mut mapped = false;
    let mut ever_shown = false;
    let mut frame: u64 = 0;

    loop {
        while let Ok(st) = rx.try_recv() {
            if st != state {
                tracing::debug!("overlay: state -> {:?}", st);
            }
            state = st;
        }

        if state == State::Idle {
            // Unmap so the surface (and its clicks) completely disappear.
            if mapped {
                surface.attach(None, 0, 0);
                surface.commit();
                flush_queue(&event_queue);
                mapped = false;
                tracing::debug!("overlay: unmapped");
            }
            if let Err(e) = event_queue.dispatch_pending(&mut host) {
                tracing::warn!("overlay: idle dispatch error: {e}");
            }
            std::thread::sleep(Duration::from_millis(40));
            continue;
        }

        // Show. First time uses the initial configure already received during
        // setup; after an unmap, re-arm via the null-buffer remap dance and
        // wait for a fresh configure (bounded — worst case we map anyway).
        if !mapped {
            if ever_shown {
                surface.attach(None, 0, 0);
                surface.commit();
                flush_queue(&event_queue);
            }
            host.configured = false;
            let deadline = Instant::now() + Duration::from_millis(1500);
            while !host.configured && Instant::now() < deadline {
                if let Err(e) = event_queue.dispatch_pending(&mut host) {
                    tracing::warn!("overlay: show-wait dispatch error: {e}");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            tracing::debug!(
                "overlay: mapped (configure during show-wait: {})",
                host.configured
            );
            host.configured = false;
            mapped = true;
            ever_shown = true;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        draw_frame(&mut mmap, state, now, font.as_ref());

        let buffer = if frame.is_multiple_of(2) { &buf_a } else { &buf_b };
        surface.attach(Some(buffer), 0, 0);
        surface.damage_buffer(0, 0, W as i32, H as i32);
        surface.commit();
        flush_queue(&event_queue);
        if frame == 0 || frame.is_multiple_of(90) {
            tracing::debug!(
                "overlay: frame {} committed ({:?}, pill alpha probe)",
                frame,
                state
            );
        }
        if let Err(e) = event_queue.dispatch_pending(&mut host) {
            tracing::warn!("overlay: main loop dispatch error: {e}");
        }
        frame += 1;
        std::thread::sleep(Duration::from_millis(33)); // ~30 fps
    }
}

// ── Software rasterizer ───────────────────────────────────────────────────────

#[inline]
fn blend_pixel(px: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    if a == 0 {
        return;
    }
    if a == 255 {
        px[0] = b;
        px[1] = g;
        px[2] = r;
        px[3] = 255;
        return;
    }
    let sa = a as f32 / 255.0;
    let da = px[3] as f32 / 255.0;
    let oa = sa + da * (1.0 - sa);
    if oa <= 0.0 {
        return;
    }
    px[0] = ((b as f32 * sa + px[0] as f32 * da * (1.0 - sa)) / oa) as u8;
    px[1] = ((g as f32 * sa + px[1] as f32 * da * (1.0 - sa)) / oa) as u8;
    px[2] = ((r as f32 * sa + px[2] as f32 * da * (1.0 - sa)) / oa) as u8;
    px[3] = (oa * 255.0) as u8;
}

fn draw_frame(buf: &mut [u8], state: State, now: u128, font: Option<&Font>) {
    buf.fill(0);
    if state == State::Idle {
        return;
    }
    let (bg, accent) = state.palette();
    let label = state.label();

    // Pill
    let px = 6i32;
    let py = 22i32;
    let pw = W as i32 - 12;
    let ph = H as i32 - 44;
    let pr = 22i32;
    fill_rounded_rect(buf, px, py, pw, ph, pr, bg);

    // Two trailing pulsing rings around the dot.
    let cx = 34f32;
    let cy = (py + ph / 2) as f32;
    let phase = ((now % 900) as f64) / 900.0;
    for k in 0..2 {
        let p = ((phase + k as f64 * 0.5) % 1.0) as f32;
        let r = 8.0 + 32.0 * p;
        let a = ((1.0 - p as f64) * 120.0 / (k as f64 + 1.0)) as u8;
        draw_ring(buf, cx, cy, r, 2.4, accent, a);
    }

    // Dot
    fill_circle(buf, cx, cy, 6.0, accent, 255);

    // Label
    if let Some(f) = font {
        draw_text(buf, f, label, 16.0, 56, cy as i32 - 6, (235, 240, 250));
    }
}

fn fill_rounded_rect(buf: &mut [u8], x: i32, y: i32, w: i32, h: i32, r: i32, color: Rgba) {
    let (cr, cg, cb, ca) = color;
    let (x1, y1) = (x + w, y + h);
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = x1.min(W as i32);
    let y1 = y1.min(H as i32);
    let rr = r * r;
    for py in y0..y1 {
        for pxx in x0..x1 {
            let inside = if pxx < x + r && py < y + r {
                dist2(pxx, py, x + r, y + r) <= rr
            } else if pxx >= x + w - r && py < y + r {
                dist2(pxx, py, x + w - r - 1, y + r) <= rr
            } else if pxx < x + r && py >= y + h - r {
                dist2(pxx, py, x + r, y + h - r - 1) <= rr
            } else if pxx >= x + w - r && py >= y + h - r {
                dist2(pxx, py, x + w - r - 1, y + h - r - 1) <= rr
            } else {
                true
            };
            if inside {
                blend_pixel(&mut buf[((py * W as i32 + pxx) * 4) as usize..], cr, cg, cb, ca);
            }
        }
    }
}

#[inline]
fn dist2(ax: i32, ay: i32, bx: i32, by: i32) -> i32 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

fn fill_circle(buf: &mut [u8], cx: f32, cy: f32, rad: f32, color: Rgb, alpha: u8) {
    let (cr, cg, cb) = color;
    let r2 = rad * rad;
    let x0 = ((cx - rad - 1.0).max(0.0)) as i32;
    let y0 = ((cy - rad - 1.0).max(0.0)) as i32;
    let x1 = ((cx + rad + 1.0).min(W as f32)) as i32;
    let y1 = ((cy + rad + 1.0).min(H as f32)) as i32;
    for py in y0..y1 {
        for pxx in x0..x1 {
            let dx = pxx as f32 - cx;
            let dy = py as f32 - cy;
            if dx * dx + dy * dy <= r2 {
                blend_pixel(&mut buf[((py * W as i32 + pxx) * 4) as usize..], cr, cg, cb, alpha);
            }
        }
    }
}

fn draw_ring(buf: &mut [u8], cx: f32, cy: f32, r: f32, thick: f32, color: Rgb, alpha: u8) {
    let (cr, cg, cb) = color;
    let outer = r + thick / 2.0 + 1.0;
    let x0 = ((cx - outer).max(0.0)) as i32;
    let y0 = ((cy - outer).max(0.0)) as i32;
    let x1 = ((cx + outer).min(W as f32)) as i32;
    let y1 = ((cy + outer).min(H as f32)) as i32;
    for py in y0..y1 {
        for pxx in x0..x1 {
            let dx = pxx as f32 - cx;
            let dy = py as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if (d - r).abs() <= thick / 2.0 {
                blend_pixel(&mut buf[((py * W as i32 + pxx) * 4) as usize..], cr, cg, cb, alpha);
            }
        }
    }
}

fn draw_text(buf: &mut [u8], font: &Font, text: &str, px: f32, x: i32, y: i32, color: Rgb) {
    let (cr, cg, cb) = color;
    let mut cursor_x = x;
    for ch in text.chars() {
        let (metrics, coverage) = font.rasterize(ch, px);
        for (i, cov) in coverage.iter().enumerate() {
            if *cov == 0 {
                continue;
            }
            let col = i as i32 % metrics.width as i32;
            let row = i as i32 / metrics.width as i32;
            let gx = cursor_x + metrics.xmin + col;
            let gy = y + metrics.ymin + row;
            if gx >= 0 && gy >= 0 && gx < W as i32 && gy < H as i32 {
                let idx = ((gy * W as i32 + gx) * 4) as usize;
                blend_pixel(&mut buf[idx..idx + 4], cr, cg, cb, *cov);
            }
        }
        cursor_x += metrics.advance_width as i32 + 1;
    }
}

fn load_font() -> Option<Font> {
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/TTF/CaskaydiaCoveNerdFont-Bold.ttf",
        "/usr/share/fonts/TTF/CaskaydiaCoveNerdFont-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/NotoSans-Bold.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = Font::from_bytes(bytes, FontSettings::default()) {
                return Some(font);
            }
        }
    }
    tracing::warn!("overlay: no system font found — indicator renders without text");
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_state_tags() {
        assert_eq!(State::parse("listening"), Some(State::Listening));
        assert_eq!(State::parse("thinking\n"), Some(State::Thinking));
        assert_eq!(State::parse("idle"), Some(State::Idle));
        assert_eq!(State::parse("garbage"), None);
        assert_eq!(State::parse(""), None);
    }

    #[test]
    fn labels() {
        assert_eq!(State::Listening.label(), "listening");
        assert_eq!(State::Thinking.label(), "thinking");
        assert_eq!(State::Idle.label(), "");
    }

    #[test]
    fn blend_opaque_replaces() {
        let mut px = [0u8, 0u8, 0u8, 0u8];
        blend_pixel(&mut px, 10, 20, 30, 255);
        assert_eq!(px, [30, 20, 10, 255]); // BGRA layout
    }

    #[test]
    fn blend_transparent_noop() {
        let mut px = [9u8, 8u8, 7u8, 200u8];
        let before = px;
        blend_pixel(&mut px, 1, 2, 3, 0);
        assert_eq!(px, before);
    }

    #[test]
    fn listening_frame_draws_pill_and_dot() {
        // Rasterize one "listening" frame into a full-size buffer and check
        // that the pill + dot actually have green pixels where expected.
        let mut buf = vec![0u8; FRAME_BYTES];
        draw_frame(&mut buf, State::Listening, 450, None);
        // Pill area (left third, vertical middle) should be opaque dark bg.
        let idx = |x: u32, y: u32| ((y * W + x) * 4) as usize;
        // Inside the pill, near the dot: (36, 38) is the dot center.
        let dot = &buf[idx(36, 38)..idx(36, 38) + 4];
        assert_eq!(dot[3], 255, "dot must be opaque");
        // Buffer is BGRA: dot[0]=blue, dot[1]=green, dot[2]=red. Accent green
        // is (74, 222, 128) → G should dominate.
        assert!(
            dot[1] > 150 && dot[2] < 130,
            "dot must be green, got {:?}",
            dot
        );
        // Pill body away from dot/text (x=300, y=30) opaque dark.
        let body = &buf[idx(300, 30)..idx(300, 30) + 4];
        assert!(body[3] > 200, "pill must be opaque, got {:?}", body);
        // Top-left corner (outside the rounded pill) must stay transparent.
        let corner = &buf[idx(2, 2)..idx(2, 2) + 4];
        assert_eq!(corner[3], 0, "outside the pill must be transparent");
    }
}