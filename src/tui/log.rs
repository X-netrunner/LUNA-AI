//! tui/log.rs — In-memory log capture for the TUI debug panel
//!
//! In TUI mode the tracing subscriber writes here instead of stderr, so
//! `tracing::debug!` lines stop spraying over the alternate screen and the
//! input box. The right-hand debug panel renders the tail of this buffer.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Rotate the session log once it passes `MAX_LOG_BYTES`, keeping `LOG_KEEP`
/// previous generations.
///
/// This existed because the log was genuinely unbounded: it had reached 336 MB
/// with no cap, and the always-on wake daemon keeps appending to it forever.
/// A log that can fill the disk is worse than no log, because the failure mode
/// is silent.
///
/// Rotation rather than truncation, on purpose — the value of this file is
/// reviewing what happened *after* the fact, so the previous generation is
/// exactly what you want to keep. `session.log.1` is the run before this one.
pub const MAX_LOG_BYTES: u64 = 32 * 1024 * 1024;
/// How many rotated generations to keep. 3 x 32 MB = 96 MB ceiling, which is
/// ~5x smaller than the unbounded file had already reached.
pub const LOG_KEEP: usize = 3;

/// Rotate `path` if it has grown past `max_bytes`.
///
/// Safe to call when two processes append to the same log (luna-wake and
/// luna-daemon both do). Neither holds a long-lived descriptor — each write
/// reopens with `O_APPEND` — so the worst a race can do is move a few lines
/// into the rotated generation. Lines are never lost or interleaved within a
/// line, because a single `write_all` of a short line to an `O_APPEND` fd is
/// atomic on Linux.
fn rotate_if_needed(path: &std::path::Path, max_bytes: u64, keep: usize) {
    let size = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        // No file yet, or it vanished under a concurrent rotation. Nothing to do.
        Err(_) => return,
    };
    if size <= max_bytes {
        return;
    }
    // Drop the oldest generation, then shift the rest up: .2 -> .3, .1 -> .2.
    let oldest = gen_path(path, keep);
    let _ = std::fs::remove_file(&oldest);
    for i in (1..keep).rev() {
        let from = gen_path(path, i);
        let to = gen_path(path, i + 1);
        if from.exists() {
            let _ = std::fs::rename(&from, &to);
        }
    }
    let _ = std::fs::rename(path, gen_path(path, 1));
    // Recreate the current log immediately. Without this, a rotation that
    // fires on the most recent write leaves NO session.log at all until the
    // next line arrives, which breaks `tail -f session.log` and any tooling
    // that opens the path expecting it to exist.
    let _ = std::fs::OpenOptions::new().create(true).append(true).open(path);
}

/// Path of the i-th rotated generation (`i == 1` is the most recent).
fn gen_path(path: &std::path::Path, i: usize) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{i}"));
    PathBuf::from(s)
}

/// Appends every log line to a file on disk (with a timestamp) so a session
/// can be reviewed after the fact. `Write::write` receives whole lines ending
/// in `\n` from the tracing fmt subscriber; each one is prefixed with the
/// current time and appended with a trailing newline.
#[derive(Clone)]
pub struct FileLog {
    path: Arc<PathBuf>,
    max_bytes: u64,
    keep: usize,
    /// Bytes written since the last size check. Checking the size on every
    /// line would add a `stat` syscall per log line forever; checking once
    /// per megabyte means the file can overshoot `max_bytes` by at most
    /// ~1 MB, which is irrelevant next to a 32 MB threshold.
    since_check: Arc<std::sync::atomic::AtomicU64>,
    check_every: u64,
}

/// How many bytes to accumulate before re-checking the file size.
const CHECK_EVERY: u64 = 1024 * 1024;

impl FileLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_limits(path, MAX_LOG_BYTES, LOG_KEEP)
    }

    /// Same writer, explicit rotation limits. Exists so tests can exercise
    /// rotation with kilobyte thresholds instead of writing 32 MB.
    ///
    /// `check_every` is clamped to `max_bytes` so a small threshold is still
    /// polled promptly — otherwise a test-sized cap would never be reached
    /// before the 1 MB polling interval and rotation would look broken.
    fn with_limits(path: impl Into<PathBuf>, max_bytes: u64, keep: usize) -> Self {
        Self {
            path: Arc::new(path.into()),
            max_bytes,
            keep,
            since_check: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            check_every: CHECK_EVERY.min(max_bytes.max(1)),
        }
    }
}

impl Write for FileLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        let now = chrono::Local::now().format("%H:%M:%S%.3f");
        let line = format!("[{}] {}", now, text.trim_end());
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Open in append mode per write — a handful of lines/sec is cheap and
        // guarantees durability even if the process is killed.
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path.as_ref())
        {
            let _ = f.write_all(line.as_bytes());
            let _ = f.write_all(b"\n");
        }

        use std::sync::atomic::Ordering::Relaxed;
        let written = self
            .since_check
            .fetch_add(line.len() as u64 + 1, Relaxed)
            + line.len() as u64
            + 1;
        if written >= self.check_every {
            self.since_check.store(0, Relaxed);
            rotate_if_needed(self.path.as_ref(), self.max_bytes, self.keep);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Bounded ring buffer of log lines shared between the tracing subscriber
/// (writer side) and the TUI (reader side).
pub struct LogBuffer {
    inner: Arc<Mutex<VecDeque<String>>>,
}

impl LogBuffer {
    pub fn new(_capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn clone_handle(&self) -> Arc<Mutex<VecDeque<String>>> {
        self.inner.clone()
    }

    pub fn lines(&self) -> Vec<String> {
        self.inner
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// Implements `std::io::Write` so it can be used as the tracing fmt writer.
/// Splits on newlines and keeps only the most recent `capacity` lines.
#[derive(Clone)]
pub struct BufferWriter {
    log: Arc<Mutex<VecDeque<String>>>,
    pending: Arc<Mutex<String>>,
    capacity: usize,
}

impl BufferWriter {
    pub fn new(log: Arc<Mutex<VecDeque<String>>>, capacity: usize) -> Self {
        Self {
            log,
            pending: Arc::new(Mutex::new(String::new())),
            capacity,
        }
    }
}

impl Write for BufferWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // tracing fmt writes whole lines ending in \n; accumulate partial tail
        let mut pending = self.pending.lock().map_err(|_| {
            std::io::Error::other("log buffer poisoned")
        })?;
        pending.push_str(&String::from_utf8_lossy(buf));
        let log = self.log.clone();
        let capacity = self.capacity;

        let mut complete: Option<String> = None;
        while let Some(pos) = pending.find('\n') {
            let line = pending[..pos].trim_end().to_string();
            pending.drain(..=pos);
            if line.is_empty() {
                continue;
            }
            complete = Some(line);
        }
        drop(pending); // release lock before touching the log mutex

        if let Some(line) = complete {
            if let Ok(mut inner) = log.lock() {
                if inner.len() >= capacity {
                    inner.pop_front();
                }
                inner.push_back(line);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Cloneable handle to a shared writer target (stderr is !Clone, so we box
/// it in an Arc and re-export a Write impl). Used for the terminal half of
/// the non-TUI tee so tracing `.with_writer(move || ..)` can clone per call.
pub struct Shared<W: Send + 'static>(Arc<std::sync::Mutex<W>>);

impl<W: Send + 'static> Clone for Shared<W> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<W: Write + Send> Shared<W> {
    pub fn new(w: W) -> Self {
        Self(Arc::new(std::sync::Mutex::new(w)))
    }
}

impl<W: Write + Send> Write for Shared<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().map_err(|_| std::io::Error::other("locked"))?.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().map_err(|_| std::io::Error::other("locked"))?.flush()
    }
}

/// Writer that fans a single tracing stream out to two targets (e.g. the TUI
/// ring buffer plus an on-disk file). Used so the debug panel and a persistent
/// session log get identical lines.
#[derive(Clone)]
pub struct TeeWriter<A: Write + Clone, B: Write + Clone> {
    a: A,
    b: B,
}

impl<A: Write + Clone, B: Write + Clone> TeeWriter<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Write + Clone, B: Write + Clone> Write for TeeWriter<A, B> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.a.write(buf)?;
        let _ = self.b.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.a.flush()?;
        let _ = self.b.flush();
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch dir per test, so these can run concurrently.
    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("luna_logtest_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_lines(path: &std::path::Path, n: usize, body: &str) {
        write_lines_with(path, n, body, 2 * 1024);
    }

    fn write_lines_with(path: &std::path::Path, n: usize, body: &str, max_bytes: u64) {
        let mut f = FileLog::with_limits(path, max_bytes, LOG_KEEP);
        for _ in 0..n {
            let mut buf = body.to_string();
            buf.push('\n');
            f.write(buf.as_bytes()).unwrap();
        }
    }

    #[test]
    fn a_small_log_is_never_rotated() {
        let d = scratch("small");
        let p = d.join("session.log");
        write_lines(&p, 20, "hello");
        assert!(p.exists(), "current log must survive");
        assert!(!gen_path(&p, 1).exists(), "nothing to rotate yet");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn crossing_the_threshold_rotates_and_preserves_old_lines() {
        let d = scratch("rotate");
        let p = d.join("session.log");
        // ~200 bytes/line x 40 lines = 8 KB; rotate at 2 KB.
        write_lines(&p, 40, &"A".repeat(200));
        assert!(std::fs::metadata(&p).unwrap().len() <= 2 * 1024 + 4096,
                "current log stays near the cap, got {}",
                std::fs::metadata(&p).unwrap().len());
        let g1 = gen_path(&p, 1);
        assert!(g1.exists(), "a rotated generation must exist");
        let old = std::fs::read_to_string(&g1).unwrap();
        assert!(old.contains(&"A".repeat(200)), "old content preserved");
        // The bug being fixed: history must NOT be discarded.
        assert!(old.lines().count() > 1, "previous run must be intact");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rotation_keeps_only_the_configured_number_of_generations() {
        let d = scratch("keep");
        let p = d.join("session.log");
        // Rotate many times over; only `keep` generations may survive.
        for _ in 0..12 {
            write_lines(&p, 40, &"B".repeat(200));
        }
        let mut n = 0;
        for i in 1..=LOG_KEEP + 4 {
            if gen_path(&p, i).exists() {
                n += 1;
            }
        }
        assert!(n <= LOG_KEEP, "kept {n} generations, max is {LOG_KEEP}");
        assert!(p.exists(), "current log must always exist");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn generations_age_out_oldest_first() {
        let d = scratch("age");
        let p = d.join("session.log");
        // One FileLog for the whole test, because the byte counter lives in the
        // writer: a fresh FileLog per run resets it and rotation never fires.
        //
        // Each run writes ~16 KB against a 10 KB threshold. The counter resets
        // on each rotation, so a run between 1x and 2x the threshold rotates
        // exactly ONCE — which is what makes the generation boundaries below
        // unambiguous. A generous generation count keeps eviction out of it,
        // so ORDERING is what is under test.
        let mut f = FileLog::with_limits(&p, 10_000, 10);
        for tag in ["RUN1", "RUN2", "RUN3"] {
            for _ in 0..40 {
                f.write(format!("{}\n", tag.repeat(100)).as_bytes()).unwrap();
            }
            f.write(format!("{tag}-END\n").as_bytes()).unwrap();
        }
        let g1 = std::fs::read_to_string(gen_path(&p, 1)).unwrap();
        let g2 = std::fs::read_to_string(gen_path(&p, 2)).unwrap();
        let g3 = std::fs::read_to_string(gen_path(&p, 3)).unwrap();
        let cur = std::fs::read_to_string(&p).unwrap();
        assert!(cur.contains("RUN3-END"), "live log holds the newest run");
        assert!(g1.contains("RUN3"), ".1 must be newer than .2");
        assert!(g2.contains("RUN2"), ".2 must be newer than .3");
        assert!(g3.contains("RUN1"), ".3 must hold the oldest surviving run");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The flip side of the cap: old runs ARE meant to be dropped. A cap that
    /// never evicts anything is not a cap.
    #[test]
    fn old_generations_are_eventually_evicted() {
        let d = scratch("evict");
        let p = d.join("session.log");
        let mut f = FileLog::with_limits(&p, 10_000, 2);
        for i in 0..10 {
            // ~16 KB per epoch x 10 = 160 KB, so the 10 KB threshold is
            // crossed many times over. With only 2 generations kept, the first
            // epoch cannot possibly survive.
            for _ in 0..40 {
                f.write(format!("EPOCH{i}-{}\n", "x".repeat(380)).as_bytes()).unwrap();
            }
        }
        let all: String = (1..=2)
            .map(|i| std::fs::read_to_string(gen_path(&p, i)).unwrap_or_default())
            .collect::<Vec<_>>()
            .concat()
            + &std::fs::read_to_string(&p).unwrap_or_default();
        assert!(all.contains("EPOCH9"), "newest must survive");
        assert!(!all.contains("EPOCH0"), "oldest must be evicted, not kept");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A rotation must never leave the live log missing — `tail -f
    /// session.log` and any tooling that opens the path depend on it.
    #[test]
    fn the_current_log_always_exists_after_rotation() {
        let d = scratch("exists");
        let p = d.join("session.log");
        for i in 0..20 {
            write_lines(&p, 40, &format!("GEN{i}").repeat(200));
            assert!(p.exists(), "session.log missing after batch {i}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_log_is_not_an_error() {
        let d = scratch("missing");
        let p = d.join("does_not_exist.log");
        rotate_if_needed(&p, 10, 3); // must not panic
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_configured_cap_is_sane() {
        // 336 MB is what the unbounded log actually reached, so the cap has to
        // be well under that to be worth having.
        assert!(MAX_LOG_BYTES * LOG_KEEP as u64 <= 128 * 1024 * 1024,
                "ceiling {MAX_LOG_BYTES}x{LOG_KEEP} is too large");
    }
}
