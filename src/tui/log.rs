//! tui/log.rs — In-memory log capture for the TUI debug panel
//!
//! In TUI mode the tracing subscriber writes here instead of stderr, so
//! `tracing::debug!` lines stop spraying over the alternate screen and the
//! input box. The right-hand debug panel renders the tail of this buffer.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Appends every log line to a file on disk (with a timestamp) so a session
/// can be reviewed after the fact. `Write::write` receives whole lines ending
/// in `\n` from the tracing fmt subscriber; each one is prefixed with the
/// current time and appended with a trailing newline.
#[derive(Clone)]
pub struct FileLog {
    path: Arc<PathBuf>,
}

impl FileLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: Arc::new(path.into()) }
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