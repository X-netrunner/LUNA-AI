//! tui/log.rs — In-memory log capture for the TUI debug panel
//!
//! In TUI mode the tracing subscriber writes here instead of stderr, so
//! `tracing::debug!` lines stop spraying over the alternate screen and the
//! input box. The right-hand debug panel renders the tail of this buffer.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};

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