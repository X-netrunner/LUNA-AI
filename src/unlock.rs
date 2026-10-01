//! Developer gate for the security tier's no-refusal mode.
//!
//! # Why this exists, and why it is shaped this way
//!
//! The obvious design — a boolean in `luna.toml` — is not a gate. The toml is
//! writable by the person running Luna, so a boolean in the toml is a speed bump
//! for the owner and nothing else. Editing `security_unrestricted = true` would
//! be the entire "unlock" procedure.
//!
//! So the config is only ever a *request*. It takes effect when a signature over
//! a fixed challenge is present on disk and verifies against the public key in
//! the config. The private key is never stored, never logged, and never written;
//! it exists only in the developer's hands and in the moment they paste it.
//!
//! What that actually buys, stated plainly so it is not oversold:
//!
//!   - Editing `luna.toml` alone does nothing. There must also be a valid
//!     signature on disk.            ← the property that is really wanted
//!   - Forging that signature requires the private key. A receipt cannot be
//!     invented, and editing the file to contain one does not verify.
//!   - Nothing secret is at rest. The receipt is a signature, which is public
//!     information; the public key is in the toml and is meant to be shared.
//!   - Revocation is immediate: blank the public key, or delete the receipt.
//!
//! What it does **not** buy, and this is the important half:
//!
//!   - It is not DRM. The user has the source. `git apply` a patch that deletes
//!     the `is_unlocked` check, or edit the prompt constant, and the gate is
//!     gone. No in-band mechanism can prevent that, and claiming otherwise would
//!     be dishonest.
//!   - It does not constrain the model. It gates one boolean. With the gate
//!     open, the security tier is already reachable and already writes
//!     exploits; measured 8/8 end-to-end with the *scoped* prompt. This changes
//!     the prompt's framing of authorisation, not the tool surface.
//!
//! So: a real gate against casual or accidental changes, and a visible,
//! deliberate act to turn it on. Not a lock against the owner.
//!
//! # Algorithm
//!
//! Ed25519, not RSA. The only pure-Rust RSA crate currently published is a
//! release candidate, and an RC is the wrong thing to build a security check
//! out of. `ring` is already in the dependency tree (via `keyring`), stable and
//! widely audited, and Ed25519 gives the property actually needed here — only
//! the holder of the 32-byte seed can produce a verifying signature — with a
//! 64-character key instead of a 1700-character PEM to paste.
//!
//! If RSA is needed specifically (interop with an existing OpenSSL or PGP
//! identity), the change is contained to this file: `sign` and `verify` are the
//! only two functions that touch the algorithm.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

/// Signed-message prefix.
///
/// Versioned so the challenge can be rotated: bump the suffix and every
/// previously issued receipt stops verifying, which is a clean way to
/// invalidate outstanding unlocks without touching anyone's key.
const CHALLENGE: &[u8] = b"luna/security-tier/unrestricted/v1";

/// Ed25519 sizes. `ring` will not hand back a wrong-length key, so these are
/// checked before anything else touches the bytes.
const SEED_LEN: usize = 32;
const PUBKEY_LEN: usize = 32;
const SIG_LEN: usize = 64;

/// What the TUI shows, and what the effective behaviour is derived from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// The config does not ask for it. Nothing to unlock.
    NotRequested,
    /// The config asks for it, but there is no valid signature on disk.
    ///
    /// Either it was never unlocked, or the receipt was deleted, or the public
    /// key in the config no longer matches the one the receipt was signed for.
    /// All three are the same situation from here: the flag is inert.
    Locked,
    /// The config asks for it and a valid signature is present. The
    /// unrestricted clause is used.
    Unlocked,
    /// No developer public key is configured, so it can never be unlocked.
    /// Distinct from `Locked` because it is permanent without a config edit.
    Unavailable,
}

impl GateState {
    pub fn as_str(self) -> &'static str {
        match self {
            GateState::NotRequested => "OFF",
            GateState::Locked => "OFF (locked)",
            GateState::Unlocked => "ON",
            GateState::Unavailable => "OFF (no developer key configured)",
        }
    }
}

/// The receipt: a signature, and nothing else.
///
/// Under `cfg(test)` the location is overridable so the suite never writes to
/// the real `~/.local/state` and cannot observe another test's receipt. A
/// `OnceLock<Mutex<_>>` rather than `static mut`, so there is no `unsafe` and no
/// aliasing question.
#[cfg(test)]
fn receipt_override() -> &'static std::sync::Mutex<Option<PathBuf>> {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<std::sync::Mutex<Option<PathBuf>>> = OnceLock::new();
    OVERRIDE.get_or_init(|| std::sync::Mutex::new(None))
}

/// Serialises every test that touches the receipt path.
///
/// The override is process-global, so two such tests running concurrently on
/// different threads would corrupt each other — one test's `lock()` deleting
/// another's receipt. `cargo test` runs tests in parallel threads, so this is
/// not hypothetical.
#[cfg(test)]
static TEST_RECEIPT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Redirect the receipt to a scratch file for the life of the guard.
///
/// Exists so no test — in any module — can write a real receipt into the user's
/// `~/.local/state` and leave the real feature silently switched on. Held for a
/// scope, not a call, so the override cannot outlive the test that asked for it.
#[cfg(test)]
pub(crate) struct TestReceipt {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
pub(crate) fn test_receipt(path: PathBuf) -> TestReceipt {
    let guard = TEST_RECEIPT_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("scratch receipt dir");
    }
    *receipt_override().lock().unwrap() = Some(path);
    TestReceipt { _guard: guard }
}

#[cfg(test)]
impl Drop for TestReceipt {
    fn drop(&mut self) {
        if let Some(p) = receipt_override().lock().unwrap().clone() {
            let _ = std::fs::remove_file(&p);
            let _ = p.parent().and_then(|d| std::fs::remove_dir(d).ok());
        }
        *receipt_override().lock().unwrap() = None;
    }
}

/// A unique scratch receipt path. Unique per call so a leaked file cannot be
/// picked up by a later test.
#[cfg(test)]
pub(crate) fn scratch_receipt(label: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!("luna_receipt_{}_{}_{}", std::process::id(), label, n))
        .join("security-unlock.sig")
}

/// The XDG state directory, where receipts belong.
///
/// State rather than config, deliberately: the receipt has to survive a config
/// edit, because the whole point is that editing the config is not sufficient.
fn state_dir() -> PathBuf {
    if let Some(d) = dirs::state_dir() {
        return d;
    }
    // `dirs` returns `Some` on Linux (defaulting to `~/.local/state`), so this
    // is a belt-and-braces path. Notably it must *not* also be applied to the
    // `state_dir()` result above — that is already the final directory, and
    // joining again yields `~/.local/state/.local/state`.
    dirs::home_dir()
        .map(|h| h.join(".local").join("state"))
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn receipt_path() -> PathBuf {
    #[cfg(test)]
    if let Ok(guard) = receipt_override().lock() {
        if let Some(p) = guard.as_ref() {
            return p.clone();
        }
    }

    state_dir().join("luna").join("security-unlock.sig")
}

/// The exact bytes that get signed: the challenge plus the public key.
///
/// Binding the public key into the signed message means a receipt is only ever
/// valid for the key it was issued under. Without this, a signature obtained
/// for one installation's key would verify against any other configured key,
/// which would make the receipt a portable bypass rather than a per-install
/// grant.
fn signed_message(public_key: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(CHALLENGE.len() + public_key.len());
    m.extend_from_slice(CHALLENGE);
    m.extend_from_slice(public_key);
    m
}

fn parse_hex(s: &str, expect_len: usize, what: &str) -> Result<Vec<u8>> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() != expect_len * 2 {
        bail!(
            "{what} must be {expect_len} bytes ({expect_len}*2 hex characters), got {}",
            cleaned.len()
        );
    }
    let bytes = (0..cleaned.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cleaned[i..i + 2], 16))
        .collect::<std::result::Result<Vec<u8>, _>>()
        .with_context(|| format!("{what} is not valid hex"))?;
    Ok(bytes)
}

/// Where the receipt lives, for display in error messages and the TUI.
///
/// The path is under `~/.local/state`, which is the right home for it: it is
/// state, not configuration, and it survives a config edit — which is exactly
/// what makes the gate work.
pub fn receipt_path_display() -> String {
    receipt_path().display().to_string()
}

/// Is a developer public key configured at all?
///
/// An empty or absent key means the feature is permanently unavailable — there
/// is nothing to unlock *against*. This is the "no key, no toggle" case, and it
/// is enforced here rather than in the UI so no code path can skip it.
pub fn has_public_key(public_key: &str) -> bool {
    !public_key.trim().is_empty()
}

/// Does a valid receipt exist for this public key?
pub fn is_unlocked(public_key: &str) -> bool {
    verify_receipt(public_key).is_ok()
}

fn verify_receipt(public_key: &str) -> Result<()> {
    let pk = parse_hex(public_key, PUBKEY_LEN, "developer public key").context(
        "the configured developer public key is not a valid 32-byte hex value, so no signature \
         can ever verify against it",
    )?;
    let sig_path = receipt_path();
    let sig_hex = std::fs::read_to_string(&sig_path).with_context(|| {
        format!(
            "no unlock receipt at {} — the security tier is not unlocked",
            sig_path.display()
        )
    })?;
    let sig = parse_hex(&sig_hex, SIG_LEN, "stored signature")?;
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, &pk)
        .verify(&signed_message(&pk), &sig)
        .map_err(|_| {
            anyhow::anyhow!(
                "the unlock receipt at {} does not verify against the configured public key",
                sig_path.display()
            )
        })
}

/// The one function the rest of Luna asks. Cheap enough to call per request,
/// and it reads a small file rather than doing key math when nothing is set.
pub fn gate_state(security_unrestricted: bool, public_key: &str) -> GateState {
    if !security_unrestricted {
        return GateState::NotRequested;
    }
    if !has_public_key(public_key) {
        return GateState::Unavailable;
    }
    if is_unlocked(public_key) {
        GateState::Unlocked
    } else {
        GateState::Locked
    }
}

/// The effective value of the no-refusal switch.
///
/// This is what the prompt builder must use, never the raw config field. The
/// whole gate is one function call wide on purpose: there is exactly one way to
/// answer "is unrestricted mode on", so it cannot be bypassed by a caller that
/// forgets to check.
pub fn unrestricted_active(security_unrestricted: bool, public_key: &str) -> bool {
    gate_state(security_unrestricted, public_key) == GateState::Unlocked
}

/// Consume a developer key, and issue a receipt if it is the right one.
///
/// The key is used and dropped. It is not stored, not logged, and not written
/// anywhere. Only its signature goes to disk.
pub fn unlock(public_key: &str, private_key: &str) -> Result<()> {
    if !has_public_key(public_key) {
        bail!(
            "No developer public key is configured, so there is nothing to unlock. Generate a \
             keypair with `luna --gen-dev-key` and put the public key in luna.toml as \
             `security_dev_public_key`."
        );
    }
    let configured = parse_hex(public_key, PUBKEY_LEN, "developer public key")?;
    let seed = parse_hex(
        private_key,
        SEED_LEN,
        "developer key",
    )?;

    // Ring takes the seed, not a PKCS#8 wrapper, which is why a 64-character hex
    // string is the whole credential. `from_seed_unchecked` is deliberate: the
    // 32 bytes come from the developer's own generator, and the variant that
    // also checks the public half would only be validating our own output.
    let pair = ring::signature::Ed25519KeyPair::from_seed_unchecked(&seed)
        .map_err(|e| anyhow::anyhow!("developer key was rejected: {e}"))?;

    // `public_key()` is a `KeyPair` trait method, not an inherent one, and the
    // struct also has a private field of the same name — so the trait has to be
    // in scope or the field shadows the method.
    use ring::signature::KeyPair as _;
    let derived = pair.public_key().as_ref();
    // `verify_slices_are_equal` returns Err on mismatch, and the comparison is
    // constant-time. Either the keys match or they do not; the error text is
    // the same in both cases, and no part of either key is echoed.
    if ring::constant_time::verify_slices_are_equal(derived, &configured).is_err() {
        // Deliberately does not say which half was wrong, and does not echo any
        // part of the key. A wrong key is a wrong key; there is nothing useful to
        // add for whoever typed it.
        bail!("That is not the developer key for this installation.");
    }

    let sig = pair.sign(&signed_message(&configured));
    let path = receipt_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    std::fs::write(&path, hex_encode(sig.as_ref()))
        .with_context(|| format!("could not write {}", path.display()))?;
    // The receipt is public data, but 0600 costs nothing and keeps the state
    // directory from advertising that this install is unlocked.
    set_owner_only(&path)?;
    Ok(())
}

/// Revoke: delete the receipt. The config keeps asking, so the TUI will show
/// `OFF (locked)` until unlocked again.
pub fn lock() -> Result<()> {
    let path = receipt_path();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        // Already locked is the desired end state, so this is not an error.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("could not remove {}", path.display())),
    }
}

/// Generate a fresh keypair. For the developer to run once, offline.
///
/// The private half is printed exactly once and is not recoverable afterwards —
/// there is no copy anywhere, which is the point.
pub fn generate_keypair() -> (String, String) {
    let mut bytes = [0u8; SEED_LEN];
    // Cannot fail for SystemRandom; an unwrap here would mean the OS CSPRNG is
    // broken, in which case generating a key is pointless anyway.
    ring::rand::SecureRandom::fill(&mut ring::rand::SystemRandom::new(), &mut bytes)
        .expect("the OS CSPRNG failed");
    let pair = ring::signature::Ed25519KeyPair::from_seed_unchecked(&bytes)
        .expect("a 32-byte seed is always a valid Ed25519 seed");
    use ring::signature::KeyPair as _;
    (hex_encode(pair.public_key().as_ref()), hex_encode(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(unix)]
fn set_owner_only(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("could not restrict permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_owner_only(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Redirects the receipt to a scratch file for the test's duration.
    ///
    /// Uses the shared `test_receipt` guard rather than a private one, so tests
    /// in *other* modules that need the same redirection are serialised against
    /// these. `cargo test` runs tests on parallel threads, and the override is
    /// process-global — a private lock here would let a TUI test delete this
    /// test's receipt mid-run.
    struct Scratch(TestReceipt);

    impl Scratch {
        fn new(name: &str) -> Self {
            Scratch(test_receipt(scratch_receipt(name)))
        }
    }

    fn receipt_file() -> PathBuf {
        receipt_override()
            .lock()
            .unwrap()
            .clone()
            .expect("receipt path not set")
    }

    /// The one property the whole design exists for.
    ///
    /// Setting the flag in the config must do nothing on its own. If this test
    /// ever fails, the feature has become a config boolean with extra steps, and
    /// the gate is decorative.
    #[test]
    fn the_config_flag_alone_does_nothing() {
        let _s = Scratch::new("flag_alone");
        let (pk, sk) = generate_keypair();

        // Asked for, not unlocked.
        assert_eq!(gate_state(true, &pk), GateState::Locked);
        assert!(!unrestricted_active(true, &pk));

        // Not asked for at all.
        assert_eq!(gate_state(false, &pk), GateState::NotRequested);
        assert!(!unrestricted_active(false, &pk));
    }

    #[test]
    fn a_valid_key_unlocks_and_a_missing_one_does_not() {
        let _s = Scratch::new("valid");
        let (pk, sk) = generate_keypair();
        assert!(!unrestricted_active(true, &pk));
        unlock(&pk, &sk).expect("the matching key must unlock");
        assert!(unrestricted_active(true, &pk));
        assert_eq!(gate_state(true, &pk), GateState::Unlocked);

        // Revoking takes effect immediately.
        lock().unwrap();
        assert!(!unrestricted_active(true, &pk));
        assert_eq!(gate_state(true, &pk), GateState::Locked);
    }

    #[test]
    fn the_wrong_key_is_rejected_and_writes_no_receipt() {
        let _s = Scratch::new("wrong");
        let (pk, _) = generate_keypair();
        let (_, other_sk) = generate_keypair();

        let err = unlock(&pk, &other_sk).expect_err("a foreign key must not unlock");
        assert!(
            err.to_string().contains("not the developer key"),
            "unhelpful error: {err}"
        );
        // The critical part: a failed attempt must not leave a usable receipt.
        assert!(!is_unlocked(&pk));
        assert!(!unrestricted_active(true, &pk));
    }

    /// A receipt is only valid for the key it was issued under. Without the
    /// public key being part of the signed message, a signature captured from one
    /// installation would verify against another.
    #[test]
    fn a_receipt_does_not_transfer_to_a_different_key() {
        let _s = Scratch::new("transfer");
        let (pk_a, sk_a) = generate_keypair();
        let (pk_b, _) = generate_keypair();

        unlock(&pk_a, &sk_a).unwrap();
        assert!(is_unlocked(&pk_a));
        assert!(
            !is_unlocked(&pk_b),
            "a receipt issued for key A verified against key B"
        );
    }

    /// The receipt cannot be forged by writing plausible-looking bytes into the
    /// file. It is checked, not merely present.
    #[test]
    fn a_hand_written_receipt_does_not_verify() {
        let _s = Scratch::new("forged");
        let (pk, _) = generate_keypair();

        std::fs::write(receipt_file(), hex_encode(&[0x5au8; SIG_LEN])).unwrap();
        assert!(!is_unlocked(&pk));

        // And a valid-length signature over the *wrong* message.
        let (_, sk) = generate_keypair();
        let seed: [u8; SEED_LEN] = parse_hex(&sk, SEED_LEN, "k").unwrap().try_into().unwrap();
        let pair = ring::signature::Ed25519KeyPair::from_seed_unchecked(&seed).unwrap();
        let sig = pair.sign(b"some other message entirely");
        std::fs::write(receipt_file(), hex_encode(sig.as_ref())).unwrap();
        assert!(
            !is_unlocked(&pk),
            "a signature over a different message verified"
        );
    }

    /// No configured key means permanently unavailable, and unlocking is
    /// impossible rather than merely refused by the UI.
    #[test]
    fn without_a_configured_key_it_can_never_be_unlocked() {
        let _s = Scratch::new("nokey");
        assert!(!has_public_key(""));
        assert!(!has_public_key("   "));
        assert_eq!(gate_state(true, ""), GateState::Unavailable);
        assert_eq!(gate_state(true, "  "), GateState::Unavailable);
        assert!(!unrestricted_active(true, ""));

        // A real, correctly-paired key still must not unlock: with nothing
        // configured there is nothing to verify a signature *against*, so a valid
        // signature would be unverifiable and must be treated as absent.
        let (pk, sk) = generate_keypair();
        let err = unlock("", &sk).expect_err("must not unlock with no configured key");
        assert!(err.to_string().contains("nothing to unlock"), "{err}");

        // Same key, but this time the public key IS configured. Now it unlocks —
        // which is what makes the assertion below meaningful rather than vacuous.
        unlock(&pk, &sk).expect("a matching pair must unlock when a key is configured");
        assert!(unrestricted_active(true, &pk));

        // And the very same receipt is inert against an empty configured key.
        assert!(
            !unrestricted_active(true, ""),
            "a valid receipt took effect with no developer key configured"
        );
        assert_eq!(gate_state(true, ""), GateState::Unavailable);
    }

    /// Garbage input must produce an error, never a panic. This is reachable
    /// from a paste in the TUI, so hostile input is the normal case, not an edge
    /// case.
    #[test]
    fn malformed_input_errors_instead_of_panicking() {
        let _s = Scratch::new("garbage");
        let (pk, _) = generate_keypair();

        for bad in [
            "",
            " ",
            "zz",
            "not hex at all",
            &"0".repeat(63),  // one short
            &"0".repeat(65),  // one long
            &"0".repeat(4096), // far too long
            "../../etc/passwd",
        ] {
            let r = unlock(&pk, bad);
            assert!(r.is_err(), "accepted malformed key {bad:?}");
        }
        assert!(!is_unlocked(&pk));
    }

    /// A malformed *configured* key must disable the feature, not crash the
    /// prompt build. `luna.toml` is hand-edited.
    #[test]
    fn a_malformed_configured_key_disables_rather_than_errors() {
        let _s = Scratch::new("badconfig");
        for bad in ["", "nope", &"0".repeat(65), &"f".repeat(64)] {
            let st = gate_state(true, bad);
            assert!(
                st != GateState::Unlocked,
                "malformed configured key {bad:?} reported unlocked"
            );
            assert!(!unrestricted_active(true, bad));
        }
    }

    /// Whitespace is stripped, because a key pasted out of a terminal or a
    /// wrapped line arrives with newlines in it.
    #[test]
    fn a_wrapped_or_spaced_key_still_works() {
        let _s = Scratch::new("spaces");
        let (pk, sk) = generate_keypair();
        let spaced: String = sk
            .chars()
            .zip(" \n\t".chars().cycle())
            .map(|(c, sep)| format!("{c}{sep}"))
            .collect();
        assert!(spaced.len() > sk.len());
        unlock(&pk, &spaced).expect("a key with whitespace must still work");
        assert!(unrestricted_active(true, &pk));
    }

    #[test]
    fn a_generated_keypair_is_the_right_shape_and_distinct_each_time() {
        let (pk_a, sk_a) = generate_keypair();
        let (pk_b, sk_b) = generate_keypair();
        assert_eq!(pk_a.len(), PUBKEY_LEN * 2);
        assert_eq!(sk_a.len(), SEED_LEN * 2);
        assert!(pk_a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(pk_a, pk_b, "two generated keys collided");
        assert_ne!(sk_a, sk_b);
        // And the pair is coherent: the seed derives that public key.
        let _s = Scratch::new("coherent");
        unlock(&pk_a, &sk_a).expect("a generated pair must be usable");
    }

    /// The receipt must land under the XDG state dir exactly once.
    ///
    /// Regression test. `dirs::state_dir()` already returns `~/.local/state`, so
    /// a fallback that also appends `.local/state` produces
    /// `~/.local/state/.local/state/luna/...`. It still *works* — the writer and
    /// the reader both use the same function — which is why it survived the
    /// first review. But it is a nonsense location, and `XDG_STATE_HOME`
    /// overrides would silently land in a doubled path.
    #[test]
    fn the_receipt_path_is_not_double_nested() {
        // Calls `state_dir()` directly rather than `receipt_path()`, so this is
        // unaffected by whatever per-test override other tests hold — no
        // `Scratch` guard needed.
        let s = state_dir()
            .join("luna")
            .join("security-unlock.sig")
            .to_string_lossy()
            .into_owned();
        assert!(
            !s.contains(".local/state/.local/state"),
            "receipt path is double-nested: {s}"
        );
        assert!(
            !s.contains("/tmp/tmp"),
            "receipt path fell back to a temp dir: {s}"
        );
        assert!(
            s.ends_with("luna/security-unlock.sig"),
            "unexpected receipt path: {s}"
        );
    }

    #[test]
    fn lock_is_idempotent() {
        let _s = Scratch::new("idempotent");
        lock().expect("locking when already locked must succeed");
        let (pk, sk) = generate_keypair();
        unlock(&pk, &sk).unwrap();
        lock().unwrap();
        lock().expect("second lock must also succeed");
        assert!(!unrestricted_active(true, &pk));
    }
}
