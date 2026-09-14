#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  safety_check.sh — system hardening, maintenance & backup
#
#  Modes:
#    light   (default) standard weekly check — ClamAV scans hot dirs only,
#            but escalates to a full home scan if the last full scan is >30
#            days old.
#    full    full-home ClamAV scan, everything else.
#    backup  backup-only: mount /dev/sda1 at /mnt/backup (if needed),
#            incremental rsync, unmount when done.
#    --fallback  exit early if the Luna daemon already ran a check recently.
#
#  Safety rules:
#    - NEVER auto-formats. The drive UUID must match or the backup is skipped.
#    - Only unmounts at the end if THIS run mounted it.
#    - flock-guarded so concurrent runs never overlap.
# ─────────────────────────────────────────────

set -u

# Resolve the human user (this script may run under `root` via systemd)
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    RUN_USER="${SUDO_USER:-$(id -un 1000 2>/dev/null || echo nobody)}"
else
    RUN_USER="${USER:-$(id -un)}"
fi
RUN_HOME=$(getent passwd "$RUN_USER" 2>/dev/null | cut -d: -f6)
[[ -z "$RUN_HOME" ]] && RUN_HOME="$HOME"

LOG_DIR="$RUN_HOME/logs/safety_check"
LOG_FILE="$LOG_DIR/safety_check_$(date +%F).log"
FULL_MARKER="$LOG_DIR/.last_full_scan"
AIDE_MARKER="$LOG_DIR/.last_aide_run"
BACKUP_DEV="${SAFETY_BACKUP_DEV:-/dev/sda1}"
BACKUP_MNT="${SAFETY_BACKUP_MNT:-/mnt/backup}"
BACKUP_UUID="${SAFETY_BACKUP_UUID:-28df6eaf-a08b-41f4-bc89-2478bb619f8d}"
BACKUP_ROOT="$BACKUP_MNT/arch-backup"
LUNA_MARKER="$RUN_HOME/.local/share/luna/last_safety_check"
ERRORS=0
MOUNTED_BY_US=0
LOCK_FILE="$LOG_DIR/.safety_check.lock"
EXEC="$0"

mkdir -p "$LOG_DIR" "$RUN_HOME/.local/share/luna" 2>/dev/null

# ── Lock — one check at a time ─────────────────
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another safety check is already running — exiting."
    exit 0
fi

# ── Modes ──────────────────────────────────────
MODE="light"
FALLBACK=0
for arg in "$@"; do
    case "$arg" in
        light|full|backup) MODE="$arg" ;;
        --fallback) FALLBACK=1 ;;
    esac
done

# ── Fallback guard — daemon-primary, timer-backup ─
if [[ "$FALLBACK" -eq 1 && "$MODE" != "backup" ]]; then
    GUARD_SECS=$(( (${SAFETY_GUARD_DAYS:-2}) * 86400 ))
    if [[ -f "$LUNA_MARKER" ]]; then
        last=$(cat "$LUNA_MARKER" 2>/dev/null)
        if [[ "$last" =~ ^[0-9]+$ ]]; then
            now=$(date +%s)
            age=$(( now - last ))
            if (( age < GUARD_SECS )); then
                echo "Luna daemon already ran a safety check recently — skipping."
                exit 0
            fi
        fi
    fi
fi

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

notify() {
    command -v notify-send >/dev/null 2>&1 || return 0
    notify-send "🛡️ Safety Check" "$1" --urgency="${2:-normal}" 2>/dev/null || true
}

section() {
    log ""
    log "══════════════════════════════════════"
    log " $1"
    log "══════════════════════════════════════"
    notify "$1"
}

check_fail() {
    log "⚠ WARNING: $1"
    ERRORS=$((ERRORS + 1))
}

is_sudo_ok() {
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
        return 0
    fi
    sudo -n true 2>/dev/null
}

# ── Start ──────────────────────────────────────
log "═══════════════════════════════════════════════"
log " Safety Check started — $(date)"
log " Running as: $RUN_USER"
if [[ "$MODE" == "full" ]]; then
    log " Mode: full"
elif [[ "$MODE" == "backup" ]]; then
    log " Mode: backup-only"
else
    log " Mode: light"
fi
log "═══════════════════════════════════════════════"

# ── 1. System Update ───────────────────────────
if [[ "$MODE" != "backup" ]]; then
    section "1/7 — System Update"
    if is_sudo_ok; then
        if sudo pacman -Syu --noconfirm >> "$LOG_FILE" 2>&1; then
            log "✓ System updated successfully"
        else
            check_fail "System update failed"
        fi
    else
        log "↻ No sudo password available — skipping system update"
    fi
fi

# ── 2. ClamAV ─────────────────────────────────
if [[ "$MODE" != "backup" ]]; then
    section "2/7 — ClamAV Malware Scan"
    if command -v freshclam >/dev/null 2>&1; then
        if is_sudo_ok; then
            sudo freshclam >> "$LOG_FILE" 2>&1 || true
        fi
    else
        log "⚠ freshclam not installed, skipping"
    fi
    if command -v clamscan >/dev/null 2>&1; then
        if [[ "$MODE" == "full" ]] || \
           [[ ! -f "$FULL_MARKER" ]] || \
           [[ $(find "$FULL_MARKER" -mtime +30 2>/dev/null) ]]; then
            log "↻ Full home scan (last full scan older than 30 days)..."
            CLAM_TARGET="$RUN_HOME"
            touch "$FULL_MARKER"
            section "2/7 — ClamAV FULL Home Scan"
        else
            CLAM_TARGET="$RUN_HOME/Downloads $RUN_HOME/.config $RUN_HOME/CustomHyprScripts"
        fi
        # shellcheck disable=SC2086
        CLAM_OUT=$(clamscan -r --bell -i $CLAM_TARGET 2>&1)
        echo "$CLAM_OUT" >> "$LOG_FILE"
        INFECTED=$(echo "$CLAM_OUT" | grep "Infected files:" | awk '{print $3}' | tail -1)
        if [[ "${INFECTED:-0}" == "0" ]]; then
            log "✓ No threats found"
        else
            check_fail "ClamAV found $INFECTED infected file(s)!"
            notify "⚠ ClamAV found $INFECTED infected file(s)!" critical
        fi
    else
        log "⚠ ClamAV not installed, skipping"
    fi
fi

# ── 3. Rootkit Check ──────────────────────────
if [[ "$MODE" != "backup" ]]; then
    section "3/7 — rkhunter Rootkit Check"
    if command -v rkhunter >/dev/null 2>&1; then
        if is_sudo_ok; then
            sudo rkhunter --update >> "$LOG_FILE" 2>&1
            sudo rkhunter --check --sk >> "$LOG_FILE" 2>&1
            WARNINGS=$(grep -c "Warning" "$LOG_FILE" || true)
            if [[ "${WARNINGS:-0}" -gt 0 ]]; then
                check_fail "rkhunter found $WARNINGS warning(s) — review log"
                notify "⚠ rkhunter: $WARNINGS warning(s) found" critical
            else
                log "✓ No rootkits detected"
            fi
        else
            log "↻ No sudo password available — skipping rkhunter"
        fi
    else
        log "⚠ rkhunter not installed, skipping"
    fi
fi

# ── 4. UFW Status ─────────────────────────────
if [[ "$MODE" != "backup" ]]; then
    section "4/7 — Firewall Status"
    if command -v ufw >/dev/null 2>&1; then
        if is_sudo_ok; then
            UFW_STATUS=$(sudo ufw status | head -1)
            log "UFW: $UFW_STATUS"
            if echo "$UFW_STATUS" | grep -q "inactive"; then
                check_fail "UFW was inactive — auto-enabling"
                notify "⚠ Firewall was OFF — re-enabling!" critical
                sudo ufw enable >> "$LOG_FILE" 2>&1
                log "↻ UFW auto-enabled"
            else
                log "✓ Firewall is active"
                sudo ufw status verbose >> "$LOG_FILE" 2>&1 || true
            fi
        else
            log "↻ No sudo password available — skipping firewall check"
        fi
    else
        log "⚠ UFW not installed, skipping"
    fi
fi

# ── 5. Lynis Audit ────────────────────────────
if [[ "$MODE" != "backup" ]]; then
    section "5/7 — Lynis Security Audit"
    if command -v lynis >/dev/null 2>&1; then
        if is_sudo_ok; then
            sudo lynis audit system --quick >> "$LOG_FILE" 2>&1 || true
            HARDENING=$(grep "Hardening index" "$LOG_FILE" | tail -1)
            log "✓ $HARDENING"
        else
            log "↻ No sudo password available — skipping lynis"
        fi
    else
        log "⚠ Lynis not installed — run: sudo pacman -S lynis"
    fi
fi

# ── 6. AIDE Integrity Check (monthly) ─────────
if [[ "$MODE" != "backup" ]]; then
    section "6/7 — AIDE File Integrity Check"
    if command -v aide >/dev/null 2>&1; then
        if is_sudo_ok; then
            if [[ ! -f /var/lib/aide/aide.db ]]; then
                log "↻ No AIDE database found — initializing (this takes a while)..."
                notify "AIDE initializing database — please wait..."
                sudo aide --init >> "$LOG_FILE" 2>&1
                sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
                log "✓ AIDE database initialized"
                touch "$AIDE_MARKER"
            else
                if [[ ! -f "$AIDE_MARKER" ]] || \
                   [[ $(find "$AIDE_MARKER" -mtime +30 2>/dev/null) ]]; then
                    log "↻ Running monthly AIDE integrity check..."
                    notify "Running AIDE scan — this may take a few minutes..."
                    AIDE_OUT=$(sudo aide --check 2>&1)
                    echo "$AIDE_OUT" >> "$LOG_FILE"
                    AIDE_CHANGES=$(echo "$AIDE_OUT" | grep -c "changed\|added\|removed" || true)
                    if [[ "${AIDE_CHANGES:-0}" -gt 0 ]]; then
                        check_fail "AIDE detected $AIDE_CHANGES change(s) — review log!"
                        notify "⚠ AIDE: $AIDE_CHANGES file change(s) detected!" critical
                    else
                        log "✓ No unauthorized file changes detected"
                    fi
                    sudo aide --update >> "$LOG_FILE" 2>&1 || true
                    sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db 2>/dev/null || true
                    touch "$AIDE_MARKER"
                else
                    LAST_RUN=$(date -r "$AIDE_MARKER" +%F)
                    log "↻ AIDE last ran on $LAST_RUN — skipping until next month"
                fi
            fi
        else
            log "↻ No sudo password available — skipping AIDE"
        fi
    else
        log "⚠ AIDE not installed — run: sudo pacman -S aide"
    fi
fi

# ── 7. Backup ─────────────────────────────────
section "7/7 — Backup to External Drive"
backup_done=0
if [[ -e "$BACKUP_DEV" ]]; then
    if ! mountpoint -q "$BACKUP_MNT"; then
        log "↻ $BACKUP_DEV already ext4 — skipping format"
        if sudo mkdir -p "$BACKUP_MNT" && sudo mount "$BACKUP_DEV" "$BACKUP_MNT" 2>>"$LOG_FILE"; then
            log "✓ Mounted $BACKUP_DEV at $BACKUP_MNT"
            MOUNTED_BY_US=1
        else
            check_fail "Could not mount $BACKUP_DEV"
        fi
    fi
    if mountpoint -q "$BACKUP_MNT"; then
        # UUID sanity check — never copy to the wrong disk
        mounted_uuid=$(sudo blkid -s UUID -o value "$BACKUP_DEV" 2>/dev/null)
        if [[ -n "$mounted_uuid" && "$mounted_uuid" != "$BACKUP_UUID" ]]; then
            check_fail "$BACKUP_DEV is UUID $mounted_uuid, expected $BACKUP_UUID — refusing to backup"
        else
            mkdir -p "$BACKUP_ROOT"
            PREV=$(ls -1dt "$BACKUP_ROOT"/*/ 2>/dev/null | head -1)
            BACKUP_PATH="$BACKUP_ROOT/$(date +%F_%H%M)"
            mkdir -p "$BACKUP_PATH"
            LINK=""
            [[ -n "$PREV" ]] && LINK="--link-dest=$PREV"
            log "↻ Starting incremental rsync -> $BACKUP_PATH (this will take a while)..."
            if rsync -a $LINK \
                --exclude=".cache" \
                --exclude=".gvfs" \
                --exclude=".local/share/Trash" \
                --exclude=".local/share/luna" \
                --exclude="*.tmp" \
                --exclude="node_modules" \
                --exclude="$BACKUP_MNT" \
                "$RUN_HOME/" "$BACKUP_PATH/" >> "$LOG_FILE" 2>&1; then
                ln -sfn "$BACKUP_PATH" "$BACKUP_ROOT/latest"
                log "✓ Backup completed successfully → $BACKUP_PATH/"
                notify "✓ Backup complete"
                backup_done=1
            else
                check_fail "Backup failed"
                notify "⚠ Backup failed!" critical
            fi
        fi
    fi
else
    log "↻ Backup drive ($BACKUP_DEV) not connected — skipping"
    check_fail "Backup skipped — drive not connected"
    notify "⚠ Backup skipped — drive not connected" critical
fi
if [[ "$MOUNTED_BY_US" -eq 1 ]] && [[ "$backup_done" -eq 1 ]]; then
    sleep 1
    if sudo umount "$BACKUP_MNT" 2>>"$LOG_FILE"; then
        log "✓ Drive unmounted safely"
    else
        log "⚠ Could not unmount (maybe still in use)"
    fi
fi

# ── Summary ───────────────────────────────────
log ""
log "═══════════════════════════════════════════════"
log " Check complete — $(date)"
if [[ "$ERRORS" -eq 0 ]]; then
    log " ✓ All checks passed"
    notify "✓ All checks passed! Log: $LOG_FILE"
else
    log " ⚠ $ERRORS issue(s) found — review $LOG_FILE"
    notify "⚠ $ERRORS issue(s) found — check log!" critical
fi
log "═══════════════════════════════════════════════"
echo "FILENAME=$LOG_FILE"
echo "ERRORS=$ERRORS"

# ── Marker (Luna daemon affinity + fallback guard) ─
echo "$(date +%s)" > "$LUNA_MARKER" 2>/dev/null || true

exit 0