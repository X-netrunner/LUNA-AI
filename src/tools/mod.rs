//! tools/mod.rs — Tool registry
//!
//! One tool per capability, sized for a 7B model to use reliably.
//! Redundant tools (list_dir, find_binary, open) removed;
//! run_shell handles all of those.
pub mod backup;
pub mod desktop;
pub mod filesystem;
pub mod learn;
pub mod proactive;
pub mod reminders;
pub mod safety;
pub mod security;
pub mod shell;
pub mod spotify;
pub mod sysmode;
pub mod todoist;
pub mod web;
pub mod whatsapp;

use crate::llm::ollama::{ToolCall, ToolDef, ToolFunction};
use anyhow::{Context, Result};
use serde_json::json;

pub fn tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "run_shell".into(),
                description: "Run ANY bash command. Use for: launching apps (append &), \
                              installing packages, file operations, system queries, anything. \
                              For pacman installs use: echo '1' | sudo pacman -S <pkg> --noconfirm \
                              For paru/yay installs use: paru -S <pkg> --noconfirm \
                              ALWAYS use this tool — never describe commands in text.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command to run. Examples: 'kitty &', 'echo 1 | sudo pacman -S htop --noconfirm', 'ls ~'"
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "edit_file".into(),
                description: "Open a file in zeditor for editing. Use for config files, scripts, code.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file (~ is expanded automatically)"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "web_search".into(),
                description: "Search the internet for current information, news,                               package names, how-to guides, or anything Luna doesn't know.                               Returns a text summary of top results.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "nmap_scan".into(),
                description: "Run an nmap scan against a target (IP, hostname, or CIDR range). \
                              Use for network reconnaissance, CTF challenges, or auditing your \
                              own network. Only scan targets you own or have permission to test.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "IP, hostname, or CIDR range, e.g. '192.168.1.1' or '10.0.0.0/24'"
                        },
                        "scan_type": {
                            "type": "string",
                            "enum": ["quick", "full", "ports", "os", "udp"],
                            "description": "quick=fast top ports, full=version+script detection, \
                                           ports=all 65535 ports, os=OS detection (needs sudo), \
                                           udp=top 20 UDP ports"
                        }
                    },
                    "required": ["target", "scan_type"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "analyze_pcap".into(),
                description: "Analyze a packet capture (.pcap/.pcapng) file using tshark. \
                              Use for CTF forensics challenges or investigating captured traffic.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the pcap file" },
                        "mode": {
                            "type": "string",
                            "enum": ["summary", "talkers", "protocols", "http", "dns", "creds"],
                            "description": "summary=file info, talkers=top IP conversations, \
                                           protocols=protocol breakdown, http=HTTP requests, \
                                           dns=DNS queries, creds=look for plaintext credentials"
                        }
                    },
                    "required": ["path", "mode"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "decode_payload".into(),
                description: "Decode an encoded string — common in CTF challenges. \
                              Supports base64, hex, URL encoding, ROT13, and raw binary.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "data": { "type": "string", "description": "The encoded string to decode" },
                        "encoding": {
                            "type": "string",
                            "enum": ["auto", "base64", "hex", "url", "rot13", "binary"],
                            "description": "auto tries base64 then hex; specify if known"
                        }
                    },
                    "required": ["data", "encoding"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "hash_file".into(),
                description: "Compute a cryptographic hash of a file (md5, sha1, sha256, sha512, or all).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "algo": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512", "all"]
                        }
                    },
                    "required": ["path", "algo"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "dns_lookup".into(),
                description: "Look up DNS records, do a reverse lookup, or query whois for a domain/IP.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "target": { "type": "string", "description": "Domain or IP" },
                        "mode": {
                            "type": "string",
                            "enum": ["dns", "reverse", "whois", "mx"]
                        }
                    },
                    "required": ["target", "mode"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "read_file".into(),
                description: "Read and return the full contents of a file.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "write_file".into(),
                description: "Write content to a file, creating it and parent dirs if needed.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "notify".into(),
                description: "Send a desktop notification.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "body": { "type": "string" }
                    },
                    "required": ["title", "body"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "process_stats".into(),
                description: "Show what Luna has learned about process usage on this machine. \
                              Lists each tracked process with how many of the last 14 days it ran, \
                              its idle status, and whether auto-kill is allowed for it.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "allow_autokill".into(),
                description: "Allow the daemon to automatically end a named process after it \
                              has been idle ~30 minutes. Only use when the user explicitly asks \
                              (e.g. 'allow auto-kill steam').".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Process name from /proc, e.g. 'steam'" }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "deny_autokill".into(),
                description: "Revoke auto-kill permission for a named process (removes it from \
                              the allowlist). Use when the user says something like 'never kill \
                              steam' or 'deny auto-kill steam'.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "set_reminder".into(),
                description: "Schedule a reminder that fires as a desktop notification even if \
                              Luna chat is closed. Give EITHER minutes-from-now OR a time of day."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "minutes": { "type": "integer", "description": "Fire this many minutes from now (use this for 'in X minutes')" },
                        "at_time": { "type": "string", "description": "Fire at HH:MM local time, 24h — e.g. '18:30'. Next occurrence." },
                        "text": { "type": "string", "description": "What to remind about" }
                    },
                    "required": ["text"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "list_reminders".into(),
                description: "Show all pending reminders.".into(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "cancel_reminder".into(),
                description: "Cancel a pending reminder by its numeric id (from list_reminders)."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer" }
                    },
                    "required": ["id"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "memory_report".into(),
                description: "Summarize what Luna has permanently learned about the user: shell \
                              workflow facts, system index, and memory stats. Use for 'what do \
                              you know about me / what have you learned'. ALWAYS relay the \
                              returned facts to the user as your answer — never reply with a \
                              menu of options instead.".into(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "find_file".into(),
                description: "Find a file by name anywhere on the system. Returns full path.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Filename to search for e.g. 'luna.toml'"
                        },
                        "search_path": {
                            "type": "string",
                            "description": "Where to search, defaults to home dir"
                        }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "system_info".into(),
                description: "Get system info: battery, cpu, ram, temp, disk, uptime, or all.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "enum": ["battery", "cpu", "ram", "temp", "disk", "uptime", "all"]
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "clipboard".into(),
                description: "Read from or write to the Wayland clipboard.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["read", "write"]
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write (write action only)"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "fetch_page".into(),
                description: "Fetch a webpage and return its text. Use for docs, wiki, current info.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": { "type": "string" }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "remember".into(),
                description: "Save a fact to permanent memory forever. Call proactively when \
                              learning important things about the user, their setup, or preferences.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "fact": { "type": "string" },
                        "category": {
                            "type": "string",
                            "enum": ["user", "system", "preference", "general"]
                        }
                    },
                    "required": ["fact", "category"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "forget".into(),
                description: "Remove facts from permanent memory by keyword.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "keyword": { "type": "string" }
                    },
                    "required": ["keyword"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "list_memories".into(),
                description: "List everything in permanent memory.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "create_skill".into(),
                description: "Save a reusable procedure as a skill (learned from experience). \
                              Call when a conversation shows a repeatable multi-step task, or to \
                              UPDATE an existing skill under the same name when it was wrong or \
                              outdated. name = short kebab-case (e.g. 'arch-package-rebuild'), \
                              description = one line, procedure = step-by-step instructions. \
                              Skills are listed by list_skills, loaded by use_skill, and recalled \
                              automatically whenever relevant.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Skill name (kebab-case)" },
                        "description": { "type": "string", "description": "One-line summary" },
                        "procedure": { "type": "string", "description": "Step-by-step procedure" }
                    },
                    "required": ["name", "procedure"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "use_skill".into(),
                description: "Load a saved skill's full procedure by name so you can apply it. \
                              Use when a saved skill is relevant to the user's request. If the \
                              skill turns out wrong or out of date, re-save the corrected \
                              version with create_skill (same name) to improve it.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Skill name to load" }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "list_skills".into(),
                description: "List every skill Luna has saved, with descriptions and use counts.".into(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "forget_skill".into(),
                description: "Delete a saved skill by name.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "search_history".into(),
                description: "Search everything Luna remembers from PAST conversations: session \
                              titles/summaries and raw turn-by-turn text. Use for 'what did we \
                              talk about last time', 'when did I ask about X', or recalling \
                              previously-discussed details. Give a topic, task name, or phrase.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Topic, task, or phrase to find" }
                    },
                    "required": ["query"]
                }),
            },
        },

       ToolDef {
           r#type: "function".into(),
           function: ToolFunction {
               name: "todoist_list".into(),
               description: "List active tasks from the user's Todoist app. Use when asked \
                             about tasks, todos, or what's on their schedule.".into(),
               parameters: json!({
                   "type": "object",
                   "properties": {
                       "filter": {
                           "type": "string",
                           "description": "Optional Todoist filter query, e.g. 'today', 'overdue', \
                                          'p1' for priority 1. Leave empty for all active tasks."
                       }
                   },
                   "required": []
               }),
           },
       },
       ToolDef {
           r#type: "function".into(),
           function: ToolFunction {
               name: "todoist_add".into(),
               description: "Add a new task to the user's Todoist app.".into(),
               parameters: json!({
                   "type": "object",
                   "properties": {
                       "content": {
                           "type": "string",
                           "description": "The task text"
                       },
                       "due": {
                           "type": "string",
                           "description": "Optional due date in natural language, e.g. 'tomorrow', 'next monday', 'jun 25'"
                       }
                   },
                   "required": ["content"]
               }),
           },
       },
       ToolDef {
           r#type: "function".into(),
           function: ToolFunction {
               name: "todoist_complete".into(),
               description: "Mark a Todoist task as complete by matching its text.".into(),
               parameters: json!({
                   "type": "object",
                   "properties": {
                       "task": {
                           "type": "string",
                           "description": "Text to match against task content, e.g. 'buy milk'"
                       }
                   },
                   "required": ["task"]
               }),
           },
       },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "index_system".into(),
                description: "Learn the user's system deeply. Scans home directory projects/scripts/configs/rust/python \
                              into permanent memory AND analyzes the full fish history to learn the top commands, \
                              most-visited directories, most-referenced file paths (editors/openers/installers), \
                              shell habits and packages. Use for 'learn about my system', 'know everything on \
                              my machine', 'study my workflow', and all general system orientation requests. \
                              The user can say 'learn about my system' to trigger this on demand.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["quick", "full"]
                        }
                    },
                    "required": ["scope"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "learn_topic".into(),
                description: "Search the web AND fetch the most relevant page in one step.                               Use this instead of calling web_search and fetch_page separately —                               it's more reliable. After it returns, call `remember` to save                               anything worth keeping permanently.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "What to learn about, e.g. 'Nothing Phone 3 specs'"
                        }
                    },
                    "required": ["topic"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "set_debug".into(),
                description: "Toggle Luna's debug logging on or off. Writes to the config file — \
                             Luna must be restarted for the change to take effect.  Accepts \
                             'on', 'off', 'debug', or 'info'.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["on", "off", "debug", "info"],
                            "description": "'on'/'debug' enables verbose logging; 'off'/'info' disables it"
                        }
                    },
                    "required": ["level"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "spotify".into(),
                description: "Control the user's Spotify account via the Spotify Web API: play their \
                              Liked Songs, playlists, artists, albums, search tracks, and transport \
                              control. The user's account is ALREADY authorized — do NOT run \
                              `luna --spotify-auth` or invent luna flags (there is no \
                              --spotify-username command). Use for 'play my liked songs', 'shuffle my \
                              liked songs', 'play my <playlist>', 'play <song name>', 'next track', \
                              'what's playing on spotify', and questions like 'what is my spotify \
                              username/account name' (action 'me'), 'how many playlists do i have' \
                              / 'list my playlists' (action 'playlists'). If a call returns an \
                              authorization error, tell the user to run `luna --spotify-auth`.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["now", "pause", "resume", "next", "previous", "shuffle",
                                     "play_liked", "search", "play_track", "play_playlist",
                                     "play_artist", "play_album", "devices", "me", "playlists"],
                            "description": "What to do: now=what's playing, pause/resume/next/previous=transport, \
                                           shuffle (set 'on' true/false), play_liked=play the user's Liked Songs \
                                           (set 'shuffle' true to shuffle), search=find tracks, \
                                           play_track/play_playlist/play_artist/play_album=play by name \
                                           (name goes in 'query' or 'playlist'), devices=list available devices, \
                                           me=the account's username/display name/email/plan, \
                                           playlists=list the user's playlists with track counts"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search text for search/play_track/play_artist/play_album (e.g. 'bohemian rhapsody')"
                        },
                        "playlist": {
                            "type": "string",
                            "description": "Playlist name for play_playlist (matched by keyword, e.g. 'gym')"
                        },
                        "on": {
                            "type": "boolean",
                            "description": "For shuffle: true = shuffle on, false = off"
                        },
                        "shuffle": {
                            "type": "boolean",
                            "description": "For play_liked: true to shuffle your Liked Songs"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },
        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "media_info".into(),
                description: "Get what's currently playing on the system (Spotify, MPV, \
                             browser, etc.) via D-Bus MPRIS. Returns song title, artist, \
                             album, and playback status. Use for 'what song is playing' \
                             or 'what am I listening to'. For Spotify-specific control \
                             use the 'spotify' tool instead.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "run_safety_check".into(),
                description: "Run or report on the system safety check. Sections: system update \
                              (pacman -Syu), ClamAV antivirus, rkhunter rootkit check, UFW firewall, \
                              Lynis audit, monthly AIDE integrity, and home-directory backup. \
                              Use for 'run a security check', 'is my system secure', 'run the weekly \
                              safety scan', 'how did the last safety check go'. It can take up to an \
                              hour (or more for the monthly full home scan). An 'attention needed' \
                              result means review the listed problems. Do NOT script it by hand with \
                              run_shell — use this tool.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["run", "status"],
                            "description": "run = execute the full check now (status then depends on what it found), \
                                           status = report how the last check went without starting a new one"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["light", "full"],
                            "description": "light (default) = quick scan of hot dirs + weekly checks; \
                                           full = full-home ClamAV scan (slow)"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "backup".into(),
                description: "Back up the user's home directory to the external drive /dev/sda1 at \
                              /mnt/backup/arch-backup/ (incremental rsync snapshots). Use for \
                              'back up my files', 'backup', 'do a backup'. Or get a status report \
                              with action=status: whether the drive is connected/mounted, last \
                              snapshot, and disk usage. The drive must be plugged in for the backup \
                              to run; if it isn't, say so and ask the user to connect it.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["run", "status"],
                            "description": "run = perform the backup now; status = report drive + last snapshot state"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "sysmode".into(),
                description: "Luna's interface to the 'sysmode' hardening-profile switcher \
                              (normally installed at /usr/local/bin/sysmode). Actions: \
                              action=status reports the current security profile and firewall/\
                              kernel state (works without root); action=check (or health) runs \
                              a full self-test of the whole setup — recon-deceiver honeypot, IDS \
                              log analyst, Cowrie SSH honeypot, tcpdump MAC scanner, decoy wifi, \
                              dnscrypt-proxy, auditd, sshd state, decoy listening ports, and \
                              log freshness — so you can say 'is the honeypot working?' or 'test \
                              if everything is working'; action=switch with 'mode' in \
                              {secure, cyber, stealth, lockdown} to change the profile — 'stealth' \
                              accepts optional 'hotspot' true/false to broadcast a decoy WiFi AP \
                              (requires [agent] sudo_password); action=reapply re-forces the \
                              current profile (root); action=logs pulls honeypot/IDS attack logs \
                              (works without root). If the tool is not installed, say so plainly \
                              and tell the user to install sysmode or set [sysmode] bin.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "check", "switch", "reapply", "logs"],
                            "description": "what to do with sysmode"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["secure", "cyber", "stealth", "lockdown"],
                            "description": "profile to switch to (only for action=switch)"
                        },
                        "hotspot": {
                            "type": "boolean",
                            "description": "stealth only: true to broadcast a decoy WiFi AP matching the spoofed hostname, false to skip"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "browser_do".into(),
                description: "Make Luna actually use a real web browser. Spins up the local \
                              Project-Vision browser-automation pipeline (a VLM that looks at the \
                              page and decides what to click/type where, executing inside a \
                              visible Chromium window with its own saved profile). Good for \
                              goals like 'open amazon and add the first PS4 controller to the \
                              cart', 'search for ryzen 7 on amazon.com', or 'fill this form \
                              https://...'. Give ONE clear natural-language task. It blocks \
                              until the task finishes or its timeout passes. Use this instead of \
                              just searching when the user explicitly wants something DONE in a \
                              browser. It cannot login for you or read captchas; if the site \
                              needs a login the browser window is visible so the user can \
                              complete it.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "the goal to accomplish in the browser, as a sentence (e.g. 'search for ps4 on amazon.com' or 'add the first result to cart')"
                        }
                    },
                    "required": ["task"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "whatsapp_send".into(),
                description: "Interact with the user's own WhatsApp via the local bridge (linked \
                              once via 'luna --whatsapp-link'). SEND-ONLY: it cannot read incoming \
                              messages, chats, or unread counts — say so plainly if asked. \
                              Actions: action=send with 'to' (full number digits-only like \
                              15551234567, OR 'myself'/'me' for the user's own number, OR a \
                              contact name like 'mom' — resolved from WhatsApp contacts) and \
                              'text' (the message body); action=lookup with 'to' as a name to \
                              report that contact's number WITHOUT sending; action=contacts to \
                              list the whole indexed contact book as names with their numbers \
                              (with an optional 'q' to filter by a name fragment, and 'show_all' \
                              to return the complete list instead of a preview) so messages \
                              can be addressed by name; action=status reports whether the bridge \
                              is up/linked and the user's own number. When a number appears in \
                              conversation, prefer the contact's NAME; only give the bare number \
                              if there's no contact entry. Never invent a recipient — if the \
                              user didn't provide one, ask. If the bridge is offline, tell the \
                              user to run 'luna --whatsapp-link' or check luna-whapp.service.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["send", "lookup", "contacts", "status"],
                            "description": "send = deliver a message (needs to + text); lookup = report a contact's number by name without sending; contacts = list the contact book (names ↔ numbers); status = check the bridge is up and linked"
                        },
                        "to": {
                            "type": "string",
                            "description": "full international phone number, digits only (e.g. 15551234567)"
                        },
                        "text": {
                            "type": "string",
                            "description": "the message body to deliver"
                        },
                        "q": {
                            "type": "string",
                            "description": "optional name filter for action=contacts (e.g. 'jane')"
                        },
                        "show_all": {
                            "type": "boolean",
                            "description": "if true, return the complete contact list; otherwise a preview (first 15 entries)"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },
    ]
}

/// Safe subset of tools exposed to the fast tier (qwen3:4b etc.).
///
/// The fast model must be able to RESOLVE "I don't know" cases itself —
/// especially web lookups — without being handed dangerous primitives like
/// `run_shell` or `write_file`. Anything here should be read-only or
/// non-destructive.
pub fn fast_tool_definitions() -> Vec<ToolDef> {
    let safe: &[&str] = &[
        "web_search",
        "fetch_page",
        "read_file",
        "find_file",
        "system_info",
        "process_stats",
        "clipboard",
        "media_info",
        "remember",
        "forget",
        "list_memories",
        "memory_report",
        "notify",
        "set_reminder",
        "list_reminders",
        "cancel_reminder",
        "dns_lookup",
    ];
    tool_definitions()
        .into_iter()
        .filter(|t| safe.contains(&t.function.name.as_str()))
        .collect()
}

pub async fn execute(tool_call: &ToolCall, config: &crate::config::LunaConfig) -> Result<String> {
    let name = &tool_call.function.name;
    let args = &tool_call.function.arguments;

    tracing::info!("Executing tool: {} with args: {}", name, args);

    let sudo_pass = config.agent.sudo_password.as_deref();

    match name.as_str() {
        "run_shell" => {
            let command = args["command"].as_str().unwrap_or("echo 'no command'");
            let result = shell::run_command(command, sudo_pass).await?;
            if result.exit_code == 0 {
                Ok(format!("SUCCESS\n{}", result.stdout.trim()))
            } else {
                Ok(format!(
                    "FAILED (exit {})\nstdout: {}\nstderr: {}",
                    result.exit_code,
                    result.stdout.trim(),
                    result.stderr.trim()
                ))
            }
        }

        "find_file" => {
            let name = args["name"].as_str().unwrap_or("*");
            let path = args["search_path"].as_str().unwrap_or("~");
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            let result = shell::run_command(
                &format!(
                    "find '{}' -name '{}' 2>/dev/null | head -10",
                    expanded, name
                ),
                sudo_pass,
            )
            .await?;
            if result.stdout.trim().is_empty() {
                Ok(format!("'{}' not found", name))
            } else {
                Ok(result.stdout.trim().to_string())
            }
        }

        "nmap_scan" => {
            let target = args["target"].as_str().unwrap_or("");
            let scan_type = args["scan_type"].as_str().unwrap_or("quick");
            if target.is_empty() {
                anyhow::bail!("No target provided");
            }
            security::nmap_scan(target, scan_type, sudo_pass).await
        }

        "analyze_pcap" => {
            let path = args["path"].as_str().unwrap_or("");
            let mode = args["mode"].as_str().unwrap_or("summary");
            security::analyze_pcap(path, mode, sudo_pass).await
        }

        "decode_payload" => {
            let data = args["data"].as_str().unwrap_or("");
            let encoding = args["encoding"].as_str().unwrap_or("auto");
            security::decode_payload(data, encoding, sudo_pass).await
        }

        "hash_file" => {
            let path = args["path"].as_str().unwrap_or("");
            let algo = args["algo"].as_str().unwrap_or("sha256");
            security::hash_file(path, algo, sudo_pass).await
        }

        "dns_lookup" => {
            let target = args["target"].as_str().unwrap_or("");
            let mode = args["mode"].as_str().unwrap_or("dns");
            security::dns_lookup(target, mode, sudo_pass).await
        }

        "edit_file" => {
            let path = args["path"].as_str().unwrap_or("");
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            shell::run_command(&format!("zeditor {} &", expanded), sudo_pass).await?;
            Ok("done".to_string())
        }

        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            web::search(
                query,
                config.search.tavily_api_key.as_deref(),
                config.search.gemini_api_key.as_deref(),
            )
            .await
        }

        "read_file" => {
            let path = args["path"].as_str().unwrap_or("/dev/null");
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            filesystem::read_file(&expanded).await
        }

        "write_file" => {
            let path = args["path"].as_str().unwrap_or("/dev/null");
            let content = args["content"].as_str().unwrap_or("");
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            filesystem::write_file(&expanded, content).await?;
            Ok(format!("Written to {}", expanded))
        }

        "notify" => {
            let title = args["title"].as_str().unwrap_or("Luna");
            let body = args["body"].as_str().unwrap_or("");
            desktop::notify(title, body, sudo_pass).await?;
            Ok("Notification sent".into())
        }

        "system_info" => {
            let query = args["query"].as_str().unwrap_or("all");
            let cmd = match query {
                "battery" => "cat /sys/class/power_supply/BAT0/capacity 2>/dev/null | xargs -I{} echo 'Battery: {}%'; cat /sys/class/power_supply/BAT0/status 2>/dev/null | xargs -I{} echo 'Status: {}'".to_string(),
                "cpu"     => "top -bn1 | grep 'Cpu(s)' | awk '{print \"CPU: \" $2+$4 \"%\"}'".to_string(),
                "ram"     => "free -h | awk '/^Mem:/ {print \"RAM: \" $3 \"/\" $2}'".to_string(),
                "temp"    => "sensors 2>/dev/null | grep -E 'Core|Tdie|temp' | head -5 || echo 'sensors not installed'".to_string(),
                "disk"    => "df -h / | awk 'NR>1 {print \"/: \" $3 \"/\" $2 \" (\" $5 \")\"}'".to_string(),
                "uptime"  => "uptime -p".to_string(),
                _         => "cat /sys/class/power_supply/BAT0/capacity 2>/dev/null | xargs -I{} echo 'Battery: {}%'; free -h | awk '/^Mem:/ {print \"RAM: \" $3 \"/\" $2}'; uptime -p; df -h / | awk 'NR>1 {print \"/: \" $3 \"/\" $2}'".to_string(),
            };
            let result = shell::run_command(&cmd, sudo_pass).await?;
            Ok(result.stdout.trim().to_string())
        }

        "clipboard" => {
            let action = args["action"].as_str().unwrap_or("read");
            match action {
                "write" => {
                    let content = args["content"].as_str().unwrap_or("");
                    let cmd = format!("printf '%s' '{}' | wl-copy", content.replace('\'', "'\\''"));
                    shell::run_command(&cmd, sudo_pass).await?;
                    Ok("Copied to clipboard".to_string())
                }
                _ => {
                    let result = shell::run_command(
                        "wl-paste 2>/dev/null || xclip -o 2>/dev/null || echo 'clipboard empty'",
                        sudo_pass,
                    )
                    .await?;
                    Ok(result.stdout.trim().to_string())
                }
            }
        }

        "fetch_page" => {
            let url = args["url"].as_str().unwrap_or("");
            if url.is_empty() {
                anyhow::bail!("No URL provided");
            }
            // Try Firecrawl keyless first (handles JS-rendered pages)
            match fetch_page_firecrawl(url).await {
                Ok(text) => Ok(text),
                Err(e) => {
                    tracing::warn!("Firecrawl failed for {}: {}", url, e);
                    // Fallback: curl + sed (no JS support)
                    let safe_url = url.replace('\'', "'\\''");
                    let cmd = format!(
                        "curl -sL --max-time 10 \
                            --user-agent 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0' \
                            '{}' \
                         | sed 's/<[^>]*>//g' | sed '/^[[:space:]]*$/d' | head -200",
                        safe_url
                    );
                    let result = shell::run_command(&cmd, sudo_pass).await?;
                    if result.stdout.trim().is_empty() {
                        Ok("Could not fetch page".to_string())
                    } else {
                        let text = result.stdout.trim();
                        let truncated = text
                            .char_indices()
                            .take_while(|(i, _)| *i < 4000)
                            .last()
                            .map(|(i, c)| &text[..i + c.len_utf8()])
                            .unwrap_or(text);
                        Ok(truncated.to_string())
                    }
                }
            }
        }

        "remember" => {
            let fact = args["fact"].as_str().unwrap_or("").to_string();
            let category = args["category"].as_str().unwrap_or("general").to_string();
            let mut pm = crate::memory::permanent::PermanentMemory::load()?;
            pm.remember(&fact, &category)
        }

        "forget" => {
            let keyword = args["keyword"].as_str().unwrap_or("");
            let mut pm = crate::memory::permanent::PermanentMemory::load()?;
            pm.forget(keyword)
        }

        "todoist_list" => {
            let token = config.todoist.api_token.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Todoist not configured — add api_token to luna.toml under [todoist]"
                )
            })?;
            let filter = args["filter"].as_str().filter(|s| !s.is_empty());
            crate::tools::todoist::list_tasks(token, filter).await
        }

        "todoist_add" => {
            let token = config
                .todoist
                .api_token
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Todoist not configured"))?;
            let content = args["content"].as_str().unwrap_or("");
            let due = args["due"].as_str().filter(|s| !s.is_empty());
            crate::tools::todoist::add_task(token, content, due).await
        }

        "todoist_complete" => {
            let token = config
                .todoist
                .api_token
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Todoist not configured"))?;
            let task = args["task"].as_str().unwrap_or("");
            crate::tools::todoist::complete_task(token, task).await
        }

        "spotify" => {
            let action = args["action"].as_str().unwrap_or("");
            crate::tools::spotify::run(action, args, config).await
        }

        "list_memories" => {
            let pm = crate::memory::permanent::PermanentMemory::load()?;
            Ok(pm.list())
        }

        "create_skill" => {
            let name = args["name"].as_str().unwrap_or("");
            let description = args["description"].as_str().unwrap_or("");
            let procedure = args["procedure"].as_str().unwrap_or("");
            crate::memory::skills::create(name, description, procedure)
        }

        "use_skill" => {
            let name = args["name"].as_str().unwrap_or("");
            crate::memory::skills::read(name)
        }

        "list_skills" => Ok(crate::memory::skills::list_and_format()),

        "forget_skill" => {
            let name = args["name"].as_str().unwrap_or("");
            crate::memory::skills::forget(name)
        }

        "search_history" => {
            let query = args["query"].as_str().unwrap_or("");
            Ok(crate::agent::learning::search_history(query))
        }

        "index_system" => {
            let summary = crate::memory::workflow::run_index_system(sudo_pass).await?;
            if summary.is_empty() {
                Ok("Nothing found to index".to_string())
            } else {
                Ok(format!("Indexed: {}", summary.join(", ")))
            }
        }

        "learn_topic" => {
            let topic = args["topic"].as_str().unwrap_or("");
            if topic.is_empty() {
                anyhow::bail!("No topic provided");
            }
            learn::learn(
                topic,
                sudo_pass,
                config.search.tavily_api_key.as_deref(),
                config.search.gemini_api_key.as_deref(),
            )
            .await
        }

        "process_stats" => {
            let tracker = crate::daemon::tracker::Tracker::load();
            let stats = tracker.stats_snapshot();
            if stats.is_empty() {
                return Ok(
                    "No process learning data yet — the daemon hasn't completed a scan cycle."
                        .into(),
                );
            }
            let allowlist: std::collections::HashSet<String> =
                crate::daemon::tracker::load_allowlist()
                    .into_iter()
                    .collect();
            let protected = config.daemon.protected_processes.clone();

            let mut rows: Vec<String> = stats
                .iter()
                .map(|(name, s)| {
                    let days_14 = {
                        let cutoff = (chrono::Local::now() - chrono::Duration::days(14))
                            .format("%Y-%m-%d")
                            .to_string();
                        s.days_seen
                            .iter()
                            .filter(|d| d.as_str() > cutoff.as_str())
                            .count()
                    };
                    let status = if protected.iter().any(|p| p == name) {
                        "protected"
                    } else if allowlist.contains(name) {
                        "auto-kill allowed"
                    } else {
                        "-"
                    };
                    format!(
                        "{:<24} {}/14d  idle {} cyc  {:>6} jiffies  {}",
                        name, days_14, s.idle_cycles, s.total_jiffies, status
                    )
                })
                .collect();
            rows.sort();

            Ok(format!(
                "Process usage ({} tracked, daemon scans every {} min):\n{}",
                stats.len(),
                config.daemon.check_interval_mins,
                rows.join("\n")
            ))
        }

        "allow_autokill" => {
            let name = args["name"].as_str().unwrap_or("");
            if name.is_empty() {
                anyhow::bail!("No process name provided");
            }
            crate::daemon::tracker::allowlist_add(name)?;
            tracing::info!("Auto-kill allowed for '{}'", name);
            Ok(format!(
                "Allowed. The daemon will end '{}' after ~{} min idle (unless it's a \
                 daily-use or protected app). Say 'deny auto-kill {}' to revoke.",
                name, config.daemon.idle_kill_minutes, name
            ))
        }

        "deny_autokill" => {
            let name = args["name"].as_str().unwrap_or("");
            if name.is_empty() {
                anyhow::bail!("No process name provided");
            }
            match crate::daemon::tracker::allowlist_remove(name)? {
                true => Ok(format!(
                    "Revoked — '{}' will never be auto-killed again.",
                    name
                )),
                false => Ok(format!(
                    "'{}' wasn't on the auto-kill list — it was already safe.",
                    name
                )),
            }
        }

        "set_reminder" => {
            let text = args["text"].as_str().unwrap_or("");
            let minutes = args["minutes"].as_u64();
            let at_time = args["at_time"].as_str();
            let r = if let Some(mins) = minutes {
                crate::tools::reminders::add_in(mins.min(u32::MAX as u64) as u32, text)?
            } else if let Some(t) = at_time {
                crate::tools::reminders::add_at(t, text)?
            } else {
                anyhow::bail!("Need either 'minutes' or 'at_time'");
            };
            let when = local_fmt(r.fire_at, "%a %H:%M");
            Ok(format!(
                "Reminder set (id {}): '{}' fires {}",
                r.id, r.text, when
            ))
        }

        "list_reminders" => {
            let all = crate::tools::reminders::list();
            if all.is_empty() {
                return Ok("No pending reminders.".into());
            }
            let rows: Vec<String> = all
                .iter()
                .map(|r| {
                    format!(
                        "#{}  {}  {}",
                        r.id,
                        local_fmt(r.fire_at, "%a %d %b %H:%M"),
                        r.text
                    )
                })
                .collect();
            Ok(rows.join("\n"))
        }

        "cancel_reminder" => {
            let id = args["id"].as_u64().unwrap_or(0);
            match crate::tools::reminders::cancel(id)? {
                true => Ok(format!("Cancelled reminder #{}.", id)),
                false => Ok(format!("No reminder with id {}.", id)),
            }
        }

        "memory_report" => {
            use std::collections::BTreeMap;
            let pm = crate::memory::permanent::PermanentMemory::load()?;
            let facts = pm.all_facts();

            let mut by_cat: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
            for f in facts {
                by_cat
                    .entry(f.category.as_str())
                    .or_default()
                    .push(&f.content);
            }

            let mut out = String::from("What Luna has permanently learned:\n");
            for (cat, items) in &by_cat {
                out.push_str(&format!("\n[{}] {}\n", cat, items.len()));
                for c in items.iter().take(15) {
                    out.push_str(&format!("  - {}\n", c));
                }
                if items.len() > 15 {
                    out.push_str(&format!("  ... and {} more\n", items.len() - 15));
                }
            }
            out.push_str(
                "\n(These facts ARE the answer to the user's question — present them \
                 clearly now. Do not ask what they want to do next.)\n",
            );
            Ok(out)
        }

        "media_info" => {
            // Use gdbus (GIO) — dbus-send is broken on some Arch builds.
            // Query Spotify first, then fall back to any active MPRIS player.
            let script = r#"
# Try Spotify first
OUT=$(gdbus call --session --dest org.mpris.MediaPlayer2.spotify \
  --object-path /org/mpris/MediaPlayer2 \
  --method org.freedesktop.DBus.Properties.Get \
  org.mpris.MediaPlayer2.Player Metadata 2>/dev/null)

# If that failed, find any MediaPlayer2 bus
if [ -z "$OUT" ]; then
    for bus in $(gdbus call --session --dest org.freedesktop.DBus \
      --object-path /org/freedesktop/DBus \
      --method org.freedesktop.DBus.ListNames 2>/dev/null | \
      tr "'" "\n" | grep MediaPlayer2); do
        OUT=$(gdbus call --session --dest "$bus" \
          --object-path /org/mpris/MediaPlayer2 \
          --method org.freedesktop.DBus.Properties.Get \
          org.mpris.MediaPlayer2.Player Metadata 2>/dev/null)
        [ -n "$OUT" ] && break
    done
fi

[ -z "$OUT" ] && echo "NO_PLAYER" && exit 0

# Parse GVariant output — extract title, artist, album
TITLE=$(echo "$OUT" | grep -oP "xesam:title': <'\K[^']+")
ARTIST=$(echo "$OUT" | grep -oP "xesam:artist': <\['\K[^']+")
ALBUM=$(echo "$OUT" | grep -oP "xesam:album': <'\K[^']+")
STATUS=$(gdbus call --session --dest org.mpris.MediaPlayer2.spotify \
  --object-path /org/mpris/MediaPlayer2 \
  --method org.freedesktop.DBus.Properties.Get \
  org.mpris.MediaPlayer2.Player PlaybackStatus 2>/dev/null | \
  grep -oP "'\\K[^']+" || echo "Unknown")

echo "TITLE=$TITLE"
echo "ARTIST=$ARTIST"
echo "ALBUM=$ALBUM"
echo "STATUS=$STATUS"
"#;
            let result = shell::run_command(script.trim(), sudo_pass).await?;
            let stdout = result.stdout.trim().to_string();
            if stdout.contains("NO_PLAYER") || stdout.is_empty() {
                Ok("No media player found running. Start Spotify or another \
                    MPRIS-compatible player first."
                    .into())
            } else {
                // Parse KEY=VALUE lines into a readable sentence
                let mut title = String::new();
                let mut artist = String::new();
                let mut album = String::new();
                let mut status = String::new();
                for line in stdout.lines() {
                    if let Some(v) = line.strip_prefix("TITLE=") {
                        title = v.to_string();
                    } else if let Some(v) = line.strip_prefix("ARTIST=") {
                        artist = v.to_string();
                    } else if let Some(v) = line.strip_prefix("ALBUM=") {
                        album = v.to_string();
                    } else if let Some(v) = line.strip_prefix("STATUS=") {
                        status = v.to_string();
                    }
                }
                let mut out = format!("Now playing: {} by {}", title, artist);
                if !album.is_empty() {
                    out.push_str(&format!(" (album: {})", album));
                }
                out.push_str(&format!(" [{}]", status));
                Ok(out)
            }
        }

        "set_debug" => {
            let level = args["level"].as_str().unwrap_or("info");
            let new_level = match level {
                "on" | "debug" => "debug",
                "off" | "info" => "info",
                other => {
                    anyhow::bail!(
                        "Invalid level '{}' — use 'on', 'off', 'debug', or 'info'",
                        other
                    );
                }
            };
            let mut cfg = crate::config::LunaConfig::load()?;
            cfg.logging.level = new_level.to_string();
            cfg.save()?;
            Ok(format!(
                "Logging set to '{}'. Restart Luna for the change to take effect.",
                new_level
            ))
        }

        "run_safety_check" => {
            let action = args["action"].as_str().unwrap_or("status");
            match action {
                "run" => {
                    let mode = args["mode"].as_str().unwrap_or("light");
                    if crate::tools::safety::is_running() {
                        Ok(
                            "A safety check is already running — I'll wait for it and report the \
                            result. Ask me again in a while."
                                .into(),
                        )
                    } else {
                        match crate::tools::safety::run(mode, sudo_pass, false).await {
                            Ok(summary) => Ok(format!(
                                "Safety check started and finished. {}\n(Full log under \
                                 ~/logs/safety_check/. Ask if you want a part of it read.)",
                                summary
                            )),
                            Err(e) => Err(e),
                        }
                    }
                }
                _ => Ok(crate::tools::safety::status()),
            }
        }

        "backup" => {
            let action = args["action"].as_str().unwrap_or("status");
            match action {
                "run" => {
                    if !std::path::Path::new("/dev/sda1").exists() {
                        Ok(
                            "Backup drive /dev/sda1 isn't connected. Plug the external drive in \
                            and say 'back up my files' again."
                                .into(),
                        )
                    } else {
                        Ok(crate::tools::backup::run(sudo_pass).await?)
                    }
                }
                _ => Ok(crate::tools::backup::status().await?),
            }
        }

        "sysmode" => {
            let scfg = &config.sysmode;
            if !scfg.enabled {
                Ok(
                    "sysmode support is disabled in config ([sysmode] enabled = false). \
                    Tell the user to re-enable it in luna.toml if they want it."
                        .into(),
                )
            } else {
                let action = args["action"].as_str().unwrap_or("status");
                match action {
                    "switch" => {
                        let mode = args["mode"].as_str().unwrap_or("");
                        if mode.is_empty() {
                            Ok("The profile to switch to was missing. Ask the user which \
                                one: secure, cyber, stealth, or lockdown."
                                .into())
                        } else {
                            let hotspot = args.get("hotspot").and_then(|v| v.as_bool());
                            match crate::tools::sysmode::switch_mode(scfg, mode, hotspot, sudo_pass)
                                .await
                            {
                                Ok(out) => Ok(out),
                                Err(e) => {
                                    Ok(format!("Couldn't switch to the {mode} profile: {e:#}"))
                                }
                            }
                        }
                    }
                    "status" => match crate::tools::sysmode::status(scfg).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't read sysmode status: {e:#}")),
                    },
                    "check" | "health" => match crate::tools::sysmode::health_check(scfg).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't run the sysmode health check: {e:#}")),
                    },
                    "logs" => match crate::tools::sysmode::logs(scfg).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't read sysmode logs: {e:#}")),
                    },
                    "reapply" => match crate::tools::sysmode::reapply(scfg, sudo_pass).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't re-apply the sysmode profile: {e:#}")),
                    },
                    _ => Ok(format!(
                        "Unknown sysmode action '{action}'. Valid actions: status, check, \
                         switch, reapply, logs."
                    )),
                }
            }
        }

        "whatsapp_send" => {
            if !config.whatsapp.enabled {
                Ok("WhatsApp is disabled in config ([whatsapp] enabled = false). Tell the user to \
                    re-enable it in luna.toml if they want to use it."
                    .into())
            } else {
                let base = Some(config.whatsapp.base_url.as_str());
                let action = args["action"].as_str().unwrap_or("status");
                match action {
                    "send" => {
                        let to = args["to"].as_str().unwrap_or("");
                        let text = args["text"].as_str().unwrap_or("").trim();
                        if to.is_empty() {
                            Ok(
                                "The recipient was missing. Ask the user for it (full number, \
                                digits only, e.g. 15551234567, or a contact name like 'mom') \
                                and don't guess."
                                    .into(),
                            )
                        } else if text.is_empty() {
                            Ok("The message text was empty. Ask the user what to say.".into())
                        } else {
                            crate::tools::whatsapp::send(to, text, base).await
                        }
                    }
                    "lookup" => {
                        let to = args["to"].as_str().unwrap_or("");
                        if to.is_empty() {
                            Ok("The name to look up was missing. Ask the user for it.".into())
                        } else {
                            crate::tools::whatsapp::lookup(to, base).await
                        }
                    }
                    "contacts" => {
                        let q = args["q"].as_str();
                        let show_all = args["show_all"].as_bool().unwrap_or(false);
                        crate::tools::whatsapp::contacts(q, show_all, base).await
                    }
                    _ => crate::tools::whatsapp::status(base).await,
                }
            }
        }

        "browser_do" => {
            if !config.browser.enabled {
                Ok("Browser automation is disabled in config ([browser] enabled = false). Tell \
                    the user to re-enable it in luna.toml."
                    .into())
            } else {
                let task = args["task"].as_str().unwrap_or("").trim();
                if task.is_empty() {
                    Ok("The browser task was empty. Ask the user what they want done in the \
                        browser."
                        .into())
                } else {
                    match crate::browser::run(task, config).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Browser task failed: {e:#}")),
                    }
                }
            }
        }

        unknown => {
            tracing::warn!("Unknown tool: {}", unknown);
            Ok(format!("Error: unknown tool '{}'", unknown))
        }
    }
}

/// Pull the URLs a research tool surfaced out of its plain-text result,
/// so the assistant can show the user the exact sources it referenced.
/// Format a unix timestamp for reminder display in local time.
/// so the assistant can show the user the exact sources it referenced.
/// Format a unix timestamp for reminder display in local time.
fn local_fmt(epoch: u64, fmt: &str) -> String {
    use chrono::TimeZone;
    chrono::Local
        .timestamp_opt(epoch as i64, 0)
        .single()
        .map(|d| d.format(fmt).to_string())
        .unwrap_or_default()
}

pub fn extract_sources(tool_name: &str, result: &str) -> Vec<String> {
    let mut sources: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        let t = s.trim();
        if !t.is_empty() && !sources.iter().any(|x| x == t) {
            sources.push(t.to_string());
        }
    };

    match tool_name {
        "web_search" | "learn_topic" => {
            for line in result.lines() {
                let line = line.trim();
                if let Some(url) = line.strip_prefix("Source:").map(str::trim) {
                    push(url);
                } else if let Some(rest) = line.strip_prefix("=== Fetched page:") {
                    let url = rest.trim_end_matches("===").trim();
                    if !url.is_empty() {
                        push(url);
                    }
                } else if let Some(rest) = line.strip_prefix("- ") {
                    if rest.contains("http") {
                        if let Some(open) = rest.rfind('(') {
                            if let Some(close) = rest.rfind(')') {
                                if close > open {
                                    push(&rest[open + 1..close]);
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    sources
}

/// Fetch a page via Firecrawl keyless — returns clean markdown
async fn fetch_page_firecrawl(url: &str) -> Result<String> {
    let body = serde_json::json!({ "url": url });
    let body_str = body.to_string();

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time",
                "30",
                "-H",
                "Content-Type: application/json",
                "-d",
                &body_str,
                "https://api.firecrawl.dev/v1/scrape",
            ])
            .output()
            .context("curl not found")
    })
    .await
    .context("spawn_blocking panicked")??;

    let resp: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&output.stdout))
        .context("Failed to parse Firecrawl response")?;

    if let Some(err) = resp.get("error") {
        anyhow::bail!("Firecrawl: {}", err);
    }

    let markdown = resp["data"]["markdown"]
        .as_str()
        .context("No markdown in Firecrawl response")?;

    let truncated: String = markdown.chars().take(4000).collect();
    Ok(truncated)
}
