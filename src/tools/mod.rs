//! tools/mod.rs — Tool registry
//!
//! One tool per capability, sized for a 7B model to use reliably.
//! Redundant tools (list_dir, find_binary, open) removed;
//! run_shell handles all of those.
pub mod desktop;
pub mod filesystem;
pub mod learn;
pub mod proactive;
pub mod security;
pub mod shell;
pub mod todoist;
pub mod web;

use crate::llm::ollama::{ToolCall, ToolDef, ToolFunction};
use anyhow::Result;
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
                description: "Scan home directory and save a structured index to permanent memory. \
                              Run once to learn the system layout.".into(),
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
                name: "media_info".into(),
                description: "Get what's currently playing on the system (Spotify, MPV, \
                             browser, etc.) via D-Bus MPRIS. Returns song title, artist, \
                             album, and playback status. Use for 'what song is playing' \
                             or 'what am I listening to'.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
    ]
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
            web::search(query).await
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
            let cmd = format!(
                "curl -sL --max-time 10 '{}' | sed 's/<[^>]*>//g' | sed '/^[[:space:]]*$/d' | head -200",
                url.replace('\'', "'\\''")
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

        "list_memories" => {
            let pm = crate::memory::permanent::PermanentMemory::load()?;
            Ok(pm.list())
        }

        "index_system" => {
            let home = std::env::var("HOME").unwrap_or("/home/netrunner".to_string());
            let commands = vec![
                ("projects", format!("find {} -name '.git' -maxdepth 4 -type d 2>/dev/null | grep -v '.cache' | sed 's/\\/.git//' | head -20", home)),
                ("scripts",  format!("find {} -name '*.sh' -maxdepth 4 2>/dev/null | grep -v '.cache' | head -20", home)),
                ("configs",  "ls ~/.config/ 2>/dev/null | head -30".to_string()),
                ("rust",     format!("find {} -name 'Cargo.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | sed 's/\\/Cargo.toml//' | head -10", home)),
                ("python",   format!("find {} -name 'pyproject.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | head -10", home)),
            ];

            let mut pm = crate::memory::permanent::PermanentMemory::load()?;
            let mut summary = Vec::new();

            for (key, cmd) in &commands {
                let result = shell::run_command(cmd, sudo_pass).await?;
                let items = result.stdout.trim();
                if !items.is_empty() {
                    let fact = format!(
                        "System index - {}: {}",
                        key,
                        items.lines().collect::<Vec<_>>().join(", ")
                    );
                    pm.remember(&fact, "system").ok();
                    summary.push(format!("**{}**: {} items", key, items.lines().count()));
                }
            }

            Ok(format!("Indexed: {}", summary.join(", ")))
        }

        "learn_topic" => {
            let topic = args["topic"].as_str().unwrap_or("");
            if topic.is_empty() {
                anyhow::bail!("No topic provided");
            }
            learn::learn(topic, sudo_pass).await
        }

        "media_info" => {
            // Query all active MPRIS players for current playback info
            let script = r#"
dbus-send --session --dest=org.mpris.MediaPlayer2.spotify \
  --object-path=/org/mpris/MediaPlayer2 \
  --type=method_call --print-reply \
  org.freedesktop.DBus.Properties.Get \
  string:org.mpris.MediaPlayer2.Player \
  string:Metadata 2>/dev/null | \
  grep -E '"(xesam:title|xesam:artist|xesam:album|xesam:url)"' | \
  sed 's/.*variant.*string "\(.*\)"/\1/' | head -4
"#;
            let result = shell::run_command(script.trim(), sudo_pass).await?;
            let stdout = result.stdout.trim().to_string();
            if stdout.is_empty() {
                // Try generic MPRIS query for any active player
                let fallback = r#"
for bus in $(dbus-send --session --dest=org.freedesktop.DBus \
  --type=method_call --print-reply /org/freedesktop/DBus \
  org.freedesktop.DBus.ListNames 2>/dev/null | \
  grep "string \"" | sed 's/.*"\(.*\)".*/\1/' | grep MediaPlayer2); do
    dbus-send --session --dest="$bus" \
      --object-path=/org/mpris/MediaPlayer2 \
      --type=method_call --print-reply \
      org.freedesktop.DBus.Properties.Get \
      string:org.mpris.MediaPlayer2.Player \
      string:Metadata 2>/dev/null | \
      grep -E '"(xesam:title|xesam:artist|xesam:album)"' | \
      sed 's/.*variant.*string "\(.*\)"/\1/' | head -3
    break
done
"#;
                let r2 = shell::run_command(fallback.trim(), sudo_pass).await?;
                let out2 = r2.stdout.trim().to_string();
                if out2.is_empty() {
                    Ok("No media player found running. Start Spotify or another \
                        MPRIS-compatible player first."
                        .into())
                } else {
                    Ok(format!("Now playing:\n{}", out2))
                }
            } else {
                Ok(format!("Now playing:\n{}", stdout))
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

        unknown => {
            tracing::warn!("Unknown tool: {}", unknown);
            Ok(format!("Error: unknown tool '{}'", unknown))
        }
    }
}

/// Pull the URLs a research tool surfaced out of its plain-text result,
/// so the assistant can show the user the exact sources it referenced.
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
