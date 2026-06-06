//! tools/mod.rs — Tool registry
//!
//! Trimmed to 11 tools — enough for a 7B model to use reliably.
//! Redundant tools (find_file, list_dir, find_binary, open) removed;
//! run_shell handles all of those.

pub mod desktop;
pub mod filesystem;
pub mod shell;
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

        unknown => {
            tracing::warn!("Unknown tool: {}", unknown);
            Ok(format!("Error: unknown tool '{}'", unknown))
        }
    }
}
