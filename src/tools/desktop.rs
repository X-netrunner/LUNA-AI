use crate::tools::shell::run_command;
use anyhow::Result;

pub async fn notify(title: &str, body: &str, sudo_pass: Option<&str>) -> Result<()> {
    run_command(
        &format!(
            "notify-send '{}' '{}'",
            title.replace('\'', "'\\''"),
            body.replace('\'', "'\\''")
        ),
        sudo_pass,
    )
    .await?;
    Ok(())
}

pub async fn open(target: &str, sudo_pass: Option<&str>) -> Result<()> {
    run_command(
        &format!("xdg-open '{}' &", target.replace('\'', "'\\''")),
        sudo_pass,
    )
    .await?;
    Ok(())
}
