# Claude Code History Backup Setup

**Date**: 2024-12-13
**Issue**: Lost conversation history from September/October due to automatic deletion
**Status**: Resolved with automated backup system

---

## Problem

Claude Code automatically deletes conversation history after **30 days** by default. This resulted in the loss of valuable conversations from September and October 2024 related to this project.

### Key Issues
- No warning or notification before deletion
- 30-day retention period is too short for long-term projects
- Silent deletion without user consent
- Historical context and decisions lost

### What Was Lost
- Conversations from ~September 13, 2024
- October development discussions
- All context and decisions made during those sessions

---

## Root Cause

Claude Code has a `cleanupPeriodDays` setting that defaults to **30 days**. This is documented but not prominently advertised:

- **Default behavior**: Automatically delete conversations older than 30 days
- **Storage location**: `~/.claude/projects/[project-hash]/`
- **Configuration**: Can be modified in `~/.claude/settings.json`

**Related GitHub Issue**: [anthropics/claude-code#4172](https://github.com/anthropics/claude-code/issues/4172)

---

## Solution

### 1. Disable Auto-Deletion

Updated `~/.claude/settings.json` to preserve conversations indefinitely:

```json
{
  "cleanupPeriodDays": 99999
}
```

This prevents future automatic deletions.

### 2. Automated Backup System

Created a backup script to preserve Claude Code history daily.

**Script Location**: `~/.local/bin/backup-claude-history.sh`

**Features**:
- Runs daily at 2 AM via cron
- Creates compressed `.tar.gz` backups (~163MB per backup)
- Keeps last 30 days of backups
- Automatic cleanup of old backups
- Maintains backup log

**Backup Location**: `/data/backups/claude-history/`

**Cron Job**:
```bash
0 2 * * * /home/gota/.local/bin/backup-claude-history.sh
```

---

## Backup Script

```bash
#!/bin/bash
# Automated backup script for Claude Code history
# Runs daily to preserve conversation history

# Configuration
BACKUP_DIR="/data/backups/claude-history"
SOURCE_DIR="$HOME/.claude"
MAX_BACKUPS=30  # Keep last 30 days of backups
DATE=$(date +%Y%m%d)
BACKUP_NAME="claude_${DATE}.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create compressed backup
tar -czf "$BACKUP_DIR/$BACKUP_NAME" -C "$HOME" .claude 2>/dev/null

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup successful: $BACKUP_NAME" >> "$BACKUP_DIR/backup.log"

    # Clean up old backups (keep only last MAX_BACKUPS)
    cd "$BACKUP_DIR"
    ls -t claude_*.tar.gz 2>/dev/null | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm --

    # Log cleanup
    REMAINING=$(ls -1 claude_*.tar.gz 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup complete. $REMAINING backups remaining." >> "$BACKUP_DIR/backup.log"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup failed!" >> "$BACKUP_DIR/backup.log"
    exit 1
fi
```

---

## Usage

### Manual Backup

Run the backup script manually if needed:

```bash
/home/gota/.local/bin/backup-claude-history.sh
```

### Check Backup Status

View recent backup logs:

```bash
tail -f /data/backups/claude-history/backup.log
```

List available backups:

```bash
ls -lh /data/backups/claude-history/
```

### Restore from Backup

If you need to restore from a specific backup:

```bash
# Backup current state first
mv ~/.claude ~/.claude.backup.$(date +%Y%m%d_%H%M%S)

# Restore from specific date
tar -xzf /data/backups/claude-history/claude_YYYYMMDD.tar.gz -C ~/
```

---

## What's Backed Up

The backup includes the entire `~/.claude/` directory:

- **Conversation history**: All project conversations
- **Settings**: Configuration files
- **Projects**: Project-specific data
- **Plans**: Saved implementation plans
- **Timelines**: Session timelines and checkpoints
- **Credentials**: Authentication tokens (encrypted)

Total size: ~434MB uncompressed, ~163MB compressed

---

## Recommendations for Others

If you're using Claude Code:

1. **Check your settings immediately**:
   ```bash
   cat ~/.claude/settings.json
   ```

2. **Add the `cleanupPeriodDays` setting** to prevent data loss:
   ```json
   {
     "cleanupPeriodDays": 99999
   }
   ```

3. **Set up automated backups** using the script above

4. **Create a manual backup now** before you lose anything:
   ```bash
   tar -czf ~/claude-backup-$(date +%Y%m%d).tar.gz ~/.claude/
   ```

---

## References

- [GitHub Issue #4172: Disable auto-deletion of past conversations](https://github.com/anthropics/claude-code/issues/4172)
- [Claude Code Data Usage Documentation](https://code.claude.com/docs/en/data-usage)
- [Anthropic Privacy Center: Data Retention](https://privacy.claude.com/en/articles/10023548-how-long-do-you-store-my-data)

---

## Lessons Learned

- Always check data retention policies for development tools
- Implement your own backup strategy for critical data
- Don't rely on defaults that may not match your needs
- 30 days is far too short for software development projects
- Silent data deletion without warning is poor UX

---

**Next Steps**:
- Monitor backup logs weekly
- Consider additional offsite backups for critical projects
- Potentially export important conversations to markdown for long-term archival
