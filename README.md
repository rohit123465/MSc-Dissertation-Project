# TIAPathology MCP Server — Setup

A Python MCP server that exposes computational pathology tools (nucleus segmentation, tissue classification, spatial analysis) to Claude over JSON-RPC.

## Prerequisites

- Claude Desktop installed
- Python virtual environment created for this project, with dependencies installed
- Whole-slide image (`.svs`) files available locally

## Step 1 — Open the MCP settings in Claude Desktop

In Claude Desktop, go to **Settings → Developer**. This is where you add and manage the MCP servers you're working on.

Click **Edit Config**. This opens the configuration file `claude_desktop_config.json`.

Typical locations:

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |

## Step 2 — Add the server entry

Paste the configuration below into that file. If the file already contains other servers, add the `tiapathology` block inside the existing `mcpServers` object rather than replacing it.

```json
{
  "mcpServers": {
    "tiapathology": {
      "command": "C:\\path\\to\\health_mcpserver\\.venv\\Scripts\\python.exe",
      "args": [
        "-u",
        "C:\\path\\to\\tiatoolbox_mcpserver.py"
      ]
    }
  }
}
```

Notes:

- `command` must point at the Python executable **inside the project virtual environment**, not a system-wide Python. On macOS or Linux this is `.venv/bin/python`.
- `args` points at the server script. `-u` forces unbuffered output so logs stream through immediately.
- On Windows, backslashes in JSON must be escaped as `\\`. Forward slashes also work.
- Both paths must be absolute.

## Step 3 — Restart Claude Desktop

Quit the app completely and reopen it. Config changes are only read on startup.

## Step 4 — Verify

Open a new chat and check that the `tiapathology` tools are listed as available. Asking Claude to run the server's health check is the quickest confirmation that the process started correctly.

## Troubleshooting

- **No tools appear** — the server failed to launch. Check the paths in `command` and `args`, and confirm the JSON is valid, since a trailing comma or unescaped backslash will make Claude Desktop skip the file silently.
- **Server starts then exits** — run the script directly in a terminal using the same interpreter to see the traceback.
- **Wrong dependencies** — confirm the interpreter in `command` is the venv one by running `where python` (Windows) or `which python` (macOS/Linux) after activating the environment.

## Usage

Every request begins with a plan-and-approval step: the server proposes a pathology plan and waits for approval before executing any model or analysis tool.
