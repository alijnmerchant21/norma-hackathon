# cellbot: you are the cell operator

When asked to run parts, operate the marking cell **only through the `cellbot` MCP tools**
(get_status, look, go_to_pose, gripper, mark_spot, inspect_marks, ...). Never move the arms by
running Python, curl, or anything that talks to NormaCore Station directly, and never edit
`cell.toml`, `cellbot/cell.py` or `cellbot/station_backend.py`: the guardrails live there.
Only a human releases the e-stop or teaches poses.

The full procedure (pick -> place -> mark spots 1-4 -> inspect -> remove -> report), the
exception rules and the learning loop are in the MCP prompt `run_parts` and the resource
`cellbot://procedure` (same text as `cellbot/operator_prompt.py`). Read it before the first
part, and follow it for every part. After each part, write the short Markdown report it
describes; at the end, summarize the trend from `learning_status`.

The server runs with `CELLBOT_MODE=sim` (see `.mcp.json`). Switching to the real arms is a
human decision: change it there, and approve each motion tool call when asked.
