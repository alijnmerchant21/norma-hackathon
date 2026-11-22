## Teleoperated handover (human demonstration) — narration

Use when the unload (or load) policy failed (primary, then fallback if one is configured), or at once when the
human asks for it. The cell moves NOTHING during the handover; a human moves the SO-101 with its leader arm
through NormaCore Station's own teleoperation. Never call any so101 motion while the handover is open (the tools
refuse it anyway: "a teleop handover is open: finish it with finish_teleop first").

Order of tool calls: `unload_part(variant="teleop")` -> relay `instructions_for_human` -> WAIT for the human to say
"done" and that teleoperation is OFF -> `finish_teleop(attempt_id)` -> `go_home so101` -> `see_part` ->
`label_loader_attempt(success, note, attempt_id)`.

Print these lines, one per event, in this style:

```
LOADER | unload | primary policy FAILED: the part is still in the work area
LOADER | unload | fallback FAILED: the part is still in the work area            (only if a fallback is configured)
DECISION | hand over to the human for a teleoperated unload | the policy cannot do this move yet; the demonstration becomes training data
HANDOVER | HOLDER ARM (SO-101) | teleoperation by the human: work area -> drop-off | started <iso_start, HH:MM:SS>
HUMAN | <instructions_for_human, word for word>
HANDOVER | waiting for the human: say "done" when the part is at the drop-off and teleoperation is OFF
HANDOVER | finished <iso_end, HH:MM:SS> (<seconds> s) | teleoperation OFF confirmed by the human
LOADER | so101 | go_home after the handover
RECORDING | attempt <id> | <iso_start> -> <iso_end> | teleop demonstration | labelled success | new training episode for the unload policy
```

For a teleoperated load, the same lines with `load` and `pickup -> work area` (training episode for the load policy).
If the camera shows the work area is NOT empty after the handover, label fail and print
`RECORDING | attempt <id> | <start> -> <end> | teleop demonstration | labelled fail | not a training episode`.
Never print `finished` / `OFF confirmed` before the human has actually said so.
