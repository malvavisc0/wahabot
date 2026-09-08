# wahabot

An agentic AI bot living in your WhatsApp — an LLM agent that *actually thinks*. It reads the conversation, decides what it needs, calls tools, reads the results, and only then replies. Built on the [WAHA](https://waha.devlike.pro) HTTP API behind a small FastAPI webhook.

It hears voice notes. It sees photos and videos. It searches the web, checks stock prices, pulls YouTube transcripts, sends documents, reacts with emoji, escalates to a human when someone asks for one, and stays quiet when it has nothing to add — all without anyone saying "use a tool." And it's yours to command from your own phone: message the bot's account and it runs your instruction with its full toolset — across chats, not just the one you're in.

```
StartEvent ──► prepare_chat_history ──► InputEvent
  user turn into memory,                   │
  frames stashed for call #1               ▼
                                   handle_llm_input
                                           │
                 ┌─────────────────────────┴──────────┐
                 │  no tool calls                     │  tool calls
                 ▼                                    ▼
              StopEvent ──► reply for the chat   handle_tool_calls ──► InputEvent (next round)
```

Brakes on the loop: `stay_silent` ends the run quietly, a repeated tool
call is never executed, a round limit force-wraps the run, and a hard
timeout keeps the webhook free. A final text produced after a delivery
tool already fired is dropped — the chat saw it once, not twice.

## Giving the model hands

An LLM by itself can only talk. wahabot gives it the chat: it can send messages, images and files, react with emoji, forward posts, read back through history, and search old messages. Beyond WhatsApp it searches the web, fetches pages with a real Chrome TLS fingerprint (most sites answer as if a browser asked), looks up stock prices, pulls YouTube transcripts, and — if you opt in — runs shell commands on the host. Reaching *other* chats — messaging or reading a person or group outside the conversation that woke the bot — is reserved for operator commands; chat runs are fenced to the current conversation (message ids included), so no participant can make the bot DM or spy on anyone.

The model picks the tool, the workflow executes it, feeds the result back, and the model decides whether it needs another round. It stops when it's done, not when a script says so.

Every tool answers with a small JSON envelope — `{"ok": true, ...}` or `{"ok": false, "error": "..."}` — and never raises: a failed lookup comes back as data the model can shrug off or retry. The whole run is capped at 120 s, so a pathological loop can't hold the webhook hostage.

## What it can see and hear

- **Photos** are downloaded, attached to that turn's LLM call, then discarded — chat memory stays text-only, no megabyte payloads rotting in the rolling buffer.
- **Voice notes** are transcribed by a WhisperX service (`WAHABOT_TRANSCRIBE_URL`) and arrive as `[voice note] <transcript>` — the bot hears what was said without being asked. Off when the URL is empty.
- **Videos** are understood as frames + spoken track (`WAHABOT_VIDEO`): evenly spaced stills are captioned by the vision model, the audio goes to WhisperX, and the turn carries `(video shows: …) [audio: "…"]` as a durable text anchor. The frames ride the first LLM call only; needs ffmpeg on PATH (the Docker image ships it).
- **Albums** arrive as a container plus N images; the handler buffers them and runs the agent once, all images attached.
- **Bare image links** in text are sniffed out, fetched, and shown to the model too.
- **Reactions** to the bot's own messages are folded into memory as context — a 👍 lands quietly, visible on the next turn, never waking the agent.

## Talking to the bot

`wahabot tell` gives the operator a direct line — not a chat message, a command run by the same agent, with its full toolset:

```bash
uv run wahabot tell "send a message to Ana: the deploy is done"
uv run wahabot tell "search the latest news about elon musk and send a summary to the group Familia"
```

The agent runs the instruction with its full toolset over the operator's own rolling history — commands share one conversation, so follow-ups ("now send that to the second group") work without restating context. No whitelist applies, no chat's history is touched, and names resolve to the right person or group automatically. The result lands in WhatsApp, not in your terminal.

You can send the same kind of command from WhatsApp by messaging the bot's own account, using its configured mention pattern:

```text
kAI do this and send a message to Roy
```

Only a matching message sent to the bot's own self-chat is treated this way — the bot's reply comes back as a quote-reply in that same chat. A voice note works too: speak "kAI do this…" and it transcribes and runs like the typed command. Messages you type from the bot account in other chats remain memory-only, and the bot never re-triggers on its own replies.

Chat participants have one sanctioned way to reach you: the `escalate` tool. When someone asks for a human, reports a problem, or complains about the bot, it forwards a bot-written report to your self-chat — once per chat per hour, never pasting the person's words (so hidden instructions can't ride the channel).

Operator commands are also the **only** runs with cross-chat reach: tools refuse to send, forward, react to, quote or read outside the current conversation on any chat-triggered run — `chat` JIDs and serialized message ids alike — so a group participant can never make the bot DM or spy on someone else. `resolve_chat` and `recent_chats` (the contact roster and chat list) refuse to run at all outside operator commands.

`wahabot forget <chat-id>` wipes one chat's persistent memory in the running bot (live context and disk file, under that chat's run lock):

```bash
uv run wahabot forget "1234567890-1234567890@g.us"
```

## Install

See [Installation guide](docs/install.md) — setup, the full env var reference,
quick start, and CLI commands.

## Startup logs

`wahabot serve` prints a short banner so one glance tells you what's running:
version + Python, session name, LLM model/endpoint, memory token ceiling, and
enabled features (vision / video / shell / transcription / langfuse). It then
logs the WAHA session's live identity, the loaded session config summary, and
the toolset the agent was built with:

```
Info: wahabot 0.4.8 (Python 3.14.6)
Info: Session: default
Info: LLM: gpt-4o-mini @ https://api.openai.com/v1
Info: Memory: 8000 token ceiling
Info: Features: vision, video, no-shell, no-transcribe
Info: Webhook: http://0.0.0.0:8080/api/webhook/default
Info: WAHA session default is live as My Name (4917...@c.us)
Info: Loaded session config from data/sessions/default.json: 0 whitelisted, 0 blacklisted, group_participation=mentioned
Info: Agent ready: fetch_chat_messages, forward_message, ...
```

For a machine-readable dump of every `WAHABOT_*` value (secrets redacted) use
`uv run wahabot config`. The shell tool shows up only with
`WAHABOT_SHELL_TOOL=true`, Langfuse tracing only when the `LANGFUSE_*` keys
are set, and the `Agent ready` line lists exactly what the model can call
this session.

## Observability

Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` and every agent turn is exported to [Langfuse](https://langfuse.com) — prompts, completions, token counts, latency, tool calls — as one session per WhatsApp chat. JIDs are masked before they leave the process. Without credentials, tracing is a no-op and nothing leaves.

## Architecture (deep dive)

The agent workflow lives under `src/wahabot/ai/` as a set of focused modules:

| Module | Role |
|---|---|
| `workflow.py` | The three-step `FunctionCallingAgentWorkflow`, `load_llm`, `build_agent` |
| `events.py` | `InputEvent` / `ToolCallEvent` |
| `context.py` | Sender tagging, reply-context rendering, `handle_message` entrypoint |
| `messages.py` | Message classification, `extract_text`, `image_media`, `video_media`, `is_replyable` |
| `albums.py` | Album reassembly: container + images buffered into one agent turn |
| `history.py` | `sanitize_chat_history` (repair) + `trim_to_budget` (token budget) |
| `tools/whatsapp.py` | WhatsApp actions: send, react, forward, search, resolve chats, escalate, list recent chats |
| `tools/external.py` | Web, finance, YouTube & (opt-in) shell tool builders |
| `tools/schemas.py` | Pydantic parameter schemas for every tool |
| `tools/envelope.py` | The unified JSON envelope (`ok` / `error`) every tool returns |
| `tools/web_search.py` / `tools/visit_url.py` / `tools/url_images.py` / `tools/shell.py` | Web lookup, image-URL & shell tool functions |
| `tools/finance.py` / `tools/youtube.py` | Market data and transcript tools |
| `video.py` / `vision.py` | Video frame extraction + anchor; image captions |
| `observability.py` | Langfuse export |

Before every LLM call, the chat history passes through two hygiene steps: **repair** (fixes dangling tool calls, orphan messages, trailing user turns that would make the API reject the payload) and **trim** (keeps the newest tail that fits the token budget, treating tool-call groups as atomic).

Memory is keyed by `(session, chat_id)` — each WhatsApp conversation gets its own continuous context. Tool results are stored as `role="tool"` messages so the model can reference them across the loop. Memory is **persisted** to `data/memory/<session>/<chat>.json` at the end of every run, so it survives restarts and LRU evictions; `wahabot forget <chat>` wipes one chat.

## Development

```bash
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run basedpyright            # type check
uv run radon cc src -s         # complexity (no C+ blocks allowed)
uvx --python 3.14 vulture src/ --min-confidence 60   # dead code
uv run pytest                          # end-to-end smoke suite
```

## Docs

- [Installation — setup, env vars, quick start](docs/install.md)
- [Agent workflow — full pipeline explanation](docs/agent-workflow.md)
- [How conversations work — contexts, memory, group participation](docs/conversations.md)
- [Session config — fields, group participation, access control](docs/session-config.md)
- [WAHA identity fields — `from` / `participant` / `to` semantics](docs/waha-identity-fields.md)
- [WAHA albums — multi-image reassembly](docs/waha-albums.md)
- [WAHA broadcast sources — status/newsletter handling](docs/waha-broadcast-sources.md)
- [Coding standard — the house rules the codebase follows](docs/coding-standard.md)
