# wahabot

wahabot is a self-hosted AI agent for WhatsApp, connected through the [WAHA](https://waha.devlike.pro) HTTP API. It brings conversation memory, media understanding, web research, and operator-controlled actions to direct messages and groups. Use an OpenAI-compatible model endpoint and keep the bot's configuration and conversation records on your own machine.

The agent reads the context, decides whether a response is useful, gathers the information it needs, and replies. It can transcribe voice notes, analyze images and sampled video frames, send real files, and ask for missing information. Its language and personality come from your session prompt. For natural replies, prioritize complete sentences and enough context to be understood, rather than imposing a fixed message-length limit.

The goal is useful participation, not a bot that answers every group message or pretends to be a person.

## It thinks before it speaks

wahabot runs a tool-calling loop: read the conversation, decide what is needed, call a tool, inspect the result, then reply. A model that supports function calling is required; image and video understanding also need a vision-capable endpoint. Research and actions finish before delivery, so the final answer can describe what actually happened.

```text
Incoming message -> context and memory -> model
                                            |
                             tool call -> inspect result -> model
                                            |
                              final reply, reaction, or silence
```

The bundled tools cover text and media delivery, quoting and numeric mentions, reactions, native forwarding, chat context, web search, page and video-metadata retrieval, human escalation, and an optional host shell. Python builder docstrings are for developers; explicit tool descriptions and parameter schemas are the instructions sent to the model.

The loop has brakes: `stay_silent` ends a run and cancels other tools in that batch, repeated calls are blocked, and round and run-time limits stop runaway work. Text, media, and forwarding share one successful-send allowance per run; reactions have a separate allowance. After successful delivery, further research or actions are blocked. In groups, mentioned mode responds when addressed, while judicious mode lets the model decide whether it has something worth adding.

## Know What Was Retrieved

- `read_chat` searches a bounded recent-message window, not every message ever sent. Display names and metadata are best-effort; missing fields or search hits do not prove absence. Inline previews identify cut bodies separately from omitted messages.
- `web_search` caps the total valid results across engines and reports partial engine failures. Snippets are leads; open relevant sources before treating their contents as verified.
- `visit_url` fetches response text or available media metadata without browser rendering. A login wall is not the requested article. YouTube captions are best-effort, may be automatic, and do not establish what a video visually shows.
- Large results attempt to spill to local files. The model can read them only when the shell tool is enabled; otherwise the operator can open them. A truncation flag does not guarantee that a file was successfully written.
- The optional shell enforces an execution deadline and attempts bounded process-group cleanup. Output previews are byte-capped, and capture errors are reported. `ok: true` means the command ran; check `exit_code` to determine whether it succeeded. The shell is not a sandbox.

## Built for how people actually text

Nobody sends one clean, complete request. The leak arrives as "there's water coming through the ceiling", then "flat 3B", then a video. wahabot holds the burst, waits for the sender to finish, and answers once instead of asking "which flat?" three times.

- Voice notes are transcribed (a WhisperX service) and heard without anyone asking. It can talk back in its own voice too, if you point it at a TTS service: the model writes the line, the configured voice speaks it.
- Photos are analyzed by the vision model. Videos use sampled frames and, when transcription is configured, the audio track; this is not exhaustive frame-by-frame viewing. Albums arrive as one turn with their images attached.
- Supported reel, TikTok, and X links can be resolved through yt-dlp and passed through the video pipeline within configured limits. YouTube links use available metadata and captions rather than the video-download path; retrieval can fail or be incomplete.
- Every chat has its own rolling memory. With persistence configured, the bot can pick up the conversation after a restart.

## You stay the boss

From your phone: message the bot's own account with its mention pattern and it runs your instruction with its full toolset, across chats, not just the one you're in.

```text
kAI do this and send a message to Roy
```

A voice note works too: speak "kAI do this..." and it transcribes and runs like the typed command. From your terminal, the same thing:

```bash
uv run wahabot tell "search the latest news about elon musk and send a summary to the group Family"
```

Commands share one rolling history for follow-ups. Names are matched against available chats and contacts; genuinely ambiguous recipients need clarification before sending. Commands must supply an explicit destination for delivery. Their final text returns to the terminal or self-chat caller, and requested messages or media go to the selected WhatsApp chat.

The fences you want in a business bot:

- WhatsApp tools fence ordinary participants to the current chat; cross-chat delivery and contact-book access belong to trusted operator commands. `escalate` is the fixed-destination human-handoff exception. These checks do not sandbox an enabled shell, so run it unprivileged with an appropriate external sandbox.
- When someone needs a human, reports a bot problem, or complains, `escalate` can forward a report to your self-chat. The model is instructed to summarize without pasted messages or secrets; the tool does not sanitize arbitrary report text. Successful forwards have an in-memory per-chat hourly cooldown. Feedback distinguishes success, cooldown, and unconfirmed delivery.
- Actions and outcomes are recorded in the audit journal on a best-effort basis (`data/audit/`). `wahabot escalations` lists recorded handoffs; a logging failure does not undo an already completed action.
- `wahabot forget <chat-id>` wipes one chat's memory, live context and disk file.

## Straight talk about the channel

Two things you should know before you put this on a business number:

- The connection is unofficial. WAHA drives a WhatsApp Web session, and unofficial automation can violate WhatsApp's terms and lead to a ban. Use a separate number, not one your business depends on.
- Ordinary runs respond to incoming messages. Trusted operator commands can initiate a message to another chat, so permissions and responsible use still matter. The project is not an outreach scheduler, and no reply policy eliminates the risk of a number being banned.

Memory, event journals, and audit records live on your machine. Conversation content is still sent to the model endpoint and any configured transcription, TTS, or tracing service. JID masking in tracing does not anonymize all message content. Configure access, retention, and service providers accordingly; tool spill files also depend on host temporary-file cleanup.

## Where this is heading

The current project is the conversation engine. Durable case ownership, CRM/job-system connectors, and an operator console are future directions, not bundled features.

## Install

Python 3.14+, [uv](https://docs.astral.sh/uv/), a reachable WAHA instance, and a function-calling model endpoint are required. See the [installation guide](docs/install.md) for configuration, optional media services, and CLI commands.

```bash
uv sync
cp .env.example .env   # fill in the LLM + WAHA values
uv run wahabot sessions init
uv run wahabot serve
```

Configure your WAHA webhook and session prompt as described in the guide. Inspect the expanded prompt with `uv run wahabot sessions view --name default`. Session configuration under `data/` is local runtime data, not part of release artifacts.

## Observability

Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` to export agent traces to [Langfuse](https://langfuse.com): prompts, completions, token counts, latency, and tool calls, as one session per WhatsApp chat. Without these credentials, no traces are exported; calls to configured model and media services still occur.

## Under the hood

The agent is a LlamaIndex workflow with per-chat memory repaired and trimmed before model calls and persisted to `data/memory/` when configured. Tools return JSON envelopes, and the workflow converts unexpected tool exceptions into error feedback. Argument types are validated strictly; compact schemas retain explicit `null` support. Old history keeps command exit codes and capture errors so execution failures do not become apparent successes.

The module map, engine semantics and every safeguard explained: [Agent workflow](docs/agent-workflow.md).

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run basedpyright            # type checking
uv run pytest                  # unit and end-to-end tests with fake services
```

## Docs

- [Installation](docs/install.md) (setup, env vars, quick start, startup banner)
- [Agent workflow](docs/agent-workflow.md) (the full pipeline)
- [How conversations work](docs/conversations.md) (contexts, memory, group participation)
- [Session config](docs/session-config.md) (fields, group participation, access control)
- [WAHA identity fields](docs/waha-identity-fields.md) (`from` / `participant` / `to` semantics)
- [WAHA albums](docs/waha-albums.md) (multi-image reassembly)
- [WAHA broadcast sources](docs/waha-broadcast-sources.md) (status/newsletter handling)
- [Coding standard](docs/coding-standard.md) (the house rules)
