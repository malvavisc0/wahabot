# wahabot

wahabot is a self-hosted AI agent for WhatsApp, connected through the [WAHA](https://waha.devlike.pro) HTTP API. It brings conversation memory, media understanding, web research, and operator-controlled actions to direct messages and groups. Use an OpenAI-compatible model endpoint and keep the bot's configuration and conversation records on your own machine.

The agent reads the context, decides whether a response is useful, gathers the information it needs, and replies. It can transcribe voice notes, analyze images and sampled video frames, send real files, generate or edit images when an image service is configured, and ask for missing information. Its language and personality come from your session prompt. For natural replies, prioritize complete sentences and enough context to be understood, rather than imposing a fixed message-length limit.

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

The bundled tools cover text and media delivery, quoting and numeric mentions, reactions, native forwarding, chat context, published-answer search, web search, page and video-metadata retrieval, human escalation, guarded HTTP/SQL skills, operator-only ticket identity linking, an optional host shell, and optional image generation/editing. Python builder docstrings are for developers; explicit tool descriptions and parameter schemas are the instructions sent to the model.

The loop has brakes: `stay_silent` ends a run and cancels other tools in that batch, repeated calls are blocked, and round and run-time limits stop runaway work. Text, media, and forwarding share one successful-send allowance per run; reactions have a separate allowance. After successful delivery, further research or actions are blocked. In groups, mentioned mode responds when addressed, while judicious mode asks the triage gate whether an unaddressed message is worth a run; once admitted, the model still decides whether to speak.

## Know What Was Retrieved

- `read_chat` searches a bounded recent-message window, not every message ever sent. List/search results are records in the same grammar the prompt uses, each carrying its `#m<n>` when known; the full raw window is spilled to a file. Display names and metadata are best-effort; missing fields or search hits do not prove absence.
- `web_search` caps the total valid results across engines and reports partial engine failures. Snippets are leads; open relevant sources before treating their contents as verified.
- `visit_url` fetches response text or available media metadata without browser rendering. A login wall is not the requested article. YouTube captions are best-effort, may be automatic, and do not establish what a video visually shows.
- Large results attempt to spill to local files. The model can read them only when the shell tool is enabled; otherwise the operator can open them. A truncation flag does not guarantee that a file was successfully written.
- The optional shell enforces an execution deadline and attempts bounded process-group cleanup. Output previews are byte-capped, and capture errors are reported. `ok: true` means the command ran; check `exit_code` to determine whether it succeeded. The shell is not a sandbox.

## Built for how people actually text

Nobody sends one clean, complete request. The leak arrives as "there's water coming through the ceiling", then "flat 3B", then a video. wahabot holds the burst, waits for the sender to finish, and answers once instead of asking "which flat?" three times.

- Voice notes are transcribed (a WhisperX service) and heard without anyone asking. It can talk back in its own voice too, if you point it at a TTS service: the model writes the line, the configured voice speaks it.
- If you point it at an image service, it can draw a picture from a caption or repaint one you send; without that service the tools are simply absent.
- Photos are analyzed by the vision model. Videos use sampled frames and, when transcription is configured, the audio track; this is not exhaustive frame-by-frame viewing. Albums arrive as one turn with their images attached.
- Supported reel, TikTok, and X links can be resolved through yt-dlp and passed through the video pipeline within configured limits. YouTube links use available metadata and captions rather than the video-download path; retrieval can fail or be incomplete.
- Every chat has its own rolling memory. With persistence configured, the bot can pick up the conversation after a restart.
- The window is not the memory. A pinned state block keeps exact facts and rules in front of the model, a per-chat recap summarizes turns that leave the window, and a per-chat thread index recalls older slices as lead-only context. A small triage gate decides whether an unaddressed group message is worth waking the main model for; DMs and addressed messages always run. If the gate service is down, DMs still answer and unaddressed groups stay silent.

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

- WhatsApp tools fence ordinary participants to the current chat; cross-chat delivery and contact-book access belong to trusted operator commands. `escalate` is the fixed-destination human-handoff exception outside tickets; customer tickets use `hand_to_operator`, which records the ticket, copies its media and alerts you. These checks do not sandbox an enabled shell, so run it unprivileged with an appropriate external sandbox.
- When someone needs a human, reports a bot problem, or complains, `escalate` can forward a report to your self-chat. The model is instructed to summarize without pasted messages or secrets; the tool does not sanitize arbitrary report text. Successful forwards have an in-memory per-chat hourly cooldown. Feedback distinguishes success, cooldown, and unconfirmed delivery.
- Actions and outcomes are recorded in the audit journal on a best-effort basis (`data/audit/`). `wahabot escalations` lists recorded handoffs; a logging failure does not undo an already completed action.
- `wahabot forget <chat-id>` wipes one chat's memory — live context, disk file, durable notes, recap, and thread index — in one shot.
- Accepted messages and outbound sends are recorded in `data/wahabot.db` (SQLite/WAL, migrations at startup), so a restart can finish pending work or escalate it without duplicating a confirmed send. Back it up with `notebook.sqlite3`.
- The bot keeps a private durable notebook per chat (`data/notebook.sqlite3`): verified facts, preferences and lessons the model itself records with the `notebook` tool, injected into every later run's prompt. Notes survive restarts and history resets; after each delivered reply the bot also files a short self-extracted lesson. Chat notes are never authority to override your policies — only operator commands can write bot-wide ones.

## Tickets and the operator desk

Every customer 1:1 conversation is a ticket. The bot opens it silently, answers when it can, and hands it to you only when a person must act. The handover alert carries the customer's words, the details already collected, what is missing, and the original photo, voice note or file. Reply in your own "Message yourself" chat by quoting the alert with what you did; the bot drafts the message in the customer's language, refuses a draft that adds an amount or date you did not give, sends it, and confirms the ticket state.

The operator console is the self-chat itself: reply to an alert or name the ticket (`142: refunded`), ask `what's open?`, `close 142`, `reopen 142`, `pause`, `resume`, `pause <name>` or `resume <name>`. Typing in a customer chat pauses the bot there for two hours; messages received while paused are saved and never replayed. Only one ticket per chat is active, and a different issue is split into a new ticket only from an open or done ticket, so operator work is never detached.

## Knowledge, documents and support hours

Published answers (question/answer pairs) can be saved from the operator desk: reply to a ticket with an explanation, then send `save` (or `save N`) when asked. The enabled set rides the prompt when it fits and is searchable with `search_knowledge` when it does not. `knowledge_only` is an opt-in answers-only mode; the personal default still answers from the full conversation and tools.

PDFs, scanned PDFs, DOCX/ODT/RTF, plain text and CSV are read during the turn within page, character, size and time limits. The extracted text is marked untrusted in the prompt, and scanned pages fall back to Tesseract. Unreadable or oversized documents stay attached with a plain-language notice.

Support hours live in the session config: non-urgent handovers outside hours are queued for the next opening, urgent ones alert immediately (and the configured night number once it has written), and the first opening of each local day sends one operator summary. Internal team groups are an allowlist: they run mentioned-only and are fenced in code from ticket writes, customer sends, TTS and greetings.

## Skills: guarded integrations

A skill is a folder the operator installs into `data/skills/<name>/`. It contains a `skill.yaml` contract and a `SKILL.md` note the model reads when it is relevant. Four supporting skills ship as templates in `src/wahabot/skills/predefined/`: generic REST, Supabase, a signed event sender (`send-to-url`), and read-only Postgres.

```yaml
name: customer-records
kind: http
base_url: https://crm.example.com/api
auth: {type: bearer, secret: CRM_API_KEY}
customer_chats: read
writes_need_operator: true
test: {method: GET, path: /health}
```

```bash
uv run wahabot skills list
uv run wahabot skills import ./customer-records --enable
uv run wahabot skills test customer-records
uv run wahabot skills enable customer-records
uv run wahabot skills disable customer-records
```

Secrets are encrypted with Fernet using `WAHABOT_SECRET_KEY` (base64, 32 bytes) and never returned to the model, web layer, prompts, logs, journal, History, tool results or errors. Without the key, secret-bearing skill calls refuse. Import validates that every referenced secret exists. `--enable` and `skills test` run only the prepared safe read (`GET`/`HEAD`/`OPTIONS`) from `skill.yaml`; they never run a write, and a changed service URL, key, instructions or permission marks the test stale until it passes again.

The runtime locks every request to the skill's base origin, refuses path escapes, absolute URLs, host-changing redirects, loopback/private/link-local/metadata addresses (unless the skill itself is a private service), pins the resolved IP, re-checks each redirect, caps body/response/time, and permits five calls per turn with repeated identical calls refused. Customer and team-group runs may use only the sender's own phone, chat id, external user ref and ticket number; another person's identifier is refused.

Customer writes are staged unless `writes_need_operator` is false; team-group
writes always ask. The self-chat shows the exact prepared request and asks
`Reply yes.`; quoting that message and answering yes executes the stored request
exactly once. Operator commands execute writes directly. Every call lands in
`skill_calls`, the ticket event trail and the audit journal.

`sql_query` is operator-only in this version: one `SELECT` or `WITH ... SELECT`, SQLite opened `mode=ro`, a read-only Postgres transaction when the optional `postgres` extra is installed, comment/multi-statement/`ATTACH`/`PRAGMA` refusal, and row/column/byte/time caps. Enabled event skills run after the ticket transaction commits for opened, needs-you, done and closed; failures never roll back or block the ticket. `send-to-url` signs a stable payload (event, event id, ticket number, state, resolver, external contact ref, timestamp) with HMAC-SHA256, retries boundedly, and sends no message body by default.

The skill loader, CLI, tests and runtime are Phase 4. The web authoring UI, marketplace, workflow builder and arbitrary code execution are intentionally deferred to Phase 5.

## Straight talk about the channel

Two things you should know before you put this on a business number:

- The connection is unofficial. WAHA drives a WhatsApp Web session, and unofficial automation can violate WhatsApp's terms and lead to a ban. Use a separate number, not one your business depends on.
- Ordinary runs respond to incoming messages. Trusted operator commands can initiate a message to another chat, so permissions and responsible use still matter. The project is not an outreach scheduler, and no reply policy eliminates the risk of a number being banned.

Memory, event journals, audit records and the intake/outbound ledger live on your machine. Conversation content is still sent to the model endpoint and any configured transcription, TTS, image, or tracing service. JID masking in tracing does not anonymize all message content. Configure access, retention, and service providers accordingly; tool spill files also depend on host temporary-file cleanup.

## Where this is heading

The current project is the conversation engine with durable tickets, a WhatsApp operator desk, published answers, bounded document reading, support hours with a restart-safe morning summary, fenced internal team groups, and the Phase 4 skills runtime. The Phase 5 web UI (including History and skill authoring), a marketplace, a workflow builder and arbitrary code execution are future directions, not bundled features.

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

The agent is a LlamaIndex workflow with per-chat memory repaired and trimmed before model calls and persisted to `data/memory/` when configured. The trim hooks feed a per-chat recap and thread index; the pinned state block rides the system message, so eviction never removes it. Tools return JSON envelopes, and the workflow converts unexpected tool exceptions into error feedback. Argument types are validated strictly; compact schemas retain explicit `null` support. Old history keeps command exit codes and capture errors so execution failures do not become apparent successes.

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
- [Installation: skills](docs/install.md#skills) (folder format, secrets, CLI, limits)
- [WAHA identity fields](docs/waha-identity-fields.md) (`from` / `participant` / `to` semantics)
- [WAHA albums](docs/waha-albums.md) (multi-image reassembly)
- [WAHA broadcast sources](docs/waha-broadcast-sources.md) (status/newsletter handling)
- [Coding standard](docs/coding-standard.md) (the house rules)
