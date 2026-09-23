# wahabot

You know the morning. Forty-seven unread messages. A voice note you'll never find time to play. A photo with no context. A question from yesterday that nobody answered, and a customer quietly deciding you're slow. The conversations are the business, but they live in nobody's system.

wahabot is an AI assistant that lives in your WhatsApp and takes that work. It reads the conversation like a person would: it listens to the voice notes, looks at the photos and videos, asks for what's missing, searches the web when it needs facts, sends the file the customer asked for, and stays quiet when it has nothing to add. You run it on your own machine, on any OpenAI-compatible model, connected through the [WAHA](https://waha.devlike.pro) HTTP API.

You have seen the chatbot demos that answer FAQs and nothing else. This is the other kind.

## It thinks before it speaks

Keyword bots fire canned answers. wahabot runs a loop: read the conversation, decide what is needed, call a tool, read the result, then reply. If the answer needs a search, it searches. If it needs the chat history, it reads it. It stops when the work is done, not when a script says so.

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

The toolset it can reach for: send messages, images, files and voice replies, quote and @-mention people, react with emoji, forward posts, read back through chat history, search old messages, search the web, fetch pages with a real Chrome TLS fingerprint (most sites answer as if a browser asked), look up stock prices, pull YouTube transcripts, and, if you opt in, run shell commands on the host.

The loop has brakes: `stay_silent` ends a run quietly, a repeated tool call is never executed twice, a round limit force-wraps a confused run, and a hard timeout keeps the webhook free. A lone-emoji reply lands as a reaction on the triggering message, never as chat text. The bot also knows when it shouldn't speak at all: in a group it answers when addressed, and in judicious mode it reads everything and decides for itself whether it has something worth saying.

## Built for how people actually text

Nobody sends one clean, complete request. The leak arrives as "there's water coming through the ceiling", then "flat 3B", then a video. wahabot holds the burst, waits for the sender to finish, and answers once instead of asking "which flat?" three times.

- Voice notes are transcribed (a WhisperX service) and heard without anyone asking. It can talk back in its own voice too, if you point it at a TTS service: the model writes the line, the configured voice speaks it.
- Photos are looked at, videos are watched (evenly spaced frames captioned by the vision model, plus the spoken track transcribed). Albums arrive as one turn with all images attached.
- Links to reels, TikToks and X posts are resolved and watched via yt-dlp, not just read. YouTube links get the full transcript instead, because captions beat six sampled frames on long-form.
- Every chat has its own rolling memory, persisted to disk, so the bot picks up each conversation where it left off and survives restarts.

## You stay the boss

From your phone: message the bot's own account with its mention pattern and it runs your instruction with its full toolset, across chats, not just the one you're in.

```text
kAI do this and send a message to Roy
```

A voice note works too: speak "kAI do this..." and it transcribes and runs like the typed command. From your terminal, the same thing:

```bash
uv run wahabot tell "search the latest news about elon musk and send a summary to the group Family"
```

Commands share one rolling history, so a follow-up like "now send that to the second group" just works. Names resolve to the right person or group automatically, and the result lands in WhatsApp, not in your terminal.

The fences you want in a business bot:

- Chat participants can never make it message, read, forward or react outside their own conversation. Cross-chat reach belongs to your commands alone, so a group member cannot make it DM or spy on anyone.
- When someone asks for a human, reports a problem, or complains about the bot, the `escalate` tool forwards a bot-written report to your self-chat, once per chat per hour, and never pastes the person's words, so hidden instructions can't ride the channel.
- Every bot action is journaled with its arguments and outcome. When a customer asks "what did the bot do?", you can show them, message by message (`data/audit/`, listed by `wahabot escalations` for the human handoffs).
- `wahabot forget <chat-id>` wipes one chat's memory, live context and disk file.

## Straight talk about the channel

Two things you should know before you put this on a business number:

- The connection is unofficial. WAHA drives a WhatsApp Web session, which violates WhatsApp's terms. Any unofficial number can be banned, with no appeal. Run the bot on a prepaid SIM, never on the number printed on your business cards.
- The bot only ever replies. It answers people who wrote first, and it never campaigns. That is a deliberate design rule: a bot that texts first, at scale, on schedule is the spam fingerprint, and reply-only behavior keeps the bot looking like an eager employee instead.

Your data stays yours. Memory, the verbatim event journal and the audit log are files on your machine; JIDs are masked before anything leaves for tracing.

## Where this is heading

Everything above exists and runs today. The commercial roadmap turns this engine into a business product: durable cases with owners, so "a customer texted" becomes "Sarah's on it"; connectors that write into your CRM or job system; an operator console where staff claim and close the work. The plan, the pricing logic and the risk register live in [docs/plans/commercial-roadmap.md](docs/plans/commercial-roadmap.md).

## Install

See the [installation guide](docs/install.md): local setup or Docker, the full env var reference, quick start, session config, and every CLI command.

```bash
uv sync
cp .env.example .env   # fill in the LLM + WAHA values
uv run wahabot sessions init
uv run wahabot serve
```

## Observability

Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` and every agent turn is exported to [Langfuse](https://langfuse.com): prompts, completions, token counts, latency, tool calls, as one session per WhatsApp chat. Without credentials, tracing is a no-op and nothing leaves.

## Under the hood

The agent is a three-step LlamaIndex workflow (prepare the history, ask the model, run its tool calls, loop until done), with per-chat memory that is repaired and trimmed before every LLM call and persisted to `data/memory/` at the end of every run. Every tool returns a small JSON envelope, `{"ok": true, ...}` or `{"ok": false, "error": "..."}`, and never raises: a failed lookup is data the model can shrug off or retry.

The module map, engine semantics and every safeguard explained: [Agent workflow](docs/agent-workflow.md).

```bash
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run basedpyright            # type check
uv run pytest                  # end-to-end smoke suite
```

## Docs

- [Installation](docs/install.md) (setup, env vars, quick start, startup banner)
- [Agent workflow](docs/agent-workflow.md) (the full pipeline)
- [How conversations work](docs/conversations.md) (contexts, memory, group participation)
- [Session config](docs/session-config.md) (fields, group participation, access control)
- [WAHA identity fields](docs/waha-identity-fields.md) (`from` / `participant` / `to` semantics)
- [WAHA albums](docs/waha-albums.md) (multi-image reassembly)
- [WAHA broadcast sources](docs/waha-broadcast-sources.md) (status/newsletter handling)
- [Commercial roadmap](docs/plans/commercial-roadmap.md) (where the product is going)
- [Coding standard](docs/coding-standard.md) (the house rules)
