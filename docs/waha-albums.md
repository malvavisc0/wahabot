# WAHA Album Messages

How multi-image albums ("media albums") arrive via the WAHA `message`
webhook, and how wahabot reassembles them.

## Delivery sequence (verified on NOWEB, 2026-09)

An album is **not** one webhook. It is three:

1. **Album container** (`_data.type == "album"`) arrives first — a shell
   with no media:
   - `hasMedia: false`, `body: ""`
   - `_data.expectedImageCount: 2` — announces how many images follow
     (`expectedVideoCount` for videos)
2. **One `message` webhook per image** (`_data.type == "image"`,
   `hasMedia: true`, `media.url` set), arriving back-to-back right
   after the container.

**There is no linkage field on NOWEB**: the images arrive with
`parentMsgId: null` and no `parentMsgKey` (those are WEBJS internals —
see the engine caveat). The container's `expectedImageCount` plus
arrival order is the only grouping signal this engine provides.

Containers/images sent *from the bot's own account* (`fromMe`) never
emit webhook events at all — own outbound media does not echo.

## Reassembly (implemented in `src/wahabot/ai/albums.py`)

- The container opens a per-chat buffer holding its
  `expectedImageCount`.
- The next `expected` image events **in the same chat** fill the buffer
  in arrival order — WhatsApp delivers an album's images back-to-back,
  so order is the reliable key when no parent link exists.
- The buffer completes when the count is reached, or after a 3s
  timeout (the count is a hint, not a guarantee — a dropped image must
  not hang the album forever).
- A completed album runs the agent **once**, with every image attached
  as `image_blocks` on a single turn — never one run per image.
- An image with no open album buffer is a standalone photo: handled
  directly, unchanged.

## Engine caveat

`parentMsgKey`, `associationType`, `expectedImageCount` are documented
as WEBJS engine internals (`_data`). On NOWEB only
`expectedImageCount` is present; the parent linkage is not. Re-verify
field names when switching engines (GOWS/WEBJS/WPP) — the order-based
grouping above degrades gracefully to WEBJS too, since its albums also
arrive back-to-back.

## Reference

- Events: https://waha.devlike.pro/docs/how-to/events/#message
- Media files: https://waha.devlike.pro/docs/how-to/receive-messages/#media-files
