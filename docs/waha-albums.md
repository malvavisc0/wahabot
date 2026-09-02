# WAHA Album Messages

How multi-image albums ("media albums") arrive via the WAHA `message`
webhook, and how to reassemble them.

## Delivery sequence

An album is **not** one webhook. It is three:

1. **Album container** (`_data.type == "album"`) arrives first — a shell
   with no media:
   - `hasMedia: false`, `body: ""`
   - `_data.expectedImageCount: 2` — announces how many images follow
     (`expectedVideoCount` for videos)
   - its own `id` is the *album key* other messages point at
2. **One `message` webhook per image** (`_data.type == "image"`,
   `hasMedia: true`, `media.url` set), each carrying the linkage:
   - `_data.parentMsgKey.id` — album message's raw ID
   - `_data.parentMsgKey.$1` — album's full serialized `id`
   - `_data.associationType` / `_data.viewMode` == `"MEDIA_ALBUM"`

The images point at the album, **not** vice-versa. Images can lag the
container by ~1s and may arrive in any order.

## Reassembly

```python
def album_key(payload: dict) -> str | None:
    parent = payload.get("_data", {}).get("parentMsgKey")
    return parent["$1"] if parent else None
```

- Group incoming `message` events by `album_key(event.payload)` matching
  the album container's `payload["id"]`.
- Buffer images per album until `expectedImageCount` is reached, or a
  timeout fires — the count is not a guarantee, just a hint.
- An image **without** `parentMsgKey` is a standalone photo: handle it
  directly, no album involved.

## Engine caveat

`parentMsgKey`, `associationType`, `expectedImageCount` are WEBJS
engine internals (`_data`). The album-then-images sequence should hold
across engines, but field names may differ — re-verify when switching
engines (GOWS/NOWEB/WPP).

## Reference

- Events: https://waha.devlike.pro/docs/how-to/events/#message
- Media files: https://waha.devlike.pro/docs/how-to/receive-messages/#media-files
