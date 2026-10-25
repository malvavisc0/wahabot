# Send ticket events to a URL

Posts one compact JSON payload when a ticket opens, needs the operator,
is done, or is closed. The payload carries the event kind, a stable
event id, the ticket number, state, who resolved it, an external contact
reference and a timestamp. It never carries the customer's message body
or phone number.

Every request is signed with HMAC-SHA256 over the exact JSON body. The
`X-Wahabot-Signature` header is `sha256=<hex>`, the event id is repeated
in `X-Wahabot-Event-Id`, and the receiver should deduplicate on it.
Delivery is retried up to three times; a failure never rolls back or
blocks the ticket.

Configure `base_url` to the receiver, store the signing key as the
`SEND_TO_URL_SECRET` secret, and enable the events you want. The safe
check calls `GET /health`; change it if the receiver uses another path.
