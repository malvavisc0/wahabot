# Generic REST integration

Read customer records with `GET` requests under the configured base
URL. Customer chats may read; writes from a customer or team group are
staged for the operator's yes unless `writes_need_operator` is false
(team-group writes always require the operator's yes).

Replace `base_url` with the real service origin, store the API key as
the `REST_API_KEY` secret, and update the safe check path. Requests
cannot leave the configured origin, follow a redirect to another host,
or reach loopback/private/metadata addresses unless the base URL itself
is private. Customer runs may only reference the sender's own
identifiers.
