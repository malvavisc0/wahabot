import re

IMPERSONATE = "chrome"

MAX_CHARS = 4000

DESCRIPTION_CHARS = 800

MAX_TRANSCRIPT_CHARS = 6000

MAX_POST_ITEMS = 100

META_TAG_RE = re.compile(r"<meta\s[^>]*>", re.I)

META_KEY_RE = re.compile(
    r"""(?:property|name)=["'](og:title|og:description|description|twitter:title|twitter:description)["']""",
    re.I,
)

META_CONTENT_RE = re.compile(r"""content=["']([\s\S]*?)["']""", re.I)

META_KEYS = {
    "og:title": "title",
    "og:description": "description",
    "description": "description",
    "twitter:title": "title",
    "twitter:description": "description",
}

MAX_META_CHARS = 2000

TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>", re.I)

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]{2,}| *\n *|\n{3,}")

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

PARAGRAPH_SENTENCES = 3

MEDIA_HOSTS = (
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "twitter.com",
    "x.com",
    "twitch.tv",
)

MEDIA_HOST_RE = re.compile(
    r"^https?://(?:[\w-]+\.)?(?:"
    + r"|".join(re.escape(host) for host in MEDIA_HOSTS)
    + r")/",
    re.IGNORECASE,
)

YOUTUBE_HOST_RE = re.compile(
    r"^https?://(?:[\w-]+\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/",
    re.IGNORECASE,
)

YOUTUBE_CLIENT = "android"
