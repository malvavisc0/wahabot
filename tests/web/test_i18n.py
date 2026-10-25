"""The operator strings load from the gettext catalog."""

from fastapi.testclient import TestClient

from wahabot.web.i18n import _, translator


def test_catalog_loads_primary_strings() -> None:
    """The compiled English catalog answers the navigation and form strings."""
    catalog = translator()
    assert catalog.gettext("History") == "History"
    assert _("Sign in") == "Sign in"
    assert _("Save Assistant") == "Save Assistant"
    assert _("a string not in the catalog") == "a string not in the catalog"


def test_pages_render_catalog_strings(auth_client: TestClient) -> None:
    """A rendered page carries the translated labels, not hard-coded markup."""
    page = auth_client.get("/app/settings")
    assert "Saved changes apply" in page.text
    assert "Who it answers" in page.text
    assert "Save Assistant" in page.text
