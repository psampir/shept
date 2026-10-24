"""The About window: one `app.about` action, one window, correct contents.

The action is the only way in, from the header-bar menu and from a caller that
has no menu at all, so the tests go through `activate_action` exactly like the
menu item does instead of calling the handler or the builder directly. The
window is not a child of any widget, which is why the first test also proves
that activating the action twice raises one window rather than stacking two.
"""

from __future__ import annotations

import pytest
from gi.repository import Adw, Gio, Gtk

import shept

#: The metadata the window has to carry, in one place so the test that reads it
#: back and the assertion that explains a failure cannot drift apart.
WEBSITE = "https://github.com/psampir/shept"
ISSUE_URL = "https://github.com/psampir/shept/issues"
DEVELOPER = "Paweł Sampir"


def _about_windows() -> list[Adw.AboutWindow]:
    """Every About window on screen, of any application.

    `app.get_windows()` does not include them: `Adw.AboutWindow` is only made
    transient for the main window, which is not how a window joins a
    `GtkApplication`. The toplevel list is the one place that sees both.
    """
    return [
        window
        for window in Gtk.Window.get_toplevels()
        if isinstance(window, Adw.AboutWindow)
    ]


@pytest.fixture(autouse=True)
def _close_about_windows():
    """Destroy the windows a test opened, so they cannot leak into the next."""
    yield
    for about in _about_windows():
        about.destroy()


def test_the_about_action_is_installed_and_enabled(app):
    action = app.lookup_action("about")
    assert isinstance(action, Gio.SimpleAction)
    assert action.get_enabled() is True


def test_activating_the_action_opens_exactly_one_window(app, window):
    window.present()
    assert _about_windows() == []

    app.activate_action("about", None)

    opened = _about_windows()
    assert len(opened) == 1
    about = opened[0]
    assert about.get_visible() is True
    # The App window is its parent, so the About window is not an orphan in
    # the taskbar when the menu is opened from a window that is not focused.
    assert about.get_transient_for() is window

    # A second click on the menu item must raise the window that exists, not
    # stack an identical copy behind it.
    app.activate_action("about", None)

    assert _about_windows() == [about]


def test_the_window_reports_the_package_version_licence_and_links(app, window):
    window.present()
    app.activate_action("about", None)

    about = _about_windows()[0]
    version = about.get_property("version")

    # The version comes from the package, so the About window cannot claim one
    # release while `shept --version` and the wheel say another.
    assert version == shept.__version__
    assert about.get_property("license-type") == Gtk.License.MIT_X11
    assert about.get_property("website") == WEBSITE
    assert about.get_property("issue-url") == ISSUE_URL
    assert about.get_property("developer-name") == DEVELOPER
    assert DEVELOPER in about.get_property("developers")
    assert about.get_property("application-name") == "Shept"
    assert about.get_property("application-icon") == shept.APP_ID
