# Copyright (c) 2025-2026 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'custom_card',
    'navset_card_tab_jwst',
]

from typing import Optional
from htmltools import Tag
from shiny.ui._navs import (
    NavSet,
    NavSetCard,
)
from shiny.ui._card import card_body
from shiny._namespaces import resolve_id_or_none
import shiny.ui as ui


def custom_card(*args, body_args={}, **kwargs):
    """
    A wrapper over a Shiny card component with an explicit card_body
    (so that I can apply class_ and other customizations).
    """
    header = None
    args = list(args)
    for arg in args:
        # Only headers and footers are CardItems (right?)
        if isinstance(arg, ui.CardItem):
            header = arg
            args.remove(arg)
            break

    return ui.card(
        header,
        card_body(*args, **body_args),
        **kwargs,
    )


class NavSetCardJWST(NavSet):
    """
    A NavSet look alike, but it doesn't really has multiple tab contents.
    """
    def __init__(
        self,
        *args,
        ul_class: str,
        id: Optional[str],
        selected: Optional[str],
    ) -> None:
        super().__init__(
            *args,
            ul_class=ul_class,
            id=id,
            selected=selected,
        )

    def layout(self, nav: Tag, content: Tag) -> Tag:
        #print(nav)
        nav_style = ui.tags.style("""
            .navset-container {
                display: flex;
                width: 100%;
                flex-wrap: nowrap;
            }
            .navset-container .nav-item {
                flex-grow: 1;
                text-align: center;
                border-radius: 5px;
                margin: 1px;
                box-sizing: border-box;
                transition: border 0.3s ease;
                border: 1px solid transparent;
            }
            .navset-container .nav-item:hover {
                border: 1px solid #006cd4;
            }
        """)
        # Wrap nav items in a div with a specific class for styling
        return ui.TagList(
            nav_style,
            ui.div(nav, class_="navset-container"),
        )


def navset_card_tab_jwst(
        nav_panel_labels,
        id: Optional[str] = None,
        selected: Optional[str] = None,
    ) -> NavSetCard:
    """
    Render nav items as a pillset (buttons) but without the tabs-content.

    Parameters
    ----------
    nav_labels: List of string
        Labels to display for the nav buttons
    id: String
        input value that holds the currently selected nav item.
    selected: String
        Choose a particular nav item to select by default value
    """
    nav_panels = [
        ui.nav_panel(label, '')
        for label in nav_panel_labels
    ]
    return NavSetCardJWST(
        *nav_panels,
        ul_class="nav nav-pills",
        id=resolve_id_or_none(id),
        selected=selected,
    )
