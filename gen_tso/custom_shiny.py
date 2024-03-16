# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'label_tooltip_button',
    'navset_card_tab_jwst',
]

from typing import Optional
from htmltools import (
    Tag,
    TagChild,
)
from shiny.ui._navs import (
    NavSet,
    NavSetCard,
    navset_title,
)
from shiny._namespaces import resolve_id_or_none
import shiny.ui as ui


def label_tooltip_button(
        label, tooltip_text, icon, label_id, button_id, placement='top',
    ):
    """
    A label text which has a tooltip and a clickable icon.
    """
    return ui.tooltip(
        ui.span(
            label,
            ui.input_action_link(
                id=button_id,
                label='',
                icon=icon,
            ),
        ),
        tooltip_text,
        placement=placement,
        id=label_id,
    )


class NavSetCardJWST(NavSet):
    """
    A NavSet look alike, but it doesn't really has multiple tab contents.
    """
    title: Optional[TagChild]

    def __init__(
        self,
        *args,
        ul_class: str,
        id: Optional[str],
        selected: Optional[str],
        title: Optional[TagChild] = None,
        header: TagChild = None,
        footer: TagChild = None,
    ) -> None:
        super().__init__(
            *args,
            ul_class=ul_class,
            id=id,
            selected=selected,
            header=header,
            footer=footer,
        )
        self.title = title
        self.header = header

    def layout(self, nav: Tag, content: Tag) -> Tag:
        nav_items = [*navset_title(self.title), nav]

        return ui.card(
            ui.card_header(self.header),
            *nav_items,
            self.footer,
        )

def navset_card_tab_jwst(
        nav_panel_labels,
        id: Optional[str] = None,
        selected: Optional[str] = None,
        header: TagChild = None,
        footer: TagChild = None,
    ) -> NavSetCard:
    """
    Render nav items as a tabset inside a card container.

    Parameters
    ----------
    *args
        A collection of nav items (e.g., :func:`shiny.ui.nav_panel`).
    id
        input value that holds the currently selected nav item.
    selected
        Choose a particular nav item to select by default value
    header
        UI to display above the selected content.
    footer
        UI to display below the selected content.
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
        header=header,
        footer=footer,
    )

