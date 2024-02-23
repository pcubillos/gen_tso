# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

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

from shiny.ui import (
    card,
    card_header,
)

class NavSetCardJWST(NavSet):
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

        return card(
            card_header(self.header),
            *nav_items,
            self.footer,
        )


def navset_card_tab_jwst(
        *args,
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
    return NavSetCardJWST(
        *args,
        ul_class="nav nav-pills",
        id=resolve_id_or_none(id),
        selected=selected,
        header=header,
        footer=footer,
    )
