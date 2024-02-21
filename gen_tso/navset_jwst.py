from typing import (
    Literal,
    Optional,
    Sequence,
)
from htmltools import (
    MetadataNode, Tag,
    TagChild,
)

from shiny.ui._navs import (
    NavSet,
    NavSetCard,
    navset_title,
    wrap_each_content,
    _make_tabs_fillable,
)
from shiny._namespaces import resolve_id_or_none

from shiny.types import NavSetArg
from shiny.ui import (
    Sidebar,
    layout_sidebar,
    CardItem,
    card,
    card_header,
    card_footer,
)

class NavSetCardJWST(NavSet):
    placement: Literal["above", "below"]
    sidebar: Optional[Sidebar]
    title: Optional[TagChild]

    def __init__(
        self,
        *args,
        ul_class: str,
        id: Optional[str],
        selected: Optional[str],
        title: Optional[TagChild] = None,
        sidebar: Optional[Sidebar] = None,
        header: TagChild = None,
        footer: TagChild = None,
        placement: Literal["above", "below"] = "above",
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
        self.sidebar = sidebar
        self.placement = placement

    def layout(self, nav: Tag, content: Tag) -> Tag:
        content = _make_tabs_fillable(content, fillable=True, gap=0, padding=0)

        contents: list[CardItem] = wrap_each_content(
            [
                child
                for child in [self.header, content]
                if child is not None
            ]
        )

        # If there is a sidebar, make a size 1 array of the layout_sidebar content
        if self.sidebar:
            contents = [
                layout_sidebar(
                    self.sidebar,
                    *contents,
                    fillable=True,
                    border=False,
                )
            ]

        nav_items = [*navset_title(self.title), nav]

        return card(
            card_header(*nav_items) if self.placement == "above" else None,
            *contents,
            card_footer(*nav_items) if self.placement == "below" else None,
            self.footer if self.footer is not None else None,
        )



def navset_card_tab_jwst(
    *args,
    id: Optional[str] = None,
    selected: Optional[str] = None,
    title: Optional[TagChild] = None,
    sidebar: Optional[Sidebar] = None,
    header: TagChild = None,
    footer: TagChild = None,
    placement: Literal["above", "below"] = "above",
) -> NavSetCard:
    """
    Render nav items as a tabset inside a card container.

    Parameters
    ----------
    *args
        A collection of nav items (e.g., :func:`shiny.ui.nav_panel`).
    id
        If provided, will create an input value that holds the currently selected nav
        item.
    selected
        Choose a particular nav item to select by default value (should match it's
        ``value``).
    sidebar
        A `Sidebar` component to display on every `nav()` page.
    header
        UI to display above the selected content.
    footer
        UI to display below the selected content.

    Example
    -------
    See :func:`~shiny.ui.nav_panel`
    """

    return NavSetCardJWST(
        *args,
        ul_class="nav nav-pills card-header-pills",
        id=resolve_id_or_none(id),
        selected=selected,
        title=title,
        sidebar=sidebar,
        header=header,
        footer=footer,
        placement=placement,
    )

