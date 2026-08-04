import streamlit.components.v1 as components


def inject_resizable_card_resizer() -> None:
    """Enable horizontal resizing for main-content Streamlit expander columns."""
    components.html(
        """
        <script>
        (() => {
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const INSTALL_KEY = "__autostatResizableCardsInstalled";
          if (parentWindow[INSTALL_KEY]) {
            return;
          }
          parentWindow[INSTALL_KEY] = true;

          const STORAGE_PREFIX = "autostat:resizable-card-columns:v3:";
          const HORIZONTAL_BLOCK_SELECTOR = '[data-testid="stHorizontalBlock"]';
          const COLUMN_SELECTOR = '[data-testid="stColumn"], [data-testid="column"]';
          const EXPANDER_SELECTOR = '[data-testid="stExpander"]';
          const MIN_VIEWPORT_WIDTH = 900;
          const MIN_COLUMN_WIDTH = 80;
          let activeDrag = null;

          function collapseHostElement() {
            const frame = window.frameElement;
            if (!frame) {
              return;
            }

            const host = frame.closest('[data-testid="stElementContainer"]');
            [frame, host].filter(Boolean).forEach((element) => {
              element.style.setProperty("display", "block", "important");
              element.style.setProperty("height", "0", "important");
              element.style.setProperty("min-height", "0", "important");
              element.style.setProperty("max-height", "0", "important");
              element.style.setProperty("margin", "0", "important");
              element.style.setProperty("padding", "0", "important");
              element.style.setProperty("border", "0", "important");
              element.style.setProperty("overflow", "hidden", "important");
              element.style.setProperty("pointer-events", "none", "important");
            });
          }

          function injectStyles() {
            if (doc.getElementById("autostat-resizable-card-styles")) {
              return;
            }

            const style = doc.createElement("style");
            style.id = "autostat-resizable-card-styles";
            style.textContent = `
              section[data-testid="stSidebar"] .autostat-resize-handle {
                display: none !important;
              }
              [data-testid="stHorizontalBlock"].autostat-resizable-card-row {
                position: relative;
                align-items: flex-start;
                overscroll-behavior-y: contain;
              }
              [data-testid="stColumn"].autostat-resizable-card-column,
              [data-testid="column"].autostat-resizable-card-column {
                position: relative;
                min-width: 0 !important;
              }
              [data-testid="stColumn"].autostat-resizable-card-column > .autostat-column-scroll-body,
              [data-testid="column"].autostat-resizable-card-column > .autostat-column-scroll-body {
                box-sizing: border-box;
                max-height: var(--autostat-column-scroll-max-height, calc(100vh - 8.25rem));
                min-width: 0;
                overflow-x: hidden;
                overflow-y: auto;
                overscroll-behavior-y: contain;
                padding-right: 0.25rem;
                scrollbar-gutter: stable;
              }
              .autostat-resize-handle {
                position: absolute;
                top: 0;
                right: -13px;
                width: 26px;
                height: 100%;
                z-index: 80;
                cursor: col-resize;
                touch-action: none;
                border-radius: 8px;
                background: transparent;
              }
              body.autostat-card-resizing,
              body.autostat-card-resizing * {
                cursor: col-resize !important;
                user-select: none !important;
              }
              @media (max-width: ${MIN_VIEWPORT_WIDTH - 1}px) {
                .autostat-resize-handle {
                  display: none !important;
                }
                [data-testid="stColumn"].autostat-resizable-card-column > .autostat-column-scroll-body,
                [data-testid="column"].autostat-resizable-card-column > .autostat-column-scroll-body {
                  max-height: none !important;
                  overflow-y: visible !important;
                  padding-right: 0 !important;
                }
              }
            `;
            doc.head.appendChild(style);
          }

          function isNarrowViewport() {
            return parentWindow.innerWidth < MIN_VIEWPORT_WIDTH;
          }

          function getMainRoot() {
            return (
              doc.querySelector('section.main') ||
              doc.querySelector('main[data-testid="stAppViewContainer"]') ||
              doc.querySelector('[data-testid="stAppViewContainer"] main') ||
              doc.body
            );
          }

          function isInSidebar(element) {
            return Boolean(element.closest('section[data-testid="stSidebar"]'));
          }

          function getColumn(element) {
            return element.closest(COLUMN_SELECTOR);
          }

          function getHorizontalBlock(column) {
            return column ? column.closest(HORIZONTAL_BLOCK_SELECTOR) : null;
          }

          function getColumns(block) {
            if (!block) {
              return [];
            }
            return Array.from(block.children).filter((child) => {
              return child.matches && child.matches(COLUMN_SELECTOR);
            });
          }

          function blockHasExpanderCards(block) {
            const columns = getColumns(block);
            if (columns.length < 2) {
              return false;
            }
            return columns.some((column) => column.querySelector(EXPANDER_SELECTOR));
          }

          function getScrollableColumnBody(column) {
            if (!column) {
              return null;
            }
            const directVerticalBlock = Array.from(column.children).find((child) => {
              return child.matches && child.matches('[data-testid="stVerticalBlock"]');
            });
            return (
              directVerticalBlock ||
              column.querySelector(':scope > [data-testid="stVerticalBlock"]') ||
              column.firstElementChild
            );
          }

          function updateColumnScrollHeights() {
            if (isNarrowViewport()) {
              return;
            }

            const bottomGap = 24;
            const minHeight = 240;
            Array.from(doc.querySelectorAll(".autostat-column-scroll-body")).forEach((element) => {
              const rect = element.getBoundingClientRect();
              const available = Math.max(minHeight, parentWindow.innerHeight - rect.top - bottomGap);
              element.style.setProperty("--autostat-column-scroll-max-height", `${available}px`);
            });
          }

          function containColumnWheel(event) {
            if (isNarrowViewport()) {
              return;
            }

            const scrollBody = event.target?.closest?.(".autostat-column-scroll-body");
            if (!scrollBody) {
              return;
            }

            const deltaY = event.deltaY || 0;
            if (!deltaY) {
              return;
            }

            const canScrollUp = scrollBody.scrollTop > 0;
            const canScrollDown =
              scrollBody.scrollTop + scrollBody.clientHeight < scrollBody.scrollHeight - 1;

            if ((deltaY < 0 && canScrollUp) || (deltaY > 0 && canScrollDown)) {
              event.stopPropagation();
            }
          }

          function getExpanderTitle(expander) {
            const header =
              expander.querySelector("details > summary") ||
              expander.querySelector('[data-testid="stExpanderToggleIcon"]')?.parentElement ||
              expander.firstElementChild;
            return (header?.textContent || "").replace(/\\s+/g, " ").trim().slice(0, 60);
          }

          function makeBlockKey(block, columns) {
            const pageKey = parentWindow.location.pathname || "main";
            const titleParts = columns.map((column, index) => {
              const titles = Array.from(column.querySelectorAll(EXPANDER_SELECTOR))
                .map(getExpanderTitle)
                .filter(Boolean)
                .join("+");
              return titles || `column-${index}`;
            });
            return `${STORAGE_PREFIX}${pageKey}:${titleParts.join("|")}`;
          }

          function readStoredPercents(key, expectedLength) {
            try {
              const stored = JSON.parse(parentWindow.localStorage.getItem(key) || "null");
              if (!Array.isArray(stored) || stored.length !== expectedLength) {
                return null;
              }
              const values = stored.map(Number);
              if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
                return null;
              }
              return values;
            } catch (_error) {
              return null;
            }
          }

          function writeStoredPercents(key, percents) {
            parentWindow.localStorage.setItem(key, JSON.stringify(percents.map((value) => {
              return Math.round(value * 1000) / 1000;
            })));
          }

          function normalizePercents(values) {
            const total = values.reduce((sum, value) => sum + value, 0);
            if (!total) {
              return values;
            }
            return values.map((value) => (value / total) * 100);
          }

          function getCurrentPercents(columns) {
            const widths = columns.map((column) => column.getBoundingClientRect().width);
            return normalizePercents(widths);
          }

          function applyColumnPercents(columns, percents) {
            if (!columns.length || isNarrowViewport()) {
              columns.forEach((column) => {
                column.style.removeProperty("flex");
                column.style.removeProperty("width");
                column.style.removeProperty("max-width");
                column.style.removeProperty("min-width");
              });
              return;
            }

            const normalized = normalizePercents(percents);
            columns.forEach((column, index) => {
              const weight = Math.max(normalized[index] || 0, 1);
              column.style.setProperty("flex", `${weight} ${weight} 0px`, "important");
              column.style.setProperty("width", "auto", "important");
              column.style.setProperty("max-width", "none", "important");
              column.style.setProperty("min-width", `${MIN_COLUMN_WIDTH}px`, "important");
            });
          }

          function restoreBlockLayout(block) {
            if (!blockHasExpanderCards(block)) {
              return;
            }
            const columns = getColumns(block);
            if (columns.length < 2) {
              return;
            }
            const key = makeBlockKey(block, columns);
            const stored = readStoredPercents(key, columns.length);
            if (stored) {
              applyColumnPercents(columns, stored);
            } else if (isNarrowViewport()) {
              applyColumnPercents(columns, []);
            }
          }

          function beginDrag(event, column) {
            if (isNarrowViewport()) {
              return;
            }

            const block = getHorizontalBlock(column);
            const columns = getColumns(block);
            const index = columns.indexOf(column);
            const partnerIndex = index + 1;

            if (
              !block ||
              !blockHasExpanderCards(block) ||
              columns.length < 2 ||
              index < 0 ||
              partnerIndex >= columns.length
            ) {
              return;
            }

            const widths = columns.map((item) => item.getBoundingClientRect().width);
            const pairWidth = widths[index] + widths[partnerIndex];
            if (pairWidth <= MIN_COLUMN_WIDTH * 2) {
              return;
            }

            activeDrag = {
              block,
              columns,
              index,
              partnerIndex,
              startX: event.clientX,
              startWidths: widths,
              storageKey: makeBlockKey(block, columns),
              handle: event.currentTarget,
            };

            activeDrag.handle.classList.add("autostat-is-dragging");
            doc.body.classList.add("autostat-card-resizing");
            event.preventDefault();
            event.stopPropagation();
          }

          function updateDrag(event) {
            if (!activeDrag) {
              return;
            }

            const {
              columns,
              index,
              partnerIndex,
              startX,
              startWidths,
              storageKey,
            } = activeDrag;
            const pairWidth = startWidths[index] + startWidths[partnerIndex];
            const delta = event.clientX - startX;
            const nextWidths = [...startWidths];
            const minCurrent = Math.min(MIN_COLUMN_WIDTH, pairWidth / 2);
            const rawCurrent = startWidths[index] + delta;
            const current = Math.min(
              Math.max(rawCurrent, minCurrent),
              pairWidth - minCurrent
            );
            nextWidths[index] = current;
            nextWidths[partnerIndex] = pairWidth - current;

            const nextPercents = normalizePercents(nextWidths);
            applyColumnPercents(columns, nextPercents);
            writeStoredPercents(storageKey, nextPercents);
            event.preventDefault();
          }

          function endDrag() {
            if (!activeDrag) {
              return;
            }
            activeDrag.handle.classList.remove("autostat-is-dragging");
            doc.body.classList.remove("autostat-card-resizing");
            activeDrag = null;
          }

          function enhanceBlock(block) {
            if (isInSidebar(block) || !blockHasExpanderCards(block)) {
              return;
            }

            const columns = getColumns(block);
            if (columns.length < 2) {
              return;
            }

            block.classList.add("autostat-resizable-card-row");

            columns.forEach((column, index) => {
              column.classList.add("autostat-resizable-card-column");
              const scrollBody = getScrollableColumnBody(column);
              if (scrollBody) {
                scrollBody.classList.add("autostat-column-scroll-body");
              }

              if (index >= columns.length - 1 || column.dataset.autostatResizable === "true") {
                return;
              }

              column.dataset.autostatResizable = "true";
              const handle = doc.createElement("div");
              handle.className = "autostat-resize-handle";
              handle.setAttribute("role", "separator");
              handle.setAttribute("aria-orientation", "vertical");
              handle.setAttribute("title", "拖动调整左右卡片宽度");
              handle.addEventListener("pointerdown", (event) => beginDrag(event, column));
              column.appendChild(handle);
            });
            restoreBlockLayout(block);
            updateColumnScrollHeights();
          }

          function enhanceAll() {
            injectStyles();
            const main = getMainRoot();
            const blocks = Array.from(main.querySelectorAll(HORIZONTAL_BLOCK_SELECTOR));
            blocks.forEach(enhanceBlock);
            updateColumnScrollHeights();
          }

          doc.addEventListener("pointermove", updateDrag, true);
          doc.addEventListener("pointerup", endDrag, true);
          doc.addEventListener("pointercancel", endDrag, true);
          doc.addEventListener("wheel", containColumnWheel, { capture: true, passive: true });
          doc.addEventListener("scroll", () => {
            parentWindow.requestAnimationFrame(updateColumnScrollHeights);
          }, true);
          parentWindow.addEventListener("resize", () => {
            if (isNarrowViewport()) {
              Array.from(doc.querySelectorAll(HORIZONTAL_BLOCK_SELECTOR)).forEach((block) => {
                applyColumnPercents(getColumns(block), []);
              });
            } else {
              enhanceAll();
              Array.from(doc.querySelectorAll(HORIZONTAL_BLOCK_SELECTOR)).forEach(restoreBlockLayout);
              updateColumnScrollHeights();
            }
          });

          if (doc.body && parentWindow.MutationObserver) {
            const observer = new parentWindow.MutationObserver(() => {
              collapseHostElement();
              parentWindow.requestAnimationFrame(enhanceAll);
            });
            observer.observe(doc.body, { childList: true, subtree: true });
          }

          collapseHostElement();
          enhanceAll();
        })();
        </script>
        """,
        height=0,
        width=0,
    )
