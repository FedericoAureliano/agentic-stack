# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "markdown>=3.6",
#   "pygments>=2.18",
#   "tokenizers>=0.20",
# ]
# ///
"""Build docs/index.html + docs/main.css from index.md.

Run with `uv run build.py` every time index.md changes.
"""

import html
import re
import shutil
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, Tag
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.md"
FAVICON = ROOT / "favicon.svg"
DOCS = ROOT / "docs"

QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"

# Fenced blocks tagged ```qwen3``` are pulled out before the markdown
# conversion and replaced with an HTML-comment placeholder (which
# python-markdown passes through untouched), then rendered separately as a
# tokenizer visualization and spliced back in afterward.
QWEN3_FENCE_RE = re.compile(r"^```qwen3\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)

# Fenced blocks tagged ```jinja``` get the same pull-out/placeholder treatment
# as ```qwen3``` blocks above, but only so the resulting <pre> can carry an
# extra "jinja-code" class (see .jinja-code in main.css) for a smaller font
# size — the jinja snippet runs long and reads better a notch smaller than
# the other code blocks on the page.
JINJA_FENCE_RE = re.compile(r"^```jinja\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    body = text[end + 4 :].lstrip("\n")
    return meta, body


def extract_qwen3_blocks(md_text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def replace(match: re.Match) -> str:
        blocks.append(match.group(1))
        return f"<!--QWEN3_BLOCK_{len(blocks) - 1}-->"

    return QWEN3_FENCE_RE.sub(replace, md_text), blocks


def extract_jinja_blocks(md_text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def replace(match: re.Match) -> str:
        blocks.append(match.group(1))
        return f"<!--JINJA_BLOCK_{len(blocks) - 1}-->"

    return JINJA_FENCE_RE.sub(replace, md_text), blocks


def render_jinja_block(raw_text: str) -> str:
    lexer = get_lexer_by_name("jinja")
    formatter = HtmlFormatter(cssclass="codehilite jinja-code", wrapcode=True)
    return highlight(raw_text, lexer, formatter).rstrip("\n")


# Raw <svg>...</svg> blocks (e.g. the loop-anim diagram) are pulled out
# before markdown conversion and spliced back in verbatim as the very last
# step, after wrap_bug_banners()/make_foldable() have run. Both of those
# reparse body_html with BeautifulSoup's html.parser, which — unlike a real
# browser's HTML parser — does not special-case SVG's camelCase attributes
# (viewBox, markerWidth, refX, ...) and lowercases them, so this is the only
# way to keep them intact in the emitted HTML.
SVG_RE = re.compile(r"^<svg\b.*?</svg>[ \t]*$", re.DOTALL | re.MULTILINE)


def extract_svg_blocks(md_text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def replace(match: re.Match) -> str:
        blocks.append(match.group(0))
        return f"<!--SVG_BLOCK_{len(blocks) - 1}-->"

    return SVG_RE.sub(replace, md_text), blocks


# Any fenced code block, generic (not just ```qwen3```), so math extraction
# below can skip over code content and leave literal "$" signs in code alone.
ANY_FENCE_RE = re.compile(r"^```.*?\n.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)

# $$...$$ display math, spanning possibly multiple lines.
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# $...$ inline math. Follows Pandoc's heuristic for telling math apart from
# literal currency: the char right after the opening $ and right before the
# closing $ must not be whitespace, and the closing $ must not be followed by
# a digit. That's what keeps "$10 / MTok ... and $50 / MTok" from being
# misread as math while still catching "$\Sigma$."
INLINE_MATH_RE = re.compile(r"\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\d)")


def extract_math_blocks(md_text: str) -> tuple[str, list[tuple[str, bool]]]:
    """Pull $...$ and $$...$$ spans out of md_text (skipping fenced code
    blocks) and replace them with HTML-comment placeholders that
    python-markdown passes through untouched. Returns the rewritten text and
    the list of (tex, is_display) pairs, indexed by placeholder number, to be
    spliced back in as MathJax delimiters after the markdown conversion.
    """
    blocks: list[tuple[str, bool]] = []

    def extract_from_segment(segment: str) -> str:
        def replace_display(match: re.Match) -> str:
            blocks.append((match.group(1), True))
            return f"<!--MATH_BLOCK_{len(blocks) - 1}-->"

        segment = DISPLAY_MATH_RE.sub(replace_display, segment)

        def replace_inline(match: re.Match) -> str:
            blocks.append((match.group(1), False))
            return f"<!--MATH_BLOCK_{len(blocks) - 1}-->"

        return INLINE_MATH_RE.sub(replace_inline, segment)

    out = []
    pos = 0
    for fence in ANY_FENCE_RE.finditer(md_text):
        out.append(extract_from_segment(md_text[pos : fence.start()]))
        out.append(fence.group(0))
        pos = fence.end()
    out.append(extract_from_segment(md_text[pos:]))

    return "".join(out), blocks


def wrap_bug_banners(body_html: str) -> str:
    """Each <!-- ADD BUG BANNER HERE --> marker in index.md turns into a
    standalone `<div class="bug-banner">` (see the emoji span inline in the
    markdown source) sitting right before the paragraph it flags. Merge the
    two into one `<div class="bug-row">` so the emoji can sit to the left of
    that paragraph and be vertically centered against it with flexbox (see
    .bug-row in main.css) instead of needing manual position math.
    """
    soup = BeautifulSoup(body_html, "html.parser")
    for banner in soup.find_all("div", class_="bug-banner"):
        emoji = banner.find("span", class_="bug-emoji")
        para = banner.find_next_sibling(True)
        if emoji is None or para is None:
            continue
        row = soup.new_tag("div", attrs={"class": "bug-row"})
        banner.replace_with(row)
        row.append(emoji.extract())
        row.append(para.extract())
    return str(soup)


HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def make_foldable(body_html: str) -> str:
    """Wrap each heading and the content under it in a <section class="fold">,
    nested by heading level, and give the heading a clickable arrow that
    toggles a "collapsed" class on its section (see the .fold-arrow /
    .fold.collapsed rules in main.css and the toggle script in render_html).
    Also gives each heading a slugified id (deduped on collision) so other
    parts of the page — e.g. the loop-anim diagram's arrow labels — can link
    straight to it with a "#slug" href.
    """
    soup = BeautifulSoup(body_html, "html.parser")
    top_level = list(soup.contents)
    for node in top_level:
        node.extract()

    seen_ids: dict[str, int] = {}

    container = soup.new_tag("div")
    stack = [(1, container)]

    for node in top_level:
        if isinstance(node, Tag) and node.name in HEADING_TAGS:
            level = int(node.name[1])
            while stack[-1][0] >= level:
                stack.pop()
            slug = slugify(node.get_text()) or "section"
            seen_ids[slug] = seen_ids.get(slug, 0) + 1
            node["id"] = slug if seen_ids[slug] == 1 else f"{slug}-{seen_ids[slug]}"
            arrow = soup.new_tag(
                "span",
                attrs={
                    "class": "fold-arrow",
                    "role": "button",
                    "tabindex": "0",
                    "aria-expanded": "true",
                },
            )
            arrow.string = ">" * (level - 1)
            node.insert(0, arrow)
            section = soup.new_tag("section", attrs={"class": "fold"})
            section.append(node)
            stack[-1][1].append(section)
            stack.append((level, section))
        else:
            stack[-1][1].append(node)

    return "".join(str(child) for child in container.contents)


# Chat-template structural tokens get grouped so related open/close tags
# always share one color (kept in sync with the .tk-* rules in main.css).
QWEN3_IM_TOKENS = {"<|im_start|>", "<|im_end|>"}
QWEN3_THINK_TOKENS = {"<think>", "</think>"}
QWEN3_TOOL_TOKENS = {"<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"}


def qwen3_token_class(piece: str, is_added: bool) -> str:
    if not is_added:
        return "tk"
    if piece in QWEN3_IM_TOKENS:
        return "tk tk-im"
    if piece in QWEN3_THINK_TOKENS:
        return "tk tk-think"
    if piece in QWEN3_TOOL_TOKENS:
        return "tk tk-tool"
    return "tk tk-special"


def render_qwen3_block(raw_text: str, tokenizer: Tokenizer) -> str:
    added = tokenizer.get_added_tokens_decoder()
    encoding = tokenizer.encode(raw_text, add_special_tokens=False)

    spans = []
    for token_id, (start, end) in zip(encoding.ids, encoding.offsets):
        if end <= start:
            continue
        raw_piece = raw_text[start:end]
        piece = html.escape(raw_piece)
        css_class = qwen3_token_class(raw_piece, token_id in added)
        spans.append(f'<span class="{css_class}">{piece}</span>')

    return '<pre class="codehilite qwen3-tokens"><code>' + "".join(spans) + "</code></pre>"


def render_html(meta: dict[str, str], body_html: str) -> str:
    title = html.escape(meta.get("title", "Untitled"))
    author = html.escape(meta.get("author", ""))
    webpage = meta.get("webpage", "")
    webpage_href = webpage if webpage.startswith("http") else f"https://{webpage}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="author" content="{author}">
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="main.css">
<script data-goatcounter="https://federicoaureliano.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
</head>
<body>
<div id="scroller"></div>
<main>
<div class="header-row">
<h1>{title}</h1>
<p class="meta"><a href="{webpage_href}">{author}</a></p>
</div>
{body_html}
</main>
<script>
window.onscroll = function () {{ updateScroller(); }};
function updateScroller() {{
  var winScroll = document.body.scrollTop || document.documentElement.scrollTop;
  var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  var scrolled = height > 0 ? (winScroll / height) * 100 : 0;
  scrolled = Math.min(100, Math.max(0, scrolled));
  document.getElementById('scroller').style.setProperty('--scroller-width', (100 - scrolled) + '%');
}}

document.querySelectorAll('.fold-arrow').forEach(function (arrow) {{
  function toggle() {{
    var section = arrow.closest('.fold');
    var collapsed = section.classList.toggle('collapsed');
    arrow.setAttribute('aria-expanded', String(!collapsed));
  }}
  arrow.addEventListener('click', toggle);
  arrow.addEventListener('keydown', function (event) {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      toggle();
    }}
  }});
}});
</script>
</body>
</html>
"""


MAIN_CSS = """:root {
  --font-sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;

  /* Obsidian light theme: Foundations/Colors (subset) */
  --color-base-00: #ffffff;
  --color-base-20: #f6f6f6;
  --color-base-30: #e4e4e4;
  --color-base-60: #707070;
  --color-base-70: #5c5c5c;
  --color-base-100: #222222;
  --color-accent: hsl(258, 88%, 66%);
  --color-accent-hover: hsl(258, 88%, 58%);

  /* Okabe & Ito colorblind-safe palette, same values as the source site */
  --color-orange: #e69f00;
  --color-skyblue: #56b4e9;
  --color-green: #009e73;
  --color-blue: #0072b2;
  --color-vermillion: #d55e00;
  --color-purple: #cc79a7;
  --color-yellow: #7a6a00;
  --color-black: #000000;
  --color-teal: #106b6b;
  --color-indigo: #5b3fa0;
  --color-lime: #218807;
  --color-magenta: #9329a3;

  --bg: var(--color-base-00);
  --fg: var(--color-base-100);
  --muted: var(--color-base-60);
  --border: var(--color-base-30);
  --code-bg: var(--color-base-20);

  /* A separate, colorblind-safe palette for code/token highlighting only.
     Derived from the Okabe & Ito colors above, but several are darkened
     because the originals (orange/skyblue/green/vermillion/purple/lime)
     fall below AA contrast (4.5:1) at normal text size against --code-bg. */
  --syntax-muted: #5c5c5c;
  --syntax-keyword: #0072b2;
  --syntax-builtin: #106b6b;
  --syntax-function: #aa4b00;
  --syntax-decorator: #a53f77;
  --syntax-type: #1674a9;
  --syntax-string: #007e5c;
  --syntax-number: #5b3fa0;
  --syntax-preproc: #7a6a00;
  --syntax-escape: #1e7a06;
  --syntax-prompt: var(--color-accent-hover);

  --font-text-size: 15px;
  --line-height-normal: 1.5;
  --line-height-tight: 1.3;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-sans);
  font-size: var(--font-text-size);
  line-height: var(--line-height-normal);
  -webkit-text-size-adjust: 100%;
}

main {
  max-width: 40rem;
  margin: 0 auto;
  padding: 3rem 1.25rem 4rem;
}

/* A small "known rough edge" marker. build.py's wrap_bug_banners() merges
   each <!-- ADD BUG BANNER HERE --> placeholder with the paragraph right
   after it into one row, so the emoji can float out into the left margin,
   vertically centered against that paragraph. Below 900px there's no margin
   to float into, so it drops inline to the left instead. */
.bug-row {
  position: relative;
}
.bug-emoji {
  position: absolute;
  right: 100%;
  top: 50%;
  transform: translate(-2rem, -50%);
  font-size: 1.4rem;
  line-height: 1;
}

@media (max-width: 900px) {
  .bug-row {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin: 1rem 0;
  }
  .bug-row p {
    margin: 0;
  }
  .bug-emoji {
    position: static;
    transform: none;
    flex: none;
  }
}

/* End-to-end loop animation (the fib(33) example): a sequence diagram
   between the User/SDK/Inference Engine/LLM lifelines. The 8 <g
   class="evt evt-N"> arrow groups (N = 0..7, one per stage) fade in and
   then STAY, so the diagram progressively builds itself up over one 13s lap
   — a 1s blank beat with no arrows, then 8 stages at 1.5s each. Unlike a
   shared-keyframes-plus-delay trick, each evt-N has its OWN @keyframes
   (evt-reveal-0..7) that reveals it at a fixed point on the SAME 13s
   timeline and holds it until 100% — since none of them use
   animation-delay, they all hit that 100%->0% reset at the exact same
   instant, so every arrow clears together (leaving that blank beat) right
   before the whole diagram replays from the beginning, instead of each one
   clearing on its own staggered schedule. */
.loop-anim {
  margin: 2rem 0;
}
.loop-svg-wrap {
  overflow-x: auto;
}
.loop-svg {
  display: block;
  width: 100%;
  max-width: 680px;
  min-width: 560px;
  height: auto;
  margin: 0 auto;
}
.lane-box {
  fill: none;
  stroke: var(--border);
}
.lane-label {
  fill: var(--fg);
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: 12px;
}
.lifeline {
  stroke: var(--border);
  stroke-width: 2;
  stroke-dasharray: 5 5;
}
.loop-arrowhead-fill {
  fill: none;
  stroke: var(--muted);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.evt {
  opacity: 0;
}
.evt-0 { animation: evt-reveal-0 13s infinite; }
.evt-1 { animation: evt-reveal-1 13s infinite; }
.evt-2 { animation: evt-reveal-2 13s infinite; }
.evt-3 { animation: evt-reveal-3 13s infinite; }
.evt-4 { animation: evt-reveal-4 13s infinite; }
.evt-5 { animation: evt-reveal-5 13s infinite; }
.evt-6 { animation: evt-reveal-6 13s infinite; }
.evt-7 { animation: evt-reveal-7 13s infinite; }
.evt-arrow {
  stroke: var(--muted);
  stroke-width: 2.6;
  stroke-linecap: round;
}
/* fill/fill-opacity/text-decoration on SVG elements turned out to render
   inconsistently across browsers (fine in Safari, an opaque black bar in
   Firefox, no matter how the color was spelled) — so each arrow label is a
   real HTML <a> in a <foreignObject> instead of SVG <text>, styled by the
   exact same a { } rule as the rest of the document (below), which every
   browser already renders identically and correctly. */
.evt-fo {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.evt-fo-link {
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: nowrap;
}

@keyframes evt-reveal-0 {
  0%, 7.6% { opacity: 0; }
  7.692% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-1 {
  0%, 19.1% { opacity: 0; }
  19.231% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-2 {
  0%, 30.7% { opacity: 0; }
  30.769% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-3 {
  0%, 42.2% { opacity: 0; }
  42.308% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-4 {
  0%, 53.7% { opacity: 0; }
  53.846% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-5 {
  0%, 65.3% { opacity: 0; }
  65.385% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-6 {
  0%, 76.8% { opacity: 0; }
  76.923% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes evt-reveal-7 {
  0%, 88.3% { opacity: 0; }
  88.462% { opacity: 1; }
  99.9% { opacity: 1; }
  100% { opacity: 0; }
}

@media (prefers-reduced-motion: reduce) {
  .evt {
    animation: none !important;
    opacity: 1;
  }
}

/* Reading-progress bar, updated by the scroll listener in render_html(). */
#scroller {
  height: 0.4rem;
  width: 100%;
  position: fixed;
  left: 0;
  top: 0;
  background-color: var(--border);
  --scroller-width: 100%;
  z-index: 10;
}

#scroller::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  width: var(--scroller-width);
  background-color: var(--color-accent);
  border-top-right-radius: 4px;
  border-bottom-right-radius: 4px;
  transition: width 0.1s ease-out;
}

.header-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 4px;
  margin: 0 0 1.5rem;
}

h1, h2, h3, h4, h5 {
  font-family: var(--font-sans);
  font-weight: 600;
  line-height: var(--line-height-tight);
  color: var(--fg);
}

h1 { font-size: 20px; margin: 0; }
h2 { font-size: 17px; margin: 2rem 0 0.5rem; }
h3 { font-size: 15px; margin: 1.5rem 0 0.25rem; }
h4 { font-size: 14.5px; margin: 1.25rem 0 0.25rem; }
h5 { font-size: 14px; margin: 1rem 0 0.25rem; }

.fold-arrow {
  display: inline-block;
  margin-right: 0.35em;
  color: var(--color-accent);
  cursor: pointer;
  user-select: none;
}
.fold-arrow:hover { color: var(--color-accent-hover); }
.fold-arrow:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
  border-radius: 2px;
}

.fold.collapsed > :not(:first-child) { display: none; }
.fold.collapsed > :first-child::after {
  content: " \\2026";
  color: var(--muted);
  font-weight: 400;
}

.meta {
  margin: 0;
  min-width: 0;
  color: var(--muted);
  text-align: right;
}

p, ul, ol { margin: 1rem 0; }
/* Adjacent margins collapse to the larger value, so both sides of the
   paragraph/list junction need shrinking, not just the list's top margin. */
p + ul, p + ol { margin-top: 0.35rem; }
p:has(+ ul), p:has(+ ol) { margin-bottom: 0.35rem; }

a {
  color: var(--fg);
  text-decoration-line: underline;
  text-decoration-color: color-mix(in srgb, var(--color-orange) 38%, transparent);
  text-decoration-thickness: 0.6em;
  text-underline-offset: -0.42em;
  text-decoration-skip-ink: none;
}
a:hover { text-decoration-color: var(--color-orange); }

code {
  font-family: var(--font-mono);
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 4px;
}

pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  font-size: 10pt;
  line-height: 125%;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

pre code {
  background: none;
  padding: 0;
}

/* The jinja snippet is long and annotated, so it gets a smaller font than
   other code blocks (see render_jinja_block / JINJA_FENCE_RE in build.py). */
.jinja-code {
  font-size: 8pt;
}

blockquote {
  margin: 1rem 0;
  padding: 4px 1rem;
  border-left: 3px solid var(--border);
  color: var(--muted);
}

hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2.25rem 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
}

th, td {
  text-align: left;
  padding: 8px 1rem;
  vertical-align: middle;
}

thead th {
  font-size: 13px;
  font-weight: 400;
  border-bottom: 2px solid var(--fg);
}

img { max-width: 100%; }

del {
  color: var(--muted);
  text-decoration-thickness: 0.15em;
  font-weight: 600;
}

@media (max-width: 480px) {
  main { padding: 2rem 1rem 3rem; }
  .header-row { flex-wrap: wrap; }
  .meta { text-align: left; }
}

/* Pygments (codehilite) syntax colors: the --syntax-* palette above, kept
   separate from the page's --color-* theme palette. */
.codehilite .c, .codehilite .c1, .codehilite .cm, .codehilite .ch, .codehilite .cs { color: var(--syntax-muted); font-style: italic; }
.codehilite .cp { color: var(--syntax-preproc); }
.codehilite .k, .codehilite .kn, .codehilite .kd, .codehilite .kc, .codehilite .kr, .codehilite .kp { color: var(--syntax-keyword); font-weight: 600; }
.codehilite .kt { color: var(--syntax-type); }
.codehilite .nb, .codehilite .bp { color: var(--syntax-builtin); }
.codehilite .nf, .codehilite .fm { color: var(--syntax-function); }
.codehilite .nc { color: var(--syntax-function); font-weight: 600; }
.codehilite .nd { color: var(--syntax-decorator); }
.codehilite .ne { color: var(--syntax-function); font-weight: 600; }
.codehilite .s, .codehilite .s1, .codehilite .s2, .codehilite .sd, .codehilite .sa, .codehilite .sb, .codehilite .dl, .codehilite .sh, .codehilite .sx { color: var(--syntax-string); }
.codehilite .se, .codehilite .si { color: var(--syntax-escape); }
.codehilite .m, .codehilite .mi, .codehilite .mf, .codehilite .mh, .codehilite .mo, .codehilite .il { color: var(--syntax-number); }
.codehilite .o, .codehilite .ow { color: var(--fg); }
.codehilite .gp { color: var(--syntax-prompt); font-weight: 600; }
.codehilite .go { color: var(--syntax-muted); }
.codehilite .gh, .codehilite .gu { color: var(--syntax-number); font-weight: 600; }
.codehilite .gd { color: var(--syntax-function); }
.codehilite .gi { color: var(--syntax-string); }
.codehilite .gr, .codehilite .err { color: var(--syntax-function); }
.codehilite .w { color: inherit; }

/* Qwen3 tokenizer visualization: each <span> is one real token from the
   Qwen3 tokenizer. Plain content tokens are unstyled text; structural
   chat-template tokens are single added tokens in the vocabulary (rather
   than ordinary BPE merges) and get a light touch of color, grouped by
   role: <|im_start|>/<|im_end|> turn markers, <think>/</think>, and the
   <tool_call>/<tool_response> pairs. */
.qwen3-tokens .tk-special { color: var(--color-accent-hover); }
.qwen3-tokens .tk-im { color: var(--color-accent-hover); font-weight: 700; }
.qwen3-tokens .tk-think { color: var(--syntax-builtin); }
.qwen3-tokens .tk-tool { color: var(--syntax-function); }
"""


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(source_text)

    # The title is already rendered in the header-row from frontmatter, so
    # drop a leading top-level heading to avoid showing it twice.
    body = re.sub(r"^#[ \t]+.*\n+", "", body, count=1)

    body, svg_blocks = extract_svg_blocks(body)
    body, qwen3_blocks = extract_qwen3_blocks(body)
    body, jinja_blocks = extract_jinja_blocks(body)
    body, math_blocks = extract_math_blocks(body)

    body_html = markdown.markdown(
        body,
        extensions=["fenced_code", "codehilite", "tables", "sane_lists", "smarty"],
        extension_configs={
            "codehilite": {"guess_lang": False},
            # Only convert dashes ("---" -> em dash, "--" -> en dash); leave
            # quotes and ellipses as plain ASCII.
            "smarty": {
                "smart_quotes": False,
                "smart_angled_quotes": False,
                "smart_ellipses": False,
            },
        },
    )

    if qwen3_blocks:
        tokenizer = Tokenizer.from_pretrained(QWEN3_TOKENIZER)
        for i, raw in enumerate(qwen3_blocks):
            rendered = render_qwen3_block(raw, tokenizer)
            body_html = body_html.replace(f"<!--QWEN3_BLOCK_{i}-->", rendered)

    for i, raw in enumerate(jinja_blocks):
        rendered = render_jinja_block(raw)
        body_html = body_html.replace(f"<!--JINJA_BLOCK_{i}-->", rendered)

    for i, (tex, is_display) in enumerate(math_blocks):
        tex = html.escape(tex)
        rendered = f"\\[{tex}\\]" if is_display else f"\\({tex}\\)"
        body_html = body_html.replace(f"<!--MATH_BLOCK_{i}-->", rendered)

    body_html = wrap_bug_banners(body_html)
    body_html = make_foldable(body_html)

    for i, raw in enumerate(svg_blocks):
        body_html = body_html.replace(f"<!--SVG_BLOCK_{i}-->", raw)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(render_html(meta, body_html), encoding="utf-8")
    (DOCS / "main.css").write_text(MAIN_CSS, encoding="utf-8")
    shutil.copyfile(FAVICON, DOCS / "favicon.svg")

    print(f"Built {DOCS / 'index.html'}, {DOCS / 'main.css'}, and {DOCS / 'favicon.svg'}")


if __name__ == "__main__":
    main()
