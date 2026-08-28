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
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, Tag
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.md"
DOCS = ROOT / "docs"

QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"

# Fenced blocks tagged ```qwen3``` are pulled out before the markdown
# conversion and replaced with an HTML-comment placeholder (which
# python-markdown passes through untouched), then rendered separately as a
# tokenizer visualization and spliced back in afterward.
QWEN3_FENCE_RE = re.compile(r"^```qwen3\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)

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


HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}


def make_foldable(body_html: str) -> str:
    """Wrap each heading and the content under it in a <section class="fold">,
    nested by heading level, and give the heading a clickable arrow that
    toggles a "collapsed" class on its section (see the .fold-arrow /
    .fold.collapsed rules in main.css and the toggle script in render_html).
    """
    soup = BeautifulSoup(body_html, "html.parser")
    top_level = list(soup.contents)
    for node in top_level:
        node.extract()

    container = soup.new_tag("div")
    stack = [(1, container)]

    for node in top_level:
        if isinstance(node, Tag) and node.name in HEADING_TAGS:
            level = int(node.name[1])
            while stack[-1][0] >= level:
                stack.pop()
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="main.css">
</head>
<body>
<main>
<div class="header-row">
<h1>{title}</h1>
<p class="meta"><a href="{webpage_href}">{author}</a></p>
</div>
{body_html}
</main>
<script>
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
h4 { font-size: 15px; margin: 1.25rem 0 0.25rem; }
h5 { font-size: 15px; margin: 1rem 0 0.25rem; }

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

    body, qwen3_blocks = extract_qwen3_blocks(body)

    body_html = markdown.markdown(
        body,
        extensions=["fenced_code", "codehilite", "tables", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False}},
    )

    if qwen3_blocks:
        tokenizer = Tokenizer.from_pretrained(QWEN3_TOKENIZER)
        for i, raw in enumerate(qwen3_blocks):
            rendered = render_qwen3_block(raw, tokenizer)
            body_html = body_html.replace(f"<!--QWEN3_BLOCK_{i}-->", rendered)

    body_html = make_foldable(body_html)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(render_html(meta, body_html), encoding="utf-8")
    (DOCS / "main.css").write_text(MAIN_CSS, encoding="utf-8")

    print(f"Built {DOCS / 'index.html'} and {DOCS / 'main.css'}")


if __name__ == "__main__":
    main()
