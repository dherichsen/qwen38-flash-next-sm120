#!/usr/bin/env python3
"""Render the checked-in technical field guide. Run with markdown-it-py==4.0.0."""
from pathlib import Path
import html,re
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'docs/current-setup.md').read_text()
# The web edition has its own short title/introduction; procedural copy is shared.
source=source[source.index('## Is the original'):]
md=MarkdownIt('commonmark',{'html':False}).enable('table');tokens=md.parse(source)
nav=[]
for i,t in enumerate(tokens):
 if t.type=='heading_open' and t.tag=='h2':
  title=tokens[i+1].content;slug=re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-');t.attrSet('id',slug);nav.append((slug,title))
 for child in t.children or []:
  if child.type=='link_open':
   href=child.attrGet('href') or ''
   if href.startswith('../'):child.attrSet('href','https://github.com/dherichsen/qwen38-flash-next-sm120/blob/main/'+href[3:])
body=md.renderer.render(tokens,md.options,{})
body=body.replace('<table>','<div class="table-wrap" role="region" aria-label="Configuration comparison" tabindex="0"><table>').replace('</table>','</table></div>')
links=''.join(f'<a href="#{slug}"><span>{i:02d}</span>{html.escape(title)}</a>' for i,(slug,title) in enumerate(nav,1))
page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.8 on one Blackwell GPU — 200K field guide</title>
<meta name="description" content="Reproduce a local Qwen3.8 Flash-Next NVFP4 deployment: RTX PRO 6000 Blackwell, 200K context, SGLang, verified compatibility patches and practical installation steps.">
<link rel="stylesheet" href="assets/guide.css"><script src="assets/guide.js" defer></script></head>
<body><a class="skip" href="#guide">Skip to guide</a>
<header class="top"><a href="./" class="wordmark">LOCAL COMPUTE / FIELD NOTES</a><a href="https://github.com/dherichsen/qwen38-flash-next-sm120">Source on GitHub ↗</a></header>
<div class="layout"><aside><p class="eyebrow">FIELD GUIDE 01</p><nav aria-label="Contents">'''+links+'''</nav><div class="edition">Recorded 11 September 2026<br>Linux · SM120 · single GPU</div></aside>
<main id="guide"><header class="hero"><p class="eyebrow">QWEN3.8 FLASH-NEXT / NVIDIA NVFP4</p><h1>A local model.<br>A reproducible setup.</h1><p class="lede">The 200K configuration running on one RTX PRO 6000 Blackwell. From the original source tree to your first verified response.</p><a class="start" href="#1-prepare-the-host-and-download-the-checkpoint">Start the installation <span aria-hidden="true">↓</span></a><a class="secondary" href="current-setup.md">Read as Markdown ↗</a>
<div class="architecture" aria-label="Request path"><div><small>YOUR CLIENT</small><strong>Chat + tools</strong><span>Medium thinking · one request</span></div><b aria-hidden="true">→</b><div><small>LOCAL RUNTIME</small><strong>SGLang / 200K</strong><span>FP8 KV · NEXTN decoding</span></div><b aria-hidden="true">→</b><div><small>HARDWARE</small><strong>96 GB Blackwell</strong><span>Embeddings offloaded to RAM</span></div></div>
<p class="boundary">A documented working setup, with source and configuration checks. Run acceptance on your own hardware before relying on it.</p></header>
<article>'''+body+'''</article><footer><p>Maintained by <a href="https://github.com/dherichsen">dherichsen</a>. Deployment code: Apache-2.0. Model terms are separate.</p><a href="#guide">Back to top ↑</a></footer></main></div><div id="copy-status" role="status" aria-live="polite"></div></body></html>
'''
(ROOT/'docs/index.html').write_text(page)
(ROOT/'docs/.nojekyll').touch()
print('Rendered docs/index.html')
