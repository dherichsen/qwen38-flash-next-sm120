"""Visual and interaction checks for the static guide; accepts a public base URL."""
import json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:18642/'
out=Path('evidence/guide');out.mkdir(parents=True,exist_ok=True);errors=[]
with sync_playwright() as p:
 browser=p.firefox.launch(headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1000})
 page.on('pageerror',lambda e:errors.append(str(e)))
 response=page.goto(base,wait_until='networkidle');assert response.status==200
 expect(page.get_by_role('heading',name='A local model. A reproducible setup.')).to_be_visible()
 assert page.locator('article h2').count()==8
 assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
 for link in page.locator('a[href^="#"]').all():
  assert page.evaluate('(id)=>document.getElementById(id)!==null',link.get_attribute('href')[1:])
 assert page.request.get(base+'assets/PublicSans.ttf').status==200
 assert page.request.get(base+'current-build-verification.json').json()['differences']=={}
 page.screenshot(path=str(out/'desktop.png'),full_page=False)
 page.get_by_role('link',name='Start the installation').click()
 expect(page.get_by_role('heading',name='1. Prepare the host and download the checkpoint')).to_be_visible()
 page.get_by_role('button',name='Copy command block 1',exact=True).click()
 expect(page.get_by_role('status')).to_contain_text('Commands copied')
 page.screenshot(path=str(out/'installation.png'))
 page.set_viewport_size({'width':390,'height':844});page.goto(base,wait_until='networkidle')
 assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
 page.screenshot(path=str(out/'mobile.png'),full_page=False)
 page.get_by_role('link',name='Start the installation').click()
 expect(page.get_by_role('heading',name='1. Prepare the host and download the checkpoint')).to_be_visible()
 assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
 page.screenshot(path=str(out/'mobile-installation.png'))
 assert not errors,errors
 browser.close()
receipt={'passed':True,'url':base,'browser':'Firefox','viewport_widths':[1440,390],'copy_button':True,'navigation':True,'no_horizontal_page_overflow':True,'page_errors':errors}
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
