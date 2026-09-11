'use strict';
document.querySelectorAll('pre').forEach((pre,index)=>{
 const button=document.createElement('button');button.type='button';button.textContent='Copy';button.setAttribute('aria-label',`Copy command block ${index+1}`);
 button.addEventListener('click',async()=>{const status=document.querySelector('#copy-status');try{await navigator.clipboard.writeText(pre.querySelector('code').textContent);status.textContent='Commands copied';button.textContent='Copied';}catch{status.textContent='Copy unavailable. Select the commands and copy manually.';}setTimeout(()=>{status.textContent='';button.textContent='Copy';},2500);});pre.prepend(button);
});
const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.removeAttribute('aria-current'));document.querySelector(`nav a[href="#${entry.target.id}"]`)?.setAttribute('aria-current','location');}}},{rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('article h2').forEach(h=>observer.observe(h));
