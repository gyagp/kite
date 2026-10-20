import { chromium } from 'playwright';
import fs from 'node:fs/promises';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
  const svg = await fs.readFile('public/kite.svg', 'utf8');
  await page.setContent(
    `<style>body{margin:0;background:transparent}svg{width:256px;height:256px}</style>${svg}`,
  );
  const png = await page.screenshot({ omitBackground: true });
  await fs.writeFile('public/kite.png', png);
  const header = Buffer.alloc(22);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(1, 4);
  header.writeUInt16LE(1, 10);
  header.writeUInt16LE(32, 12);
  header.writeUInt32LE(png.length, 14);
  header.writeUInt32LE(22, 18);
  await fs.mkdir('build', { recursive: true });
  await fs.writeFile('build/kite.ico', Buffer.concat([header, png]));
} finally {
  await browser.close();
}
