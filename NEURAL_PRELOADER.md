# Neural Preloader — what changed vs drishti-34-fixed4

Only 3 files touched. No backend code, no page content changed.

## Added
- `app/static/neural-preloader.js` — one self-injecting asset (styles, font and
  logic inside it; Space Grotesk embedded, logo taken from `/static/logo.png`).

## Edited (one line each, before `</body>`)
- `app/static/welcome.html` — `<script src="/static/neural-preloader.js" data-mode="preloader" data-alpha="0.5"></script>`
- `app/static/app.html` — `<script src="/static/neural-preloader.js" data-mode="ambient" data-alpha="0.4"></script>`

The lamp sign-in page (`index.html`) is deliberately NOT tagged — it is locked
and its scene is opaque, so a canvas would either hide behind it or haze over it.

## Behaviour
- Entry page (`/`): a dense 3D neuron field in Drishti's theme (deep green +
  gold) fills the screen. DRISHTI letters scatter, tumble in 3D and lock one by
  one while the counter and bar climb to 100%.
- At 100%: a fire-wave sweeps the neuron network, the letters fade, and the
  Drishti logo zooms in to centre stage, holds a beat, then zooms through
  into the real welcome page.
- After the preloader, the neuron field stays as a soft ambient network behind
  the welcome page. The chat page (`/app`) gets the same ambient network.
- Shows once per browser session. Open the site with `?preloader=1` added to
  the URL to see it again. Tap/click during the second half skips to the site.
- Mouse steers the network on every page; clicking fires the nearest neurons.
- `prefers-reduced-motion` shortens the whole sequence.
- There is no stand-in/preview home, no Replay button and no "edition" pill —
  the preloader opens straight into the real app.

## Remove it
Delete the two `<script>` lines above and `app/static/neural-preloader.js`.
Nothing else references it.

## Verified
Full app run locally: preloader frames, logo zoom, welcome reveal, reload
(no replay), Continue -> /app, mobile entry — all checked in a real browser,
0 new console errors (the only failed request was the app's own Google Fonts
link through the test sandbox proxy). Backend suite: 31/31 pytest passed,
same as the base build.
