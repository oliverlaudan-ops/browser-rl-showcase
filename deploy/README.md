# Deployment: rl-showcase.future-pulse.de

Statische Single-Page-App (Vite-Build), serviert via nginx direkt aus
`/var/www/rl-showcase/`. Kein Container, kein Backend.

## Voraussetzungen

- DNS A-Record: `rl-showcase.future-pulse.de → 62.171.143.32` (VPS-IP)
  via Hostinger DNS-Panel
- nginx installiert (`apt install nginx`)
- certbot installiert (`apt install certbot python3-certbot-nginx`)
- Build-Artefakt: `dist/` aus `npm run build`

## Schritte

### 1. DNS setzen

Im Hostinger-Panel für `future-pulse.de`:
- Neuer A-Record: Host `rl-showcase`, Value `62.171.143.32`, TTL 300
- Warten bis propagiert (~5min mitunter, `host rl-showcase.future-pulse.de` checken)

### 2. Verzeichnis vorbereiten

```bash
sudo mkdir -p /var/www/rl-showcase
sudo chown -R $USER:$USER /var/www/rl-showcase
```

### 3. Build deployen

Lokal:
```bash
npm install
npm run build
rsync -avz --delete dist/ oliver@vmd191585:/var/www/rl-showcase/
```

Oder via `scp`, `gh release`, oder was auch immer der übliche Workflow ist.
Mit den anderen `future-pulse.de`-Subdomains gibt's schon eine Routine.

### 4. nginx-vhost installieren

```bash
sudo cp deploy/nginx-vhost.conf /etc/nginx/sites-available/rl-showcase.future-pulse.de
sudo ln -s /etc/nginx/sites-available/rl-showcase.future-pulse.de /etc/nginx/sites-enabled/
sudo nginx -t
```

### 5. SSL mit certbot

```bash
sudo certbot --nginx -d rl-showcase.future-pulse.de
```

`certbot` modifiziert die vhost-Datei automatisch (fügt `ssl_certificate`-Direktiven ein
und ergänzt den 80→301→443-Redirect-Block). Danach:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Health-Check

```bash
curl -I https://rl-showcase.future-pulse.de
# → 200 OK, Content-Type text/html

curl -I https://rl-showcase.future-pulse.de/assets/$(ls dist/assets | grep tfjs)
# → 200 OK, Content-Type application/javascript, Cache-Control: public, immutable
```

Browser öffnen, "Start training" klicken, CartPole sollte balancieren lernen.

## Updates deployen

```bash
npm run build
rsync -avz --delete dist/ oliver@vmd191585:/var/www/rl-showcase/
```

Der HTML-File hat `Cache-Control: no-cache`, also sieht man Änderungen sofort.
Asset-Chunks haben 1-Jahr-Cache, sind aber gehasht (`index-XXX.js`) — neuer
Build = neuer Hash = Browser holt neue Version.

## Troubleshooting

**404 auf `/assets/tfjs-*.js`** — wahrscheinlich nginx hat die Datei nicht
gefunden. Check `ls /var/www/rl-showcase/assets/`.

**Mixed-Content-Warnings in der Browser-Console** — sollte nicht passieren,
weil alles unter HTTPS läuft. Falls doch: `grep -r "http://" dist/` und schauen,
ob irgendein Asset hardcoded HTTP referenziert.

**"SharedArrayBuffer is not defined"** — die `Cross-Origin-Embedder-Policy`-
Header sind gesetzt, Browser muss `crossOriginIsolated=true` haben. Das ist
*optional* — TF.js funktioniert auch ohne SharedArrayBuffer, nur langsamer.
Falls Probleme: Header-Block in `nginx-vhost.conf` auskommentieren.

**TF.js-Bundle ist 1.5MB** — beim ersten Laden spürbar, danach durch
Browser-Cache kein Problem. Für Production mit viel Traffic:
`@tensorflow/tfjs` durch die schlankeren Pakete `@tensorflow/tfjs-core` +
`@tensorflow/tfjs-converter` + `@tensorflow/tfjs-backend-webgl` ersetzen,
spart ~30% Bundle-Size.
