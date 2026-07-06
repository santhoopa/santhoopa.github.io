# Santhoopa Jayawardhana — Personal Portfolio & Space Sim

[![Live Website](https://img.shields.io/badge/Live-santhoopajayawardhana.online-6EF5C8?style=for-the-badge&logo=google-chrome&logoColor=05060A)](https://www.santhoopajayawardhana.online/)
[![GitHub Pages](https://img.shields.io/badge/Hosted_on-GitHub_Pages-black?style=for-the-badge&logo=github&logoColor=white)](https://pages.github.com/)

Welcome to the public repository for my personal developer portfolio and interactive playground. Built with a futuristic terminal/cyberpunk aesthetic, this single-page application showcases my experience in full-stack engineering, distributed systems, and agentic AI.

---

## 🚀 Key Features

* **Futuristic HUD UI**: A premium, high-performance user interface designed with a dark, high-contrast, cyberpunk-inspired theme.
* **Custom Motion & Telemetry Engines**:
  * Simulated bootloader loading sequence (`SANTHOOPA.SYS // BOOT`).
  * Smooth parallax backgrounds and particle collision fields using HTML5 Canvas.
  * Adaptive cursor tracking (magnetic dot and ring tracer) with auto-hiding on mobile.
  * Interactive **Back to Top** navigation control with responsive scroll-threshold trigger.
* **Interactive Mini-Game**: A retro asteroid-shooter arcade simulation, **`DEBRIS.FIELD // SIM`** (playable at `/game.html` or via the floating sidebar), that includes:
  * Realistic **elastic, momentum-conserving rock-vs-rock collisions**.
  * Custom **vector spaceship rendering** with dual-thruster animations and edge-bounce boundary physics.
  * Procedurally generated asteroid **craters**.
  * Standalone **Web Audio API synthesizer** generating real-time retro game SFX (shots, hits, explosions, wave alerts, and active engine hum).
* **Technical Writing**: Features a showcase of published articles, including insights on **Spec-Driven Development (SDD) with Coding Agents** and deep learning topics.
* **Telemetry & Chat**: Integrated with Tidio for real-time channels and Umami Cloud for privacy-respecting analytics.
* **SEO & Knowledge Graph Ready**: Optimised metadata, Open Graph profiles, and JSON-LD structured data schema (Person type) for clean search indexing.

---

## 🛠️ Technology Stack

* **Frontend & Logic**: Vanilla HTML5, JavaScript (ES6+), Canvas API, and custom-tailored animations.
* **Styling**: Vanilla CSS3 (custom HSL variables, responsive grids, media query containers, micro-interactions).
* **Assets & Graphics**: PNG icons, custom OG previews, and web fonts (`Space Grotesk`, `JetBrains Mono`, `Instrument Serif`).

---

## 📂 Repository Structure

```
├── CNAME                      # Custom domain configuration
├── README.md                  # Project documentation
├── index.html                 # Main portfolio page (HUD, sections, scripts)
├── game.html                  # DEBRIS.FIELD // SIM (Asteroids arcade clone)
├── favicon.png                # Standard page icon
├── apple-touch-icon.png       # iOS home-screen icon
├── og-image.png               # Social preview thumbnail
├── robots.txt                 # Search crawler instructions
├── sitemap.xml                # XML Sitemap for search engines
└── google121e0b55ec82350a.html # Google Search Console verification
```

---

## 💻 Running & Previewing Locally

Since the site is built on vanilla web standards, you can run it locally without any build steps or package managers.

### Option 1: Live Server (Recommended)
If you use VS Code, install the **Live Server** extension, open the repository folder, and click **Go Live** in the status bar.

### Option 2: Python HTTP Server
Run the following command in your terminal from the project root:
```bash
# Python 3
python -m http.server 8000
```
Then navigate to `http://localhost:8000` in your web browser.

---

## 📬 Contact & Links

* **Portfolio**: [www.santhoopajayawardhana.online](https://www.santhoopajayawardhana.online/)
* **LinkedIn**: [/in/santhoopa-jayawardhana-baa058132](https://www.linkedin.com/in/santhoopa-jayawardhana-baa058132)
* **ResearchGate**: [/profile/Santhoopa-Jayawardhana](https://www.researchgate.net/profile/Santhoopa-Jayawardhana)
* **Medium Blog**: [@santhoopajayawardhana](https://medium.com/@santhoopajayawardhana)
