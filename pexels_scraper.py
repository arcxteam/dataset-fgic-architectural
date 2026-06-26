#!/usr/bin/env python3
"""
Pexels Image Scraper — config-driven, 3-mode cascade with perceptual dedup.
Config: config_images.json | Keys: .env | Output: dataset/<class>/

CLI:
  python pexels_scraper.py [N]            # scrape N/class + recompress
  python pexels_scraper.py --recompress   # recompress only
  python pexels_scraper.py --stats        # dataset stats
  python pexels_scraper.py --dedup        # remove visual dupes
  python pexels_scraper.py --dedup --dry-run
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path(__file__).resolve().parent


def _load_env():
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    with open(env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


@dataclass
class ScrapeConfig:
    target_per_class: int = 1680
    raw_dir: Path = field(default_factory=lambda: BASE_DIR / "dataset")
    classes: Dict[str, List[str]] = field(default_factory=dict)

    compress_max_kb: int = 200
    compress_max_dim: int = 720
    compress_min_dim: int = 200
    quality_start: int = 85
    quality_min: int = 40
    quality_step: int = 5

    min_width: int = 150
    min_height: int = 150
    max_aspect: float = 3.5
    min_aspect: float = 0.28
    min_color_std: float = 10.0

    download_workers: int = 12
    download_timeout: int = 15
    api_delay: float = 0.7
    api_per_page: int = 80
    api_max_pages: int = 100
    orientations: List[str] = field(default_factory=lambda: ["landscape", "portrait"])
    skip_url_patterns: List[str] = field(default_factory=lambda: [
        "logo", "avatar", "profile", "icon", "1x1", "sprite", "drawing",
        "people", "ai", "unreal", "badge", "button", "pixel", "tracking",
        "emoji", "favicon", "text", "illustration",
    ])
    skip_extensions: List[str] = field(default_factory=lambda: [".svg", ".gif", ".webm", ".mp4", ".ico"])

    bd_max_scrolls: int = 80
    bd_stale_limit: int = 5
    bd_scroll_delay: float = 1.5

    phash_enabled: bool = True
    phash_size: int = 16
    phash_threshold: int = 8

    pexels_key: str = ""
    brightdata_auth: str = ""

    @classmethod
    def from_json(cls, path: Path) -> "ScrapeConfig":
        with open(path) as f:
            d = json.load(f)
        comp = d.get("compression", {})
        qf = d.get("quality_filters", {})
        sc = d.get("scraping", {})
        pd = d.get("perceptual_dedup", {})
        return cls(
            target_per_class=d.get("target_per_class", 1680),
            raw_dir=BASE_DIR / d.get("output_dir", "dataset"),
            classes={c["name"]: c["keywords"] for c in d.get("classes", [])},
            compress_max_kb=comp.get("max_kb", 200),
            compress_max_dim=comp.get("max_dimension", 720),
            compress_min_dim=comp.get("min_dimension", 200),
            quality_start=comp.get("quality_start", 85),
            quality_min=comp.get("quality_min", 40),
            quality_step=comp.get("quality_step", 5),
            min_width=qf.get("min_width", 150),
            min_height=qf.get("min_height", 150),
            max_aspect=qf.get("max_aspect_ratio", 3.5),
            min_aspect=qf.get("min_aspect_ratio", 0.28),
            min_color_std=qf.get("min_color_std", 10.0),
            download_workers=sc.get("download_workers", 12),
            download_timeout=sc.get("download_timeout_seconds", 15),
            api_delay=sc.get("api_delay_seconds", 0.7),
            api_per_page=sc.get("api_per_page", 80),
            api_max_pages=sc.get("api_max_pages", 100),
            orientations=sc.get("orientations", ["landscape", "portrait"]),
            skip_url_patterns=sc.get("skip_url_patterns", []),
            skip_extensions=sc.get("skip_extensions", []),
            bd_max_scrolls=sc.get("brightdata_max_scrolls", 80),
            bd_stale_limit=sc.get("brightdata_stale_limit", 5),
            bd_scroll_delay=sc.get("brightdata_scroll_delay", 1.5),
            phash_enabled=pd.get("enabled", True),
            phash_size=pd.get("hash_size", 16),
            phash_threshold=pd.get("threshold", 8),
            pexels_key=os.environ.get("PEXELS_API_KEY", ""),
            brightdata_auth=os.environ.get("BRIGHTDATA_AUTH", ""),
        )


class PexelsScraper:
    _UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

    def __init__(self, cfg: ScrapeConfig):
        self.cfg = cfg
        self.hashes: Set[str] = set()
        self.perceptual_hashes: Dict[str, Dict[str, bool]] = {}
        self.lock = threading.Lock()
        self.stats: Dict[str, int] = {c: 0 for c in cfg.classes}
        self.next_idx: Dict[str, int] = {c: 0 for c in cfg.classes}
        self.reject_stats: Dict[str, int] = {}
        self.src_stats: Dict[str, int] = {}
        self._stop = False
        self.api_remaining = 25000
        self.driver = None
        self.brightdata_ok = False

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self._UA})

        self.log_path = BASE_DIR / "pexels_scraping.log"
        self.checkpoint_path = BASE_DIR / "pexels_checkpoint.json"

        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))

        self._imagehash = None
        self._numpy = None

    def _lazy_imagehash(self):
        if self._imagehash is None:
            try:
                import imagehash
                self._imagehash = imagehash
            except ImportError:
                pass
        return self._imagehash

    def _lazy_numpy(self):
        if self._numpy is None:
            try:
                import numpy
                self._numpy = numpy
            except ImportError:
                pass
        return self._numpy

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def should_stop(self, cls_name: Optional[str] = None) -> bool:
        if self._stop:
            return True
        if cls_name and self.stats.get(cls_name, 0) >= self.cfg.target_per_class:
            return True
        if sum(self.stats.values()) >= self.cfg.target_per_class * len(self.cfg.classes):
            return True
        return False

    def valid_url(self, url: str) -> bool:
        if not url or not url.startswith("http"):
            return False
        lo = url.lower()
        if any(p in lo for p in self.cfg.skip_url_patterns):
            return False
        if lo.endswith(tuple(self.cfg.skip_extensions)):
            return False
        return True

    def _decode_image(self, content: bytes) -> Optional[Image.Image]:
        try:
            return Image.open(BytesIO(content)).convert("RGB")
        except Exception:
            return None

    def compress(self, content: bytes) -> bytes:
        img = self._decode_image(content)
        if img is None:
            return content
        if max(img.size) > self.cfg.compress_max_dim:
            img.thumbnail((self.cfg.compress_max_dim, self.cfg.compress_max_dim), Image.Resampling.LANCZOS)
        q = self.cfg.quality_start
        while q >= self.cfg.quality_min:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            if len(buf.getvalue()) / 1024 <= self.cfg.compress_max_kb:
                return buf.getvalue()
            q -= self.cfg.quality_step
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=self.cfg.quality_min, optimize=True)
        return buf.getvalue()

    def quality_check(self, img: Image.Image) -> Tuple[bool, str]:
        w, h = img.size
        if w < self.cfg.min_width or h < self.cfg.min_height:
            return False, f"too_small({w}x{h})"
        ratio = w / h
        if ratio > self.cfg.max_aspect or ratio < self.cfg.min_aspect:
            return False, f"bad_ratio({ratio:.2f})"
        np = self._lazy_numpy()
        if np is not None:
            arr = np.array(img)
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                small = arr[::max(1, arr.shape[0] // 50), ::max(1, arr.shape[1] // 50)]
                std = float(small.std())
                if std < self.cfg.min_color_std:
                    return False, f"low_color_std({std:.1f})"
        return True, "ok"

    def is_visual_duplicate(self, img: Image.Image, cls_name: str) -> bool:
        ih = self._lazy_imagehash()
        if not self.cfg.phash_enabled or ih is None:
            return False
        try:
            h = ih.phash(img, hash_size=self.cfg.phash_size)
            h_str = str(h)
            with self.lock:
                cls_phashes = self.perceptual_hashes.get(cls_name, {})
                for existing_h in cls_phashes:
                    if h - ih.hex_to_hash(existing_h) <= self.cfg.phash_threshold:
                        return True
                cls_phashes[h_str] = True
                self.perceptual_hashes[cls_name] = cls_phashes
            return False
        except Exception:
            return False

    def _reject(self, reason: str):
        with self.lock:
            self.reject_stats[reason] = self.reject_stats.get(reason, 0) + 1

    def save_img(self, content: bytes, save_dir: Path, cls_name: str, src: str) -> bool:
        with self.lock:
            if self.should_stop(cls_name):
                return False
            if self.stats.get(cls_name, 0) >= self.cfg.target_per_class:
                return False

        content = self.compress(content)
        if len(content) / 1024 < 3:
            self._reject("too_small_after_compress")
            return False

        img = self._decode_image(content)
        if img is None:
            self._reject("decode_err")
            return False

        passed, reason = self.quality_check(img)
        if not passed:
            self._reject(reason)
            return False

        if self.is_visual_duplicate(img, cls_name):
            self._reject("visual_duplicate")
            return False

        h = hashlib.sha256(content).hexdigest()
        with self.lock:
            if h in self.hashes:
                self._reject("sha256_duplicate")
                return False
            if self.should_stop(cls_name):
                return False
            self.hashes.add(h)
            idx = self.next_idx[cls_name]
            self.next_idx[cls_name] = idx + 1

        filepath = save_dir / f"{cls_name}_{idx:05d}.jpg"
        try:
            with open(filepath, "wb") as f:
                f.write(content)
        except Exception:
            with self.lock:
                self.hashes.discard(h)
                self.next_idx[cls_name] = idx
            self._reject("write_failed")
            return False

        if not filepath.exists() or filepath.stat().st_size == 0:
            with self.lock:
                self.hashes.discard(h)
                self.next_idx[cls_name] = idx
            self._reject("write_verified_fail")
            return False

        with self.lock:
            self.stats[cls_name] += 1
            self.src_stats[src] = self.src_stats.get(src, 0) + 1
        return True

    def download_url(self, url: str) -> Optional[bytes]:
        try:
            r = self.session.get(url, timeout=self.cfg.download_timeout, stream=True)
            if r.status_code == 200 and len(r.content) > 3000:
                return r.content
        except Exception:
            pass
        return None

    def dl_save(self, url: str, save_dir: Path, cls_name: str, src: str):
        if self.should_stop(cls_name):
            return
        try:
            content = self.download_url(url)
            if content:
                self.save_img(content, save_dir, cls_name, src)
        except Exception:
            pass

    # --- checkpoint ---

    def _parse_idx(self, fn: str, cls_name: str) -> int:
        base = fn.replace(f"{cls_name}_", "").rsplit(".", 1)[0]
        try:
            return int(base)
        except ValueError:
            return -1

    def load_existing(self):
        ih = self._lazy_imagehash()
        for cls_name in self.cfg.classes:
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            cls_phashes: Dict[str, bool] = {}
            max_idx = -1
            for fn in os.listdir(cd):
                fp = cd / fn
                if not fp.is_file():
                    continue
                try:
                    data = fp.read_bytes()
                    h = hashlib.sha256(data).hexdigest()
                    with self.lock:
                        self.hashes.add(h)
                        self.stats[cls_name] += 1
                    idx = self._parse_idx(fn, cls_name)
                    if idx > max_idx:
                        max_idx = idx
                    if ih and self.cfg.phash_enabled:
                        try:
                            img = Image.open(BytesIO(data))
                            ph = ih.phash(img, hash_size=self.cfg.phash_size)
                            cls_phashes[str(ph)] = True
                        except Exception:
                            pass
                except Exception:
                    pass
            with self.lock:
                self.next_idx[cls_name] = max_idx + 1 if max_idx >= 0 else 0
            if cls_phashes:
                with self.lock:
                    self.perceptual_hashes[cls_name] = cls_phashes

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return
        try:
            data = json.loads(self.checkpoint_path.read_text())
            if "hashes" in data:
                with self.lock:
                    self.hashes.update(data["hashes"])
            if "perceptual_hashes" in data:
                with self.lock:
                    for cls_name, ph_list in data["perceptual_hashes"].items():
                        self.perceptual_hashes[cls_name] = {ph: True for ph in ph_list}
            ph_count = sum(len(v) for v in self.perceptual_hashes.values())
            self.log(f"Checkpoint: {len(data.get('hashes', []))} hashes, {ph_count} phash")
        except Exception:
            pass

    def checkpoint(self):
        ph_serializable = {}
        with self.lock:
            for cls_name, ph_dict in self.perceptual_hashes.items():
                ph_serializable[cls_name] = list(ph_dict.keys())
        data = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": dict(self.stats),
            "src_stats": dict(self.src_stats),
            "reject_stats": dict(self.reject_stats),
            "hashes": list(self.hashes),
            "perceptual_hashes": ph_serializable,
            "total": sum(self.stats.values()),
            "api_remaining": self.api_remaining,
        }
        try:
            self.checkpoint_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def rebuild_checkpoint_from_disk(self):
        ih = self._lazy_imagehash()
        self.log("Rebuilding checkpoint from disk...")
        new_hashes: Set[str] = set()
        new_phashes: Dict[str, Dict[str, bool]] = {}
        new_stats: Dict[str, int] = {c: 0 for c in self.cfg.classes}
        new_next_idx: Dict[str, int] = {c: 0 for c in self.cfg.classes}

        for cls_name in self.cfg.classes:
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            cls_phashes: Dict[str, bool] = {}
            max_idx = -1
            for fn in sorted(os.listdir(cd)):
                fp = cd / fn
                if not fp.is_file():
                    continue
                try:
                    data = fp.read_bytes()
                    new_hashes.add(hashlib.sha256(data).hexdigest())
                    new_stats[cls_name] += 1
                    idx = self._parse_idx(fn, cls_name)
                    if idx > max_idx:
                        max_idx = idx
                    if ih and self.cfg.phash_enabled:
                        try:
                            img = Image.open(BytesIO(data))
                            ph = ih.phash(img, hash_size=self.cfg.phash_size)
                            cls_phashes[str(ph)] = True
                        except Exception:
                            pass
                except Exception:
                    pass
            new_next_idx[cls_name] = max_idx + 1 if max_idx >= 0 else 0
            if cls_phashes:
                new_phashes[cls_name] = cls_phashes

        with self.lock:
            self.hashes = new_hashes
            self.perceptual_hashes = new_phashes
            self.stats = new_stats
            self.next_idx = new_next_idx

        self.log(f"Checkpoint rebuilt: {sum(new_stats.values())} SHA256, "
                 f"{sum(len(v) for v in new_phashes.values())} phash")
        self.checkpoint()

    def renumber_files(self):
        self.log("Renumbering files...")
        total = 0
        for cls_name in self.cfg.classes:
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            files = sorted(
                [f for f in os.listdir(cd) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                key=lambda fn: self._parse_idx(fn, cls_name) if self._parse_idx(fn, cls_name) >= 0 else 999999
            )
            rename_map = [(fn, f"{cls_name}_{i:05d}.jpg") for i, fn in enumerate(files)
                          if fn != f"{cls_name}_{i:05d}.jpg"]
            if not rename_map:
                self.log(f"  {cls_name}: {len(files)} files — sequential ok")
                continue
            temp = []
            for old_fn, new_fn in rename_map:
                try:
                    os.rename(cd / old_fn, cd / f"_tmp_{old_fn}")
                    temp.append((f"_tmp_{old_fn}", new_fn))
                except Exception:
                    pass
            for tmp_fn, new_fn in temp:
                try:
                    os.rename(cd / tmp_fn, cd / new_fn)
                    total += 1
                except Exception:
                    pass
            self.log(f"  {cls_name}: {len(files)} files, {len(rename_map)} renamed")
        self.log(f"Renumbering complete: {total} renamed")

    # --- search: unified selenium ---

    def _selenium_search(self, driver, search_url: str, max_scrolls: int,
                         selector: str, extract_fn, stale_limit: int = 5,
                         scroll_delay: float = 1.5) -> List[str]:
        urls: Set[str] = set()
        try:
            driver.get(search_url)
            time.sleep(5)
            prev_count = 0
            stale = 0
            for _ in range(max_scrolls):
                if self._stop:
                    break
                try:
                    elements = driver.find_elements("css selector", selector)
                    if elements:
                        driver.execute_script(
                            "arguments[0].scrollIntoView({behavior:'smooth'});", elements[-1])
                    time.sleep(scroll_delay)
                    if len(elements) == prev_count:
                        stale += 1
                        if stale >= stale_limit:
                            break
                    else:
                        stale = 0
                    prev_count = len(elements)
                except Exception:
                    break
            for el in driver.find_elements("css selector", selector):
                try:
                    url = extract_fn(el)
                    if url and self.valid_url(url):
                        urls.add(url)
                except Exception:
                    pass
        except Exception:
            pass
        return list(urls)

    @staticmethod
    def _extract_photo_link(a) -> Optional[str]:
        href = a.get_attribute("href") or ""
        if "/photo/" not in href:
            return None
        parts = href.rstrip("/").split("/")
        pid = parts[-1].split("-")[0] if parts else ""
        if pid and pid.isdigit():
            return (f"https://images.pexels.com/photos/{pid}/"
                    f"pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640")
        return None

    @staticmethod
    def _extract_img_src(img) -> Optional[str]:
        src = img.get_attribute("src") or ""
        if "images.pexels.com/photos/" not in src:
            return None
        clean = src.split("?")[0]
        if not clean.endswith((".jpg", ".jpeg", ".png")):
            clean += ".jpeg"
        return clean

    # --- mode 1: pexels api ---

    def api_search(self, query: str, orientation: str = "landscape") -> List[str]:
        urls = []
        page = 1
        while page <= self.cfg.api_max_pages and not self._stop:
            try:
                params = {"query": query, "per_page": self.cfg.api_per_page, "page": page}
                if orientation:
                    params["orientation"] = orientation
                r = self.session.get(
                    "https://api.pexels.com/v1/search",
                    params=params,
                    headers={"Authorization": self.cfg.pexels_key},
                    timeout=15,
                )
                if r.status_code == 429:
                    self.log("    API rate limited, pausing 60s...")
                    time.sleep(60)
                    continue
                if r.status_code != 200:
                    self.log(f"    API error: {r.status_code}")
                    break

                remaining = r.headers.get("X-Ratelimit-Remaining")
                if remaining:
                    self.api_remaining = int(remaining)

                data = r.json()
                photos = data.get("photos", [])
                if not photos:
                    break
                for p in photos:
                    src_large = p.get("src", {}).get("large", "")
                    if src_large and self.valid_url(src_large):
                        urls.append(src_large)

                total_results = data.get("total_results", 0)
                if page * self.cfg.api_per_page >= total_results:
                    break
                if page % 10 == 0:
                    self.log(f"    API page {page}: +{len(photos)} "
                             f"(total={total_results}, remaining={self.api_remaining})")
                page += 1
                time.sleep(self.cfg.api_delay)
            except Exception as e:
                self.log(f"    API error: {e}")
                break
        return urls

    # --- mode 2: bright data selenium ---

    def init_brightdata(self):
        if not self.cfg.brightdata_auth:
            self.log("Bright Data: no auth, skipped")
            self.brightdata_ok = False
            return
        try:
            from selenium.webdriver import Remote
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection

            addr = f"https://{self.cfg.brightdata_auth}@brd.superproxy.io:9515"
            conn = ChromiumRemoteConnection(addr, "goog", "chrome")
            opts = Options()
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            self.driver = Remote(conn, options=opts)
            self.brightdata_ok = True
            self.log("Bright Data: connected")
        except Exception as e:
            self.log(f"Bright Data: failed ({e})")
            self.brightdata_ok = False

    def close_brightdata(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            self.brightdata_ok = False

    def brightdata_search(self, search_url: str) -> List[str]:
        if not self.brightdata_ok or not self.driver:
            return []
        urls = self._selenium_search(
            self.driver, search_url, self.cfg.bd_max_scrolls,
            "a[href*='/photo/']", self._extract_photo_link,
            stale_limit=self.cfg.bd_stale_limit,
            scroll_delay=self.cfg.bd_scroll_delay,
        )
        try:
            articles = self.driver.find_elements("css selector", "a[href*='/photo/']")
            self.log(f"    Bright Data: {len(urls)} URLs from {len(articles)} articles")
        except Exception:
            self.log(f"    Bright Data: {len(urls)} URLs")
        if not urls:
            self.close_brightdata()
            time.sleep(2)
            self.init_brightdata()
        return urls

    # --- mode 3: local selenium ---

    def local_selenium_search(self, search_url: str, max_scrolls: int = 15) -> List[str]:
        try:
            from selenium.webdriver import Remote
            from selenium.webdriver.chrome.options import Options

            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            driver = Remote(options=opts)
            driver.set_page_load_timeout(20)
            try:
                urls = self._selenium_search(
                    driver, search_url, max_scrolls,
                    "img[src*='images.pexels.com']", self._extract_img_src,
                    stale_limit=5, scroll_delay=2.0,
                )
                self.log(f"    Local Selenium: {len(urls)} URLs")
                return urls
            finally:
                driver.quit()
        except Exception as e:
            self.log(f"    Local Selenium error: {e}")
            return []

    # --- download ---

    def download_batch(self, urls: List[str], save_dir: Path, cls_name: str, src: str):
        random.shuffle(urls)
        need = max(0, self.cfg.target_per_class - self.stats.get(cls_name, 0))
        cap = min(len(urls), need * 3)
        if cap <= 0:
            return
        self.log(f"    Downloading {cap}/{len(urls)} URLs...")
        saved = 0
        last_log = time.time()
        with ThreadPoolExecutor(self.cfg.download_workers) as ex:
            futs = {ex.submit(self.dl_save, u, save_dir, cls_name, src): u for u in urls[:cap]}
            for fut in as_completed(futs):
                if self.should_stop(cls_name):
                    break
                try:
                    fut.result()
                except Exception:
                    pass
                saved += 1
                now = time.time()
                c = self.stats.get(cls_name, 0)
                if saved % 30 == 0 or now - last_log > 20:
                    last_log = now
                    self.log(f"      [{cls_name}] {c}/{self.cfg.target_per_class} ({saved} processed)")

    def scrape_class(self, cls_name: str, queries: List[str]):
        if self.should_stop(cls_name):
            self.log(f"[{cls_name}] SKIP — target reached")
            return

        save_dir = self.cfg.raw_dir / cls_name
        save_dir.mkdir(parents=True, exist_ok=True)

        need = self.cfg.target_per_class - self.stats.get(cls_name, 0)
        self.log(f"[{cls_name}] START — need {need}")

        all_urls: List[str] = []

        for qi, query in enumerate(queries):
            if self.should_stop(cls_name):
                break
            self.log(f'  Query [{qi+1}/{len(queries)}]: "{query}"')
            search_url = f"https://www.pexels.com/search/{quote(query)}/"

            if self.api_remaining > 20:
                for orient in self.cfg.orientations:
                    api_urls = self.api_search(query, orientation=orient)
                    before = len(set(u.split("?")[0] for u in all_urls))
                    all_urls.extend(api_urls)
                    after = len(set(u.split("?")[0] for u in all_urls))
                    self.log(f"    API ({orient}): {len(api_urls)} URLs, +{after - before} new")
                    if self.should_stop(cls_name):
                        break

            if self.should_stop(cls_name):
                break

            if self.brightdata_ok and self.stats.get(cls_name, 0) < self.cfg.target_per_class:
                bd_urls = self.brightdata_search(search_url)
                before = len(set(u.split("?")[0] for u in all_urls))
                all_urls.extend(bd_urls)
                after = len(set(u.split("?")[0] for u in all_urls))
                self.log(f"    Bright Data: +{after - before} new URLs")

            self.download_batch(all_urls, save_dir, cls_name, "pexels")
            all_urls = []

            self.log(f"  [{cls_name}] {self.stats.get(cls_name, 0)}/{self.cfg.target_per_class}")
            if self.stats.get(cls_name, 0) >= self.cfg.target_per_class:
                break
            self.checkpoint()

        if self.stats.get(cls_name, 0) < self.cfg.target_per_class:
            self.log(f"  [{cls_name}] Local Selenium fallback...")
            for query in queries:
                if self.should_stop(cls_name):
                    break
                local_urls = self.local_selenium_search(
                    f"https://www.pexels.com/search/{quote(query)}/")
                if local_urls:
                    self.download_batch(local_urls, save_dir, cls_name, "pexels_local")
                if self.stats.get(cls_name, 0) >= self.cfg.target_per_class:
                    break

        self.log(f"[{cls_name}] DONE — {self.stats.get(cls_name, 0)}/{self.cfg.target_per_class}")
        self.checkpoint()

    # --- recompress ---

    def recompress_raw(self):
        self.log("Re-compressing all raw images...")
        for cls_name in self.cfg.classes:
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            files = [f for f in os.listdir(cd) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            total_orig = total_new = count = 0
            for fn in files:
                fp = cd / fn
                try:
                    data = fp.read_bytes()
                    orig_size = len(data)
                    total_orig += orig_size
                    result = self.compress(data)
                    new_size = len(result)
                    total_new += new_size
                    if new_size < orig_size:
                        fp.write_bytes(result)
                        count += 1
                except Exception:
                    pass
            self.log(f"  {cls_name}: {count}/{len(files)} recompressed | "
                     f"{total_orig/1024/1024:.1f} → {total_new/1024/1024:.1f} MB")

    # --- dedup ---

    def perceptual_dedup(self, dry_run: bool = False):
        ih = self._lazy_imagehash()
        if ih is None:
            self.log("ERROR: imagehash not installed")
            return
        if not self.cfg.phash_enabled:
            self.log("Perceptual dedup disabled in config")
            return

        hs = self.cfg.phash_size
        th = self.cfg.phash_threshold
        self.log(f"Perceptual dedup: hash_size={hs}, threshold={th}, dry_run={dry_run}")

        total_removed = 0
        total_kept = 0

        for cls_name in sorted(self.cfg.classes):
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            files = sorted(f for f in os.listdir(cd) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            file_hashes = []
            for fn in files:
                try:
                    img = Image.open(cd / fn)
                    file_hashes.append((fn, ih.phash(img, hash_size=hs)))
                except Exception:
                    pass

            to_delete: Set[str] = set()
            kept_hashes = []
            for fn, h in file_hashes:
                if any(h - kh <= th for kh in kept_hashes):
                    to_delete.add(fn)
                else:
                    kept_hashes.append(h)

            kept = len(file_hashes) - len(to_delete)
            total_kept += kept
            total_removed += len(to_delete)

            action = "WOULD DELETE" if dry_run else "DELETING"
            self.log(f"  {cls_name}: {len(file_hashes)} files, {len(to_delete)} dupes, {kept} kept ({action})")

            if not dry_run:
                for fn in to_delete:
                    try:
                        os.remove(cd / fn)
                    except Exception:
                        pass

        self.log(f"Total: {total_kept} kept, {total_removed} removed, dry_run={dry_run}")

        if not dry_run and total_removed > 0:
            self.renumber_files()
            self.rebuild_checkpoint_from_disk()

    # --- stats ---

    def print_summary(self):
        total = sum(self.stats.values())
        self.log("=" * 60)
        self.log(f"DONE: {total} images")
        for n, c in self.stats.items():
            self.log(f"  {n}: {c}/{self.cfg.target_per_class}")
        for s, c in sorted(self.src_stats.items(), key=lambda x: -x[1]):
            self.log(f"    source {s}: {c}")
        self.log("  Reject reasons:")
        for r, c in sorted(self.reject_stats.items(), key=lambda x: -x[1]):
            self.log(f"    {r}: {c}")
        for cls_name in self.cfg.classes:
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                continue
            files = [f for f in os.listdir(cd) if f.lower().endswith((".jpg", ".jpeg"))]
            sizes = [(cd / f).stat().st_size for f in files]
            if sizes:
                self.log(f"  {cls_name}: avg={sum(sizes)/len(sizes)/1024:.0f}KB, "
                         f"min={min(sizes)/1024:.0f}KB, max={max(sizes)/1024:.0f}KB")
        self.log("=" * 60)

    def show_stats(self):
        print("\n" + "=" * 60)
        print("DATASET STATS")
        print("=" * 60)
        total = 0
        for cls_name in sorted(self.cfg.classes):
            cd = self.cfg.raw_dir / cls_name
            if not cd.exists():
                print(f"  {cls_name}: NOT FOUND")
                continue
            files = [f for f in os.listdir(cd) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            sizes, dims_w, dims_h = [], [], []
            corrupt = 0
            for fn in files:
                fp = cd / fn
                try:
                    sizes.append(fp.stat().st_size)
                    img = Image.open(fp)
                    w, h = img.size
                    dims_w.append(w)
                    dims_h.append(h)
                except Exception:
                    corrupt += 1
            n = len(files)
            total += n
            if sizes:
                print(f"  {cls_name}: {n} files, corrupt={corrupt}")
                print(f"    Size: avg={sum(sizes)/len(sizes)/1024:.0f}KB, "
                      f"min={min(sizes)/1024:.0f}KB, max={max(sizes)/1024:.0f}KB")
                print(f"    Dim: {min(dims_w)}x{min(dims_h)} — {max(dims_w)}x{max(dims_h)}")
                print(f"    Over 200KB: {sum(1 for s in sizes if s > 200*1024)}, "
                      f"Over 720px: {sum(1 for w in dims_w if w > 720) + sum(1 for h in dims_h if h > 720)}")
            else:
                print(f"  {cls_name}: 0 files")
        print(f"\n  TOTAL: {total} images, {len(self.cfg.classes)} classes")
        print("=" * 60)

    # --- main ---

    def run(self):
        self.log("=" * 60)
        self.log("PEXELS SCRAPER")
        self.log(f"  Config: config_images.json")
        self.log(f"  Target: {self.cfg.target_per_class}/class "
                 f"({self.cfg.target_per_class * len(self.cfg.classes)} total)")
        self.log(f"  Output: {self.cfg.raw_dir}")
        self.log(f"  Compression: max {self.cfg.compress_max_kb}KB, max {self.cfg.compress_max_dim}px")
        self.log(f"  Mode 1: Pexels API | Mode 2: Bright Data | Mode 3: Local Selenium")
        self.log("=" * 60)

        self.load_existing()
        self.load_checkpoint()
        if self.hashes:
            self.log(f"Resume: {len(self.hashes)} existing hashes")

        self.cfg.raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.init_brightdata()
            for cls_name, queries in self.cfg.classes.items():
                if self.should_stop():
                    self.log("TARGET REACHED — stopping")
                    break
                self.scrape_class(cls_name, queries)
        except Exception as e:
            self.log(f"FATAL: {e}")
        finally:
            self.close_brightdata()

        self.checkpoint()
        self.recompress_raw()
        self.print_summary()


if __name__ == "__main__":
    config_path = BASE_DIR / "config_images.json"
    if not config_path.exists():
        print(f"FATAL: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    cfg = ScrapeConfig.from_json(config_path)

    if "--stats" in sys.argv:
        PexelsScraper(cfg).show_stats()
    elif "--recompress" in sys.argv:
        s = PexelsScraper(cfg)
        s.recompress_raw()
        s.show_stats()
    elif "--dedup" in sys.argv:
        s = PexelsScraper(cfg)
        s.perceptual_dedup(dry_run="--dry-run" in sys.argv)
        s.show_stats()
    else:
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            cfg.target_per_class = int(sys.argv[1])
        PexelsScraper(cfg).run()
