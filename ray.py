#!/usr/bin/env python3
"""
ray-cinema — JSON tanımlı sahneleri Monte Carlo path tracing ile render eden,
sıfırdan yazılmış fiziksel ışık taşıma motoru (NumPy vektörize).

Özellikler: Lambertian + metal (pürüzlülük) + dielektrik (cam, Snell+Fresnel),
yayıcı malzemeler, ince lens alan derinliği, çoklu örneklem AA, ACES tonemap.

    python3 ray.py --scene scenes/glass.json
    python3 ray.py --scene scenes/glass.json --preview
    python3 ray.py --scene scenes/glass.json --turntable 12
"""

import sys
import argparse
import json
import shutil
import math
import os
import struct
import subprocess
import sys
import zlib

import numpy as np

EPS = 1e-4
CHUNK = 48000  # aynı anda izlenen ışın sayısı (bellek sınırı)


# ---------------------------------------------------------------- png

def png_yaz(path, img):
    h, w, _ = img.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += img[y].tobytes()

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- vektör

def birim(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def yansit(v, n):
    """Ayna yansıması: v - 2(v·n)n."""
    return v - 2 * np.sum(v * n, axis=-1, keepdims=True) * n


def kirilt(v, n, oran):
    """Snell kırılması. toplam iç yansıma -> maske False."""
    oran = np.asarray(oran, float)
    if oran.ndim == 1:
        oran = oran[:, None]
    kos = np.sum(v * n, axis=-1, keepdims=True)
    perp = oran * (v - kos * n)
    par = (oran * np.abs(kos) - np.sqrt(
        np.maximum(0.0, 1.0 - oran * oran * (1 - np.abs(kos) ** 2)))) * n
    t = birim(perp + par)
    return t, (np.abs(kos) > 1e-6)[:, 0]


def fresnel_schlick(kosinus, ior):
    r0 = ((1 - ior) / (1 + ior)) ** 2
    return r0 + (1 - r0) * (1 - kosinus) ** 5


def rastgele_birim_yari_kure(rng, n):
    """Küresel kabuk üzerinde normalize edilmemiş rastgele yön (diffuse)."""
    v = rng.normal(size=(n, 3))
    uz = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(uz == 0, 1.0, uz)


# ---------------------------------------------------------------- sahne

class Sahne:
    def __init__(self, cfg):
        self.kureler = []
        for k in cfg["objects"]:
            self.kureler.append({
                "merkez": np.array(k["center"], float),
                "yaricap": float(k["radius"]),
                "tur": k["material"]["type"],
                "renk": np.array(k["material"].get("albedo", [1.0, 1.0, 1.0]), float),
                "pürüzlülük": float(k["material"].get("roughness", 0.0)),
                "ior": float(k["material"].get("ior", 1.5)),
                "yayma": float(k["material"].get("emission", 0.0)),
            })
        self.gok_renk_ust = np.array(cfg["sky"]["top"], float)
        self.gok_renk_alt = np.array(cfg["sky"]["bottom"], float)
        self.gok_guc = float(cfg["sky"].get("strength", 1.0))
        self.yayicilar = [i for i, k in enumerate(self.kureler) if k["yayma"] > 0]

    def gokyuzu(self, yon):
        t = 0.5 * (birim(yon)[..., 1:2] + 1.0)
        return (self.gok_renk_alt * (1 - t) + self.gok_renk_ust * t) * self.gok_guc

    def kesisim(self, origin, direction, t_min, t_max):
        """Tüm kürelerle kesişim. Döner: (t, nesne_indeksi) en yakın."""
        n_ray = origin.shape[0]
        en_yakin_t = np.full(n_ray, t_max)
        en_yakin_k = np.full(n_ray, -1, dtype=int)
        for k, kure in enumerate(self.kureler):
            ok = origin - kure["merkez"]
            b = np.sum(ok * direction, axis=-1)
            c = np.sum(ok * ok, axis=-1) - kure["yaricap"] ** 2
            disk = b * b - c
            var = disk > 0
            kk = np.sqrt(np.where(var, disk, 0.0))
            t1 = -b - kk
            t2 = -b + kk
            for t in (t1, t2):
                gecerli = var & (t > t_min) & (t < en_yakin_t)
                en_yakin_t = np.where(gecerli, t, en_yakin_t)
                en_yakin_k = np.where(gecerli, k, en_yakin_k)
        return en_yakin_t, en_yakin_k


# ---------------------------------------------------------------- kamera

class Kamera:
    def __init__(self, cfg, en, boy, rng):
        self.en, self.boy = en, boy
        self.pos = np.array(cfg["position"], float)
        self.bak = np.array(cfg["look_at"], float)
        self.vfov = math.radians(cfg.get("vfov", 45))
        self.apertura = cfg.get("aperture", 0.0)
        self.odak = cfg.get("focus_distance") or np.linalg.norm(self.bak - self.pos)
        self.rng = rng

    def isinlar(self, jitter=None):
        rng = self.rng
        en, boy = self.en, self.boy
        i = np.arange(en)
        j = np.arange(boy)
        jj, ii = np.meshgrid(j, i, indexing="ij")
        ii = ii.reshape(-1).astype(float)
        jj = jj.reshape(-1).astype(float)
        if jitter is None:
            jitter = rng.random((ii.size, 2))
        ii += jitter[:, 0]
        jj += jitter[:, 1]
        oran = en / boy
        yuk = 2 * math.tan(self.vfov / 2)
        gen = yuk * oran
        u = (ii / en - 0.5) * gen
        v = (0.5 - jj / boy) * yuk
        ileri = birim(self.bak - self.pos)
        sag = birim(np.cross(ileri, np.array([0.0, 1.0, 0.0])))
        yukari = np.cross(sag, ileri)
        dirs = (ileri[None, :]
                + u[:, None] * sag[None, :]
                + v[:, None] * yukari[None, :])
        origin = np.repeat(self.pos[None, :], dirs.shape[0], axis=0)
        if self.apertura > 0:
            disk = rng.normal(size=(dirs.shape[0], 2)) * self.apertura / 2
            ofset = disk[:, 0:1] * sag[None, :] + disk[:, 1:2] * yukari[None, :]
            origin = origin + ofset
            dirs = dirs - ofset
        return origin, birim(dirs)


# ---------------------------------------------------------------- path tracing

def izle(sahne, origin, direction, rng, derinlik):
    """Path tracing + dogrudan isik orneklemesi (NEE).

    x0 degiskeni: lambertian yansimasi sonrasi kacinir (cift sayim olmasin),
    metal/cam gibi spekulur sekmelerde isigi gormeye devam eder.
    """
    n_ray = origin.shape[0]
    renk = np.zeros((n_ray, 3))
    gecerli = np.ones(n_ray, dtype=bool)
    katki = np.ones((n_ray, 3))
    spekulor = np.ones(n_ray, dtype=bool)   # isigi gorebilir mi (cift sayimsiz)

    yayici_var = len(sahne.yayicilar) > 0

    for sekme in range(derinlik):
        if not gecerli.any():
            break
        idx = np.where(gecerli)[0]
        o, d = origin[idx], direction[idx]
        t_hit, k_hit = sahne.kesisim(o, d, EPS, 1e30)
        isabet = k_hit >= 0

        kacti = ~isabet
        if kacti.any():
            i_kac = idx[kacti]
            renk[i_kac] += katki[i_kac] * sahne.gokyuzu(d[kacti])
            gecerli[i_kac] = False
        if not isabet.any():
            continue

        ab = idx[isabet]
        oo, dd = o[isabet], d[isabet]
        tt, kk = t_hit[isabet], k_hit[isabet]

        konum = oo + dd * tt[:, None]
        kureler = [sahne.kureler[k] for k in kk]
        merkez = np.stack([k["merkez"] for k in kureler])
        normal_geo = birim(konum - merkez)                  # disa bakan (dielektrik icin)
        yuzey = np.sign(np.sum(dd * normal_geo, axis=-1, keepdims=True))
        yuzey = np.where(yuzey == 0, 1.0, yuzey)
        normal = -normal_geo * yuzey                        # isina bacakli yuzey normali

        # yayiciya dogrudan carpma: yalniz spekulor yolda say (NEE ile cift olmasin)
        yay = np.array([k["yayma"] for k in kureler])
        yayilan = yay > 0
        if yayilan.any():
            m_spk = yayilan & spekulor[ab]
            if m_spk.any():
                renk[ab[m_spk]] += katki[ab[m_spk]] * \
                    np.stack([k["renk"] for k in kureler])[m_spk] * yay[m_spk][:, None]
            gecerli[ab[yayilan]] = False

        yansayan = ~yayilan
        if not yansayan.any():
            continue

        ab_y = ab[yansayan]
        konum_y = konum[yansayan]
        normal_y = normal[yansayan]
        dd_y = dd[yansayan]
        katki_y = katki[ab_y]
        spec_y = spekulor[ab_y]
        kk2 = [sahne.kureler[k] for k in kk[yansayan]]
        tipler = [k["tur"] for k in kk2]
        renkler = np.stack([k["renk"] for k in kk2])

        # ---- NEE: dogrudan isik orneklemesi (lambertian yuzeylerde)
        if yayici_var:
            lam = np.array([t == "lambertian" for t in tipler])
            lam = lam & spekulor[ab_y]
            if lam.any():
                print(f"NEE: {int(lam.sum())} isin ornekleniyor", file=sys.stderr)
            if lam.any():
                m = int(lam.sum())
                konum_l = konum_y[lam]
                normal_l = normal_y[lam]
                secilen = np.asarray(sahne.yayicilar)[
                    rng.integers(0, len(sahne.yayicilar), size=m)]
                lig_k = [sahne.kureler[i] for i in secilen]
                lc = np.stack([k["merkez"] for k in lig_k])
                lr = np.array([k["yaricap"] for k in lig_k])
                Le = np.stack([k["renk"] for k in lig_k]) * \
                    np.array([k["yayma"] for k in lig_k])[:, None]

                to_l = lc - konum_l
                dist = np.linalg.norm(to_l, axis=-1)
                w = to_l / dist[:, None]
                yrd = np.where(np.abs(w[:, 1:2]) < 0.9,
                               np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
                u = birim(np.cross(w, yrd))
                vv = np.cross(w, u)

                sin2_max = np.maximum((lr / dist) ** 2, 1e-9)
                cos_max = np.sqrt(1 - sin2_max)
                e1, e2 = rng.random(m), rng.random(m)
                cos_t = 1 - e1 * (1 - cos_max)
                sin_t = np.sqrt(1 - cos_t ** 2)
                phi = 2 * np.pi * e2
                ldir = birim(w * cos_t[:, None]
                             + (u * np.cos(phi)[:, None] + vv * np.sin(phi)[:, None]) * sin_t[:, None])

                t_g, k_g = sahne.kesisim(konum_l + normal_l * EPS, ldir, EPS, 1e30)
                gorunur = k_g == secilen   # en yakin vurus isiksa: gorunur
                cos_yuzey = np.clip(np.sum(ldir * normal_l, axis=-1), 0, None)
                pdf = 1.0 / (2 * np.pi * (1 - cos_max))
                katki_nee = katki_y[lam] * renkler[lam] * \
                    (Le * cos_yuzey[:, None] / pdf[:, None]) * gorunur[:, None]
                renk[ab_y[lam]] += katki_nee

        # ---- yansima dallari
        yeni_yon = np.zeros_like(dd_y)
        devamlar = np.ones(len(ab_y), dtype=bool)
        yeni_spek = np.zeros(len(ab_y), dtype=bool)

        for tur in ("lambertian", "metal", "dielectric"):
            maske = np.array([t == tur for t in tipler])
            if not maske.any():
                continue
            m = int(maske.sum())
            n = normal_y[maske]
            ddd = dd_y[maske]
            if tur == "lambertian":
                hedef = n + rastgele_birim_yari_kure(rng, m)
                uzun = np.linalg.norm(hedef, axis=-1, keepdims=True)
                yeni_yon[maske] = np.where(uzun < 1e-9, n, hedef)
                katki_y[maske] = katki_y[maske] * renkler[maske]
                yeni_spek[maske] = False        # NEE sayildi
            elif tur == "metal":
                p = np.array([k["pürüzlülük"] for k in kk2 if k["tur"] == tur])
                yansima = yansit(ddd, n)
                karisti = rastgele_birim_yari_kure(rng, m) * p[:, None]
                yon = birim(yansima + karisti)
                gecerli_m = np.sum(yon * n, axis=-1) > 0
                yeni_yon[maske] = yon
                katki_y[maske] = katki_y[maske] * renkler[maske] * gecerli_m[:, None]
                devamlar[maske] = gecerli_m
                yeni_spek[maske] = True
            elif tur == "dielectric":
                ior = np.array([k["ior"] for k in kk2 if k["tur"] == tur])
                disari = np.sum(ddd * normal_geo[yansayan][maske], axis=-1) > 0
                n_kullan = np.where(disari[:, None], normal_geo[yansayan][maske], -normal_geo[yansayan][maske])
                oran = np.where(disari, ior, 1.0 / ior)[:, None]
                kr, gecerli_snell = kirilt(ddd, n_kullan, oran)
                kosinus = np.abs(np.sum(kr * n_kullan, axis=-1))
                fresnel = fresnel_schlick(kosinus, ior)
                yazura = rng.random(m) < fresnel
                yeni_yon[maske] = np.where(yazura[:, None], yansit(ddd, n_kullan), kr)
                devamlar[maske] = gecerli_snell | yazura
                yeni_spek[maske] = True

        origin[ab_y] = konum_y + normal_y * EPS
        direction[ab_y] = yeni_yon
        katki[ab_y] = katki_y
        spekulor[ab_y] = yeni_spek
        gecerli[ab_y] = devamlar

    return renk


def render(sahne, kamera, gen_cekirdek, spp, derinlik):
    toplam = np.zeros((kamera.boy * kamera.en, 3))
    for s in range(spp):
        rng = np.random.default_rng(gen_cekirdek + s)
        o, d = kamera.isinlar()
        toplam += izle(sahne, o, d, rng, derinlik)
        if (s + 1) % 10 == 0:
            print(f"  spp {s + 1}/{spp}")
    toplam /= spp
    return toplam.reshape(kamera.boy, kamera.en, 3)


def aces(img):
    """ACES yaklaşık tonemap + gamma."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip(img, 0, 8)
    out = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return (out ** (1 / 2.2) * 255).astype(np.uint8)


# ---------------------------------------------------------------- ana

def yukle_ve_render(scene_path, spp=None, en=None, boy=None, aci=None, cekirdek=1):
    cfg = json.load(open(scene_path, encoding="utf-8"))
    r = cfg["render"]
    W = en or r.get("width", 960)
    H = boy or r.get("height", 540)
    spp = spp or r.get("spp", 100)
    derinlik = r.get("max_depth", 8)
    cam_cfg = dict(cfg["camera"])
    if aci is not None:
        # turntable: kamerayi sahne etrafinda dondur
        pos = np.array(cam_cfg["position"]) - np.array(cam_cfg["look_at"])
        t = math.radians(aci)
        donuk = np.array([
            pos[0] * math.cos(t) - pos[2] * math.sin(t),
            pos[1],
            pos[0] * math.sin(t) + pos[2] * math.cos(t),
        ])
        cam_cfg["position"] = list(donuk + np.array(cam_cfg["look_at"]))
    sahne = Sahne(cfg)
    rng = np.random.default_rng(cekirdek)
    kamera = Kamera(cam_cfg, W, H, rng)
    return render(sahne, kamera, cekirdek, spp, derinlik), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--turntable", type=int, default=0, help="kare sayisi (GIF)")
    args = ap.parse_args()

    if args.preview:
        img, cfg = yukle_ve_render(args.scene, spp=12, en=320, boy=180)
        out = cfg["render"]["output"].replace(".png", "_preview.png")
        png_yaz(out, aces(img))
        print(f"== BİTTİ -> {out}")
        return

    if args.turntable:
        kareler = []
        cfg = json.load(open(args.scene, encoding="utf-8"))
        for i in range(args.turntable):
            aci = 360 * i / args.turntable
            print(f"turntable {i + 1}/{args.turntable} (aci {aci:.0f})")
            img, _ = yukle_ve_render(args.scene, spp=24, en=480, boy=270, aci=aci)
            kareler.append(img)
        out = cfg["render"]["output"].replace(".png", "_turntable.gif")
        tmp = out + ".frames"
        os.makedirs(tmp, exist_ok=True)
        for i, img in enumerate(kareler):
            png_yaz(f"{tmp}/f{i:03d}.png", aces(img))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                        "-i", f"{tmp}/f%03d.png", "-vf",
                        "palettegen=max_colors=256", f"{tmp}/pal.png"], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                        "-i", f"{tmp}/f%03d.png", "-i", f"{tmp}/pal.png",
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
                        "-loop", "0", out], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"== BİTTİ -> {out}")
        return

    img, cfg = yukle_ve_render(args.scene)
    out = cfg["render"]["output"]
    png_yaz(out, aces(img))
    print(f"== BİTTİ -> {out}")


if __name__ == "__main__":
    main()
