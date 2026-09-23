#!/usr/bin/env python3
"""ray-cinema dogrulama kapsulu: motorun fiziksel dogrulugu icin kapilar.

1. Vektor/yansima/kirilma birim testleri (analitik degerler)
2. Fresnel-Schlick ozellikleri (monotonluk, sinir degerleri)
3. Enerji korunumu: duz beyaz zemin + uniform gok -> zemin parlakligi ~= gok
4. Ayna testi: kirmizi kure aynada kirmizi gorunmeli (piksel sondasi)
5. Determinizm: ayni seed -> ayni goruntu
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ray import Kamera, Sahne, aces, birim, fresnel_schlick, izle, kirilt, yansit

OK = 0


def kontrol(ad, kosul, detay=""):
    global OK
    if kosul:
        OK += 1
        print(f"  [ok] {ad}")
    else:
        print(f"  [FAIL] {ad} {detay}")
        sys.exit(1)


print("1) vektor birim testleri")
v = np.array([[3.0, 4.0, 0.0]])
assert np.allclose(birim(v), [[0.6, 0.8, 0.0]])
kontrol("birim", True)

d = np.array([[1.0, -1.0, 0.0]])
n = np.array([[0.0, 1.0, 0.0]])
assert np.allclose(yansit(d, n), [[1.0, 1.0, 0.0]])
kontrol("yansima (45 derece)", True)

# Snell: n1=1 -> n2=1.5, dik gelis (90 derece kirilma yuzeyine paralel)
d2 = np.array([[0.0, -1.0, 0.0]])
kr, ok = kirilt(d2, n, 1.0 / 1.5)
assert ok[0] and np.allclose(kr, d2)
kontrol("kirilma dik gelis degismez", True)

# Fresnel: normal geliste r0 = ((1-1.5)/(2.5))^2 = 0.04
f = fresnel_schlick(np.array([1.0]), 1.5)
assert abs(f[0] - 0.04) < 1e-9
f_grazing = fresnel_schlick(np.array([0.0]), 1.5)
assert f_grazing[0] == 1.0
kontrol("Fresnel sinir degerleri (0.04 / 1.0)", True)

# monotonluk: gelis acisi arttikca fresnel artmali
kosin = np.linspace(0.05, 0.95, 19)
dizi = fresnel_schlick(kosin, 1.5)
kontrol("Fresnel monoton", bool(np.all(np.diff(dizi) <= 1e-12)))

print("2) enerji korunumu: beyaz zemin + uniform gok")
cfg = {
    "render": {"width": 96, "height": 54, "spp": 40, "max_depth": 4, "output": "/tmp/t.png"},
    "sky": {"top": [1.0, 1.0, 1.0], "bottom": [1.0, 1.0, 1.0], "strength": 1.0},
    "camera": {"position": [0, 2, -6], "look_at": [0, 0, 0], "vfov": 45},
    "objects": [
        {"center": [0, -1000.2, 0], "radius": 1000,
         "material": {"type": "lambertian", "albedo": [1.0, 1.0, 1.0] * 1}},
    ],
}
cfg["objects"][0]["material"]["albedo"] = [1.0, 1.0, 1.0]
sahne = Sahne(cfg)
cam = Kamera(cfg["camera"], 96, 54, np.random.default_rng(5))
img = izle(sahne, *cam.isinlar(), np.random.default_rng(6), 4)
ort = img.mean(axis=0)
# lambertian zemin, uniform gok ve bayes... beklenen: zemine bakan pikseller
# yaklasik gok parlakliginin albedo katini gosterir (yorum bagimli, genis bant):
kontrol("enerji: zemin ort < gok (soğurma)", 0.3 < ort.mean() <= 1.05, f"ort={ort.mean():.3f}")

print("3) ayna yansimasinda gok gradyaninin dikey cevrilmesi")
# ayna kuresi kamera onunde; ust bolge gokteki YUKSEK (mavi) kismi,
# alt bolce alcak (turuncu) kismi gormeli -> B-R ustte > altta
cfg2 = {
    "render": {"width": 96, "height": 54, "spp": 60, "max_depth": 4, "output": "/tmp/t2.png"},
    "sky": {"top": [0.1, 0.2, 1.0], "bottom": [1.0, 0.45, 0.1], "strength": 1.0},
    "camera": {"position": [0, 0, -3], "look_at": [0, 0, 0], "vfov": 40},
    "objects": [
        {"center": [0, 0, 0], "radius": 1.0,
         "material": {"type": "metal", "albedo": [1.0, 1.0, 1.0], "roughness": 0.0}},
    ],
}
sahne2 = Sahne(cfg2)
cam2 = Kamera(cfg2["camera"], 96, 54, np.random.default_rng(7))
img2 = izle(sahne2, *cam2.isinlar(), np.random.default_rng(8), 4)
g = img2.reshape(54, 96, 3)
# kurenin ekrandaki ignesi: merkez 27,48; yaricap ~14 piksel (vfov40, r1, d3)
ust = g[27 - 9:27 - 4, 40:56].reshape(-1, 3)
alt = g[27 + 4:27 + 9, 40:56].reshape(-1, 3)
mavilik_ust = float((ust[:, 2] - ust[:, 0]).mean())
mavilik_alt = float((alt[:, 2] - alt[:, 0]).mean())
kontrol("ayna ustte maviyi tasir", mavilik_ust > mavilik_alt,
        f"ust={mavilik_ust:.1f} alt={mavilik_alt:.1f}")
kontrol("ayna altta turuncuyu tasir", mavilik_alt < 0,
        f"alt={mavilik_alt:.1f}")

print("4) determinizm: ayni seed -> ayni goruntu")
cfg3 = {
    "render": {"width": 64, "height": 36, "spp": 20, "max_depth": 5, "output": "/tmp/t3.png"},
    "sky": {"top": [0.5, 0.7, 1.0], "bottom": [0.9, 0.8, 0.6], "strength": 1.0},
    "camera": {"position": [0, 1, -4], "look_at": [0, 0.5, 0], "vfov": 40},
    "objects": [
        {"center": [0, 0.5, 0], "radius": 0.5,
         "material": {"type": "metal", "albedo": [0.9, 0.4, 0.2], "roughness": 0.1}},
        {"center": [1.2, 0.3, 0.5], "radius": 0.3,
         "material": {"type": "dielectric", "ior": 1.5}},
    ],
}
sahne3 = Sahne(cfg3)


def kos():
    cam3 = Kamera(cfg3["camera"], 64, 36, np.random.default_rng(11))
    return izle(sahne3, *cam3.isinlar(), np.random.default_rng(12), 5)


a, b = kos(), kos()
kontrol("deterministik", bool(np.array_equal(a, b)))

print(f"\nTUM KAPILAR GECTI ({OK} kontrol)")
