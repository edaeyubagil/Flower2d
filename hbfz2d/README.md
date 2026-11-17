# hbfz2d — 2B profil fay modeli (GPS + InSAR)

Flower2d'deki fiziğe (vida dislokasyonu / Segall formülleri) dayanan, uzlaşımları testlerle
kilitlenmiş küçük bir kod. Hedef: düşey doğrultu atımlı faylarda **kayma hızı** ve
**kilitlenme derinliği**. İlk uygulama: Havran–Balıkesir Fay Zonu.

## Uzlaşımlar (hepsi `tests/` ile kilitli)

| Büyüklük | Tanım |
|---|---|
| `strike` | Fay doğrultusu azimutu, kuzeyden saat yönünde (herhangi bir değer) |
| `s`, `n` | Doğrultu birim vektörü; `n` = doğrultuya bakarken **sol** taraf |
| `x`, `y` | `s` ve `n` yönündeki mesafeler (km), merkezde aeqd izdüşüm |
| `V` | Kayma hızı, **V > 0 sağ-yanal** (Flower2d: `ss = -V`) |
| `D` | Kilitlenme derinliği (km) |
| `v_par(y)` | `Σ (V_i/π)·atan((y − y0_i)/D_i)` |
| LOS | Yer→uydu birim vektörüyle iç çarpım, **pozitif = uyduya doğru** |
| `heading` | **Uçuş yönü** azimutu (sağa bakan SAR); Flower2d'nin `av_heading`'i = uçuş + 90° |
| `incidence` | Dikeyden geliş açısı |

Ters yazılmış doğrultu (ör. 77 ile 257) aynı fiziksel fay için aynı D–K hız alanını verir.

## Kullanım

```bash
python -m pytest tests                       # uzlaşım + Flower2d çapraz kontrol testleri
python scripts/hazirla_gps.py --org VEL.org --faults FAYLAR.geojson \
       --haric local/gps_haric.txt --out local/gps_profil.csv --fig local/gps_profil.png
```

`local/` git dışıdır: kullanıcı verisi ve veriden türetilen dosyalar (istasyon listeleri dahil)
repoya girmez.

Gereksinimler: Python ≥ 3.10, numpy, pyproj, matplotlib, pytest (inversiyon için PyMC 5).
