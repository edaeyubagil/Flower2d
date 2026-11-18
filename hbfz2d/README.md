# hbfz2d — 2B profil ve 3B segment fay modeli (GPS + InSAR)

Flower2d'deki fiziğe (vida dislokasyonu / Segall formülleri) dayanan, uzlaşımları testlerle
kilitlenmiş küçük bir kod; 3B kısım Okada (1985) dikdörtgen dislokasyonlarını kullanır. Hedef:
düşey doğrultu atımlı faylarda **kayma hızı** ve **kilitlenme derinliği**. İlk uygulama:
Havran–Balıkesir Fay Zonu.

## Uzlaşımlar (hepsi `tests/` ile kilitli)

| Büyüklük | Tanım |
|---|---|
| `strike` | Fay doğrultusu azimutu, kuzeyden saat yönünde (herhangi bir değer) |
| `s`, `n` | Doğrultu birim vektörü; `n` = doğrultuya bakarken **sol** taraf |
| `x`, `y` | `s` ve `n` yönündeki mesafeler (km), merkezde aeqd izdüşüm |
| `V` | Kayma hızı, **V > 0 sağ-yanal** (Flower2d: `ss = -V`; Okada `U1 = -V`) |
| `D` | Kilitlenme derinliği (km) |
| `v_par(y)` | 2B: `Σ (V_i/π)·atan((y − y0_i)/D_i)` |
| LOS | Yer→uydu birim vektörüyle iç çarpım, **pozitif = uyduya doğru** |
| `heading` | **Uçuş yönü** azimutu (sağa bakan SAR); Flower2d'nin `av_heading`'i = uçuş + 90° |
| `incidence` | Dikeyden geliş açısı |
| `b_perp` | Bölgesel `dv_perp/dy` (mm/yıl/km), her modelde var |
| `Lxx`, `Lyx`, `Lxy` | İsteğe bağlı bölgesel `dv_par/dx`, `dv_perp/dx`, `dv_par/dy` (mm/yıl/km) |
| `omega` | İsteğe bağlı rijit dönme: `v_par = −ω·y`, `v_perp = +ω·x` (ω > 0 saat yönü tersine) |

Ters yazılmış doğrultu (ör. 77 ile 257) aynı fiziksel fay için aynı D–K hız alanını verir.

## 3B kurgular (`profil2d/model3d.py`)

Her fayın segment izleri tek bir ize birleştirilir (paralel kollar ortalanır); D tüm faylarda ortak.

| Kurgu | Kayma | Not |
|---|---|---|
| `derin` | yalnız haritalı iz boyunca, D'nin altında | bölgesel hareket fay uçlarının ötesine taşınmaz |
| `blok` | iz + uzantı boyunca, D'nin altında | = blok hareketi − 0..D geri kayma; uzantı → ∞ iken 2B limit |
| `blok_krip` | blok hareketi − yalnız haritalı izde 0..D geri kayma | uzantılar yüzeyde kripli |

## Kullanım

```bash
python -m pytest tests                       # uzlaşım + Flower2d çapraz kontrol testleri
python scripts/hazirla_gps.py --org VEL.org --faults FAYLAR.geojson \
       --haric local/gps_haric.txt --out local/gps_profil.csv --fig local/gps_profil.png
python scripts/ters_coz_3b.py --gps local/gps_profil.csv --insar local/insar_asc_profil.csv \
       --insar local/insar_dsc_profil.csv --faults FAYLAR.geojson --kurgu blok --uzanti 1e6 \
       --gradyan kaymasiz --out local/ters3b
```

`local/` git dışıdır: kullanıcı verisi ve veriden türetilen dosyalar (istasyon listeleri dahil)
repoya girmez.

Gereksinimler: Python ≥ 3.10, numpy, scipy, pyproj, h5py, emcee, matplotlib, pytest.
