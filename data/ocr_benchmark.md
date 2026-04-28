# OCR Technique Benchmark
_Generated: 2026-04-28 11:43_

## Overview
Tests every combination of **Segmentation** × **Preprocessing** across all 5 test images.
**Score** = avg_char_confidence × completeness (fraction of lines decoded).

---

## 🖼️ download.jpg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| projection + morph_close | 2 | 2 | 0.552 | 1.00 | **0.552** | 1336.8 | ښې / کې |
| projection + otsu | 2 | 2 | 0.488 | 1.00 | **0.488** | 651.7 | چېېټت / دبا ب ب ب بد ښې |
| projection + clahe+adaptive | 2 | 2 | 0.455 | 1.00 | **0.455** | 1127.8 | تنيشتتيممتي / ک |
| yolo + clahe+adaptive | 1 | 1 | 0.330 | 1.00 | **0.330** | 1878.1 | اې ې ه ا ا يف |
| yolo + adaptive_21_10 | 1 | 1 | 0.314 | 1.00 | **0.314** | 1665.5 | بجچې م ه ې ي به ول ے |
| yolo + adaptive_51_20 | 1 | 1 | 0.300 | 1.00 | **0.300** | 1604.8 | ’سړه اې دې را و اي |
| yolo + otsu | 1 | 1 | 0.298 | 1.00 | **0.298** | 1728.3 | ﻿اعي همه ا ي رې ا م ا ملي |
| yolo + morph_close | 1 | 1 | 0.295 | 1.00 | **0.295** | 1945.4 | داچي ه ا و رې د ا غا |
| full_image + adaptive_51_20 | 1 | 1 | 0.224 | 1.00 | **0.224** | 128.4 | د ب ب ب ک |
| full_image + morph_close | 1 | 1 | 0.213 | 1.00 | **0.213** | 365.6 | ک |
| full_image + adaptive_21_10 | 1 | 1 | 0.197 | 1.00 | **0.197** | 191.3 | داب ب ب ب ک |
| full_image + otsu | 1 | 1 | 0.186 | 1.00 | **0.186** | 232.0 | س ب ب ب ر |
| projection + adaptive_51_20 | 2 | 1 | 0.265 | 0.50 | **0.133** | 219.5 | ب ب ب ب ب تې |
| projection + adaptive_21_10 | 2 | 1 | 0.258 | 0.50 | **0.129** | 446.3 | ب ب ب ب ب ب کې |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 277.2 | _(empty)_ |
| full_image + clahe+adaptive | 1 | 0 | 0.000 | 0.00 | **0.000** | 326.1 | _(empty)_ |
| yolo + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 1807.2 | _(empty)_ |
| projection + raw_gray | 2 | 0 | 0.000 | 0.00 | **0.000** | 883.0 | _(empty)_ |

## 🖼️ images.jpg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| full_image + adaptive_51_20 | 1 | 0 | 0.000 | 0.00 | **0.000** | 33.4 | _(empty)_ |
| full_image + adaptive_21_10 | 1 | 0 | 0.000 | 0.00 | **0.000** | 59.0 | _(empty)_ |
| full_image + otsu | 1 | 0 | 0.000 | 0.00 | **0.000** | 82.8 | _(empty)_ |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 107.6 | _(empty)_ |
| full_image + clahe+adaptive | 1 | 0 | 0.000 | 0.00 | **0.000** | 134.7 | _(empty)_ |
| full_image + morph_close | 1 | 0 | 0.000 | 0.00 | **0.000** | 160.3 | _(empty)_ |
| yolo + adaptive_51_20 | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.0 | _(empty)_ |
| yolo + adaptive_21_10 | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.0 | _(empty)_ |
| yolo + otsu | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.1 | _(empty)_ |
| yolo + raw_gray | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.1 | _(empty)_ |
| yolo + clahe+adaptive | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.1 | _(empty)_ |
| yolo + morph_close | 0 | 0 | 0.000 | 0.00 | **0.000** | 208.1 | _(empty)_ |
| projection + adaptive_51_20 | 1 | 0 | 0.000 | 0.00 | **0.000** | 30.0 | _(empty)_ |
| projection + adaptive_21_10 | 1 | 0 | 0.000 | 0.00 | **0.000** | 56.6 | _(empty)_ |
| projection + otsu | 1 | 0 | 0.000 | 0.00 | **0.000** | 82.4 | _(empty)_ |
| projection + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 112.0 | _(empty)_ |
| projection + clahe+adaptive | 1 | 0 | 0.000 | 0.00 | **0.000** | 141.5 | _(empty)_ |
| projection + morph_close | 1 | 0 | 0.000 | 0.00 | **0.000** | 166.0 | _(empty)_ |

## 🖼️ synthetic_multiline.jpg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| projection + otsu | 5 | 5 | 0.818 | 1.00 | **0.818** | 3726.7 | زه له علمه له مذهيه نه له دپېلې / چې دوئ کې دې ښادۍ په مراسو |
| yolo + otsu | 5 | 5 | 0.807 | 1.00 | **0.807** | 1948.1 | نهله علمه له مزهبه نه له د یينه / چې دوئې کې دې ښادي په مراس |
| projection + adaptive_51_20 | 5 | 5 | 0.773 | 1.00 | **0.773** | 1040.5 | هله عليمه له بذهپل نه په لےېلے / چې د وئ کې دې ښادۍ په مراسو |
| yolo + adaptive_51_20 | 5 | 5 | 0.758 | 1.00 | **0.758** | 710.2 | هله علمه له مهپل ڼه په در پېنظ / چې دوئې کې د ښادی په مراشو  |
| projection + morph_close | 5 | 5 | 0.758 | 1.00 | **0.758** | 6260.3 | ې  نخصمل خلملفې دپللي / چې ښدَ وئغکېک د ښا دغې پۀ مرباسو کېې |
| projection + adaptive_21_10 | 5 | 5 | 0.747 | 1.00 | **0.747** | 2780.6 | امله عیملهنله بي ذهپل نۀ له نل پېلې / چېښد وئ کې د ښادي پۀ م |
| yolo + morph_close | 5 | 5 | 0.726 | 1.00 | **0.726** | 3502.7 | ’خةه لا عليما ا هذ هبل ته له د ېنه / چې ښد وئېنکښې کښ دې پښا |
| yolo + adaptive_21_10 | 5 | 5 | 0.725 | 1.00 | **0.725** | 1400.4 | زېه له علميو له مدذهبل نۀ له در نېظ / چې دوئې کې د ښ دی۔ په  |
| projection + clahe+adaptive | 5 | 5 | 0.611 | 1.00 | **0.611** | 5289.2 | ل لمملی تَال  لېسل پهسپ کښ فپې لي / ن کې ې د ا خ د اې چ ا ا  |
| yolo + clahe+adaptive | 5 | 5 | 0.589 | 1.00 | **0.589** | 2979.0 | پکه غکيما له يد به لم پۀ په د پب / چې شبشي و دې کې شا ر ب ي  |
| full_image + morph_close | 1 | 1 | 0.332 | 1.00 | **0.332** | 313.8 | د پ ت |
| full_image + adaptive_21_10 | 1 | 1 | 0.307 | 1.00 | **0.307** | 120.0 | د و ب ه |
| full_image + clahe+adaptive | 1 | 1 | 0.291 | 1.00 | **0.291** | 260.6 | د و پ |
| full_image + otsu | 1 | 1 | 0.275 | 1.00 | **0.275** | 157.8 | دا ـ پ ب |
| full_image + adaptive_51_20 | 1 | 1 | 0.226 | 1.00 | **0.226** | 70.8 | د ـ ه |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 196.1 | _(empty)_ |
| yolo + raw_gray | 5 | 0 | 0.000 | 0.00 | **0.000** | 2431.2 | _(empty)_ |
| projection + raw_gray | 5 | 0 | 0.000 | 0.00 | **0.000** | 4468.8 | _(empty)_ |

## 🖼️ test_page.jpg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| yolo + adaptive_51_20 | 8 | 6 | 0.611 | 0.75 | **0.458** | 891.4 | "سنږدم / ہ / سه  ہېش استم سې |
| yolo + clahe+adaptive | 8 | 7 | 0.497 | 0.88 | **0.435** | 3469.1 | ېښبے پال خې / پکار / اپل رٹ اہسب) د سې |
| projection + morph_close | 2 | 1 | 0.821 | 0.50 | **0.411** | 625.4 | پلار |
| projection + adaptive_51_20 | 2 | 2 | 0.401 | 1.00 | **0.401** | 116.3 | پداه / ښ |
| yolo + adaptive_21_10 | 8 | 6 | 0.533 | 0.75 | **0.399** | 1536.7 | "پې نمي / پمه / سہېوش اس سې |
| yolo + morph_close | 8 | 6 | 0.501 | 0.75 | **0.376** | 4119.3 | په لال مع / پکار / ۀ س پو ے امب، سې |
| yolo + otsu | 8 | 6 | 0.486 | 0.75 | **0.364** | 2168.3 | ۔پے پلاړ / ےپلکے / پ ٣ ء ب اس |
| projection + otsu | 2 | 1 | 0.651 | 0.50 | **0.326** | 309.8 | پلا |
| projection + adaptive_21_10 | 2 | 1 | 0.532 | 0.50 | **0.266** | 214.3 | پنا |
| projection + clahe+adaptive | 2 | 1 | 0.515 | 0.50 | **0.258** | 497.0 | ب پلاه |
| full_image + adaptive_51_20 | 1 | 0 | 0.000 | 0.00 | **0.000** | 33.8 | _(empty)_ |
| full_image + adaptive_21_10 | 1 | 0 | 0.000 | 0.00 | **0.000** | 61.6 | _(empty)_ |
| full_image + otsu | 1 | 0 | 0.000 | 0.00 | **0.000** | 86.2 | _(empty)_ |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 112.4 | _(empty)_ |
| full_image + clahe+adaptive | 1 | 0 | 0.000 | 0.00 | **0.000** | 141.6 | _(empty)_ |
| full_image + morph_close | 1 | 0 | 0.000 | 0.00 | **0.000** | 167.0 | _(empty)_ |
| yolo + raw_gray | 8 | 0 | 0.000 | 0.00 | **0.000** | 2820.6 | _(empty)_ |
| projection + raw_gray | 2 | 0 | 0.000 | 0.00 | **0.000** | 399.8 | _(empty)_ |

## 🖼️ WhatsApp Image 2026-04-27 at 1.04.43 AM.jpeg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| yolo + adaptive_51_20 | 1 | 1 | 0.907 | 1.00 | **0.907** | 243.5 | د پرهې وړمې وول کړو |
| projection + adaptive_51_20 | 1 | 1 | 0.889 | 1.00 | **0.889** | 163.3 | د پوهې ورمې وول کړو |
| projection + adaptive_21_10 | 1 | 1 | 0.883 | 1.00 | **0.883** | 304.1 | د پوهې ورمې عؤل کړخ |
| yolo + morph_close | 1 | 1 | 0.875 | 1.00 | **0.875** | 939.1 | د پوهې وړمې وؤل کړف |
| full_image + adaptive_51_20 | 1 | 1 | 0.870 | 1.00 | **0.870** | 144.6 | د پوهې وړمې وول کړو. |
| full_image + morph_close | 1 | 1 | 0.870 | 1.00 | **0.870** | 884.0 | د ېوهې وړمې وؤل کړف |
| yolo + otsu | 1 | 1 | 0.859 | 1.00 | **0.859** | 525.0 | د پوهې ور مې وول کړو |
| projection + otsu | 1 | 1 | 0.846 | 1.00 | **0.846** | 440.4 | د پوهې ور مې ېډول کړو |
| projection + morph_close | 1 | 1 | 0.838 | 1.00 | **0.838** | 910.0 | د يوهې وړمې فؤل کړف |
| full_image + otsu | 1 | 1 | 0.821 | 1.00 | **0.821** | 440.7 | د پوهې كور مې عډل کړو |
| yolo + adaptive_21_10 | 1 | 1 | 0.821 | 1.00 | **0.821** | 384.9 | د پرهې وړمې وول کړخ |
| full_image + adaptive_21_10 | 1 | 1 | 0.814 | 1.00 | **0.814** | 283.0 | د پو هې وړمې عول کړد |
| yolo + clahe+adaptive | 1 | 1 | 0.483 | 1.00 | **0.483** | 802.9 | ې د تقونهې وژي في څ ؤۀڅول کړ شادي ب ش شې ي دد د |
| projection + clahe+adaptive | 1 | 1 | 0.475 | 1.00 | **0.475** | 758.4 | شپې ېو همې و ژيقې د ؤۀڅل کړ شايت د ش ي ش ب شېب شد ش د تش |
| full_image + clahe+adaptive | 1 | 1 | 0.474 | 1.00 | **0.474** | 738.9 | ت  د يو تهي و ثي قضنې څي فرڅي بد کړ ات  دش د ب ا د د ب ې ت |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 594.0 | _(empty)_ |
| yolo + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 662.5 | _(empty)_ |
| projection + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 591.2 | _(empty)_ |

## 🖼️ WhatsApp Image 2026-04-27 at 1.35.15 AM.jpeg

| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |
|---|---|---|---|---|---|---|---|
| yolo + morph_close | 1 | 1 | 0.799 | 1.00 | **0.799** | 614.1 | هوارز خمونه مې په زړه دې هړ په شمه |
| yolo + adaptive_51_20 | 1 | 1 | 0.782 | 1.00 | **0.782** | 242.6 | هواړ د خمونه مې په زړه دې ه په شمه |
| yolo + otsu | 1 | 1 | 0.757 | 1.00 | **0.757** | 388.7 | خوار زخمونه مې په زړه دې هم په شمه |
| yolo + adaptive_21_10 | 1 | 1 | 0.736 | 1.00 | **0.736** | 317.8 | خوا د خصونه مې په زړ دې هم په شمه |
| yolo + clahe+adaptive | 1 | 1 | 0.669 | 1.00 | **0.669** | 546.7 | خوا د ځمونه هغې پک و دې هې په شمه |
| full_image + morph_close | 1 | 1 | 0.232 | 1.00 | **0.232** | 255.2 | ای د |
| full_image + clahe+adaptive | 1 | 1 | 0.228 | 1.00 | **0.228** | 216.5 | ’شبوه ا ايې |
| full_image + adaptive_51_20 | 1 | 1 | 0.224 | 1.00 | **0.224** | 44.5 | ای د ب ب ې |
| full_image + adaptive_21_10 | 1 | 1 | 0.223 | 1.00 | **0.223** | 84.1 | ای د ب ب د |
| full_image + otsu | 1 | 1 | 0.221 | 1.00 | **0.221** | 121.4 | ’ بو ې يې |
| projection + adaptive_21_10 | 2 | 1 | 0.369 | 0.50 | **0.185** | 293.5 | ا ه په ب ي |
| projection + morph_close | 2 | 1 | 0.366 | 0.50 | **0.183** | 869.2 | ’روا سد ه دب ې ش هې |
| projection + adaptive_51_20 | 2 | 1 | 0.328 | 0.50 | **0.164** | 157.2 | ا  په ب ب ي ي |
| projection + otsu | 2 | 1 | 0.297 | 0.50 | **0.148** | 441.6 | ’ړا ها د ه ده |
| projection + clahe+adaptive | 2 | 1 | 0.288 | 0.50 | **0.144** | 716.4 | ا  ه ب ب د اۀي |
| full_image + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 159.6 | _(empty)_ |
| yolo + raw_gray | 1 | 0 | 0.000 | 0.00 | **0.000** | 474.3 | _(empty)_ |
| projection + raw_gray | 2 | 0 | 0.000 | 0.00 | **0.000** | 578.2 | _(empty)_ |

---
## 🏆 Best Technique Per Image

| Image | Best Technique | Score | Text Preview |
|---|---|---|---|
| download.jpg | projection + morph_close | 0.552 | ښې / کې |
| images.jpg | full_image + adaptive_51_20 | 0.000 | _(empty)_ |
| synthetic_multiline.jpg | projection + otsu | 0.818 | زه له علمه له مذهيه نه له دپېلې / چې دوئ کې دې ښادۍ په مراسوکې ګډون کړي لکه ا ښې |
| test_page.jpg | yolo + adaptive_51_20 | 0.458 | "سنږدم / ہ / سه  ہېش استم سې |
| WhatsApp Image 2026-04-27 at 1.04.43 AM.jpeg | yolo + adaptive_51_20 | 0.907 | د پرهې وړمې وول کړو |
| WhatsApp Image 2026-04-27 at 1.35.15 AM.jpeg | yolo + morph_close | 0.799 | هوارز خمونه مې په زړه دې هړ په شمه |

---
## ⚙️ Configuration Used

- Model: `crnn_pashto.pth`
- Vocab size: 181
- Device: cpu
- Test images: ['download.jpg', 'images.jpg', 'synthetic_multiline.jpg', 'test_page.jpg', 'WhatsApp Image 2026-04-27 at 1.04.43 AM.jpeg', 'WhatsApp Image 2026-04-27 at 1.35.15 AM.jpeg']