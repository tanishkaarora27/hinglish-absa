# Labelling Guidelines — Hinglish Quick-Commerce Reviews

Goal: for each review, mark (a) which **aspects** are mentioned, (b) the **sentiment** towards each mentioned aspect, and (c) whether the review is **sarcastic**. Target: 1,500–3,000 reviews; two annotators on at least 200 shared reviews so you can report Cohen's kappa (`prepare_labelled_data.py --sheet2`).

Open `data/labelling_sheet.csv` in Google Sheets / Excel. A `?` in an aspect column is the rule-based extractor's guess — replace it with `pos`/`neg`/`neu`, or clear it if wrong. Add aspects it missed.

## 1. Aspect categories (choose all that apply)
| Column | Covers | Typical trigger words |
|---|---|---|
| `delivery_speed` | how fast/late the order arrived, waiting, ETA promises | late, der, jaldi, 10 min, ghanta, on time, wait |
| `product_quality` | freshness, expiry, damage, taste, wrong/missing items, quantity, fake products | fresh, taza, sada, expired, kharab, galat item, missing |
| `packaging` | packet/box/seal/bag condition, leaks, spills, cold-chain | packing, leak, seal, tuta, box, bag |
| `price` | prices, MRP mismatch, fees (handling/surge/delivery), discounts, coupons | mehnga, sasta, charges, fee, offer, loot |
| `delivery_partner` | the rider's behaviour, communication, handover | delivery boy, bhaiya, rude, polite, call, gate |
| `app_support` | app bugs/UI/payment/login/tracking/cancellation, customer care, refunds | app, crash, OTP, refund, customer care, chat, cancel |
| `general` | overall opinion **only when no specific aspect is named** ("bakwas app" → `app_support`, not general; "worst experience" → `general`) | — |

Rules
* Label an aspect only if the review actually talks about it. Do not infer (a 1-star with just "bad" → `general: neg`).
* One review can have several aspects; each gets its own sentiment.
* "Order" alone is not an aspect. "Order late aaya" → delivery_speed. "Order galat aaya" → product_quality.
* Brand comparisons ("Zepto se sasta") → the aspect being compared (price).

## 2. Sentiment per aspect: `pos` / `neg` / `neu`
* `pos` – clearly satisfied with that aspect.
* `neg` – dissatisfied / complaining about that aspect.
* `neu` – mentioned without a clear opinion, or explicitly "okay/average/normal" ("delivery 20 min, theek hai", "prices market jaise hi").
* Mixed within ONE aspect ("packing thodi dented but theek thi") → pick the final verdict the reviewer lands on; if truly balanced → `neu`.
* Use the **intended** meaning, not the surface words: a sarcastic "great delivery, 2 ghante late" is `delivery_speed: neg`. (Sarcasm is flagged separately.)
* Negation flips: "fresh nahi tha" → neg. "koi problem nahi" → pos.

## 3. Sarcasm flag: `1` / `0`
Mark `1` when the literal wording says the opposite of the intended meaning, usually mock praise or mock thanks for a bad experience.
* Strong signals: praise words + bad event ("wah", "great job", "thank you so much", "genius", 👏 🙃 😂 with a complaint), quotation marks around a promise ("'10 minute' delivery"), exaggerated congratulation, a 1-star rating with "amazing service".
* Not sarcasm: plain anger ("bakwas service"), humour without reversal, polite negative feedback, a genuine compliment.
* When unsure, mark `0` and write the doubt in `notes`; sarcasm is rare (expect ~5–15%) and ambiguous, so err on the side of `0`.

## 4. Other rules
* Keep the text exactly as scraped (do not fix spelling — the model must learn real spelling noise).
* Skip (leave all columns blank) reviews that are not about the service at all (spam, "nice", single emoji, pure Devanagari if you decided on Roman script only, pure English with no Hindi word — the sheet already filters most).
* Do not label more than ~150 reviews in one sitting; agreement drops when tired.
* Resolve disagreements between annotators by discussion and write the decision here as a new rule.

## 5. Worked examples
| Review | Aspects → sentiment | Sarcastic |
|---|---|---|
| "Delivery 8 min me aa gayi, bahut fast" | delivery_speed: pos | 0 |
| "Doodh expired tha, packet khula hua tha" | product_quality: neg, packaging: neg | 0 |
| "Wah bhai kya baat hai, 10 min bola aur 2 ghante lag gaye 👏" | delivery_speed: neg | 1 |
| "Price thoda mehnga hai but delivery fast hai" | price: neg, delivery_speed: pos | 0 |
| "Delivery time normal tha, 20 minute ke aas paas" | delivery_speed: neu | 0 |
| "Theek thaak hai, kuch khaas nahi" | general: neu | 0 |
| "Thank you Zepto, fruits me se keede nikle, kya quality hai 👌" | product_quality: neg | 1 |
