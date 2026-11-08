"""
Builds data/demo/reviews.csv and data/demo/aspect_labels.csv.

*** THESE REVIEWS ARE SYNTHETIC (written by hand for a smoke test). ***
They are NOT real app reviews and metrics computed on them are NOT reportable.
They exist only so the notebook and demo app run end-to-end before you have
collected and labelled real Blinkit / Zepto / Instamart reviews.

Format produced (same as what scripts/prepare_labelled_data.py produces from
your labelling sheet):
  reviews.csv        : review_id, text, rating, sarcastic, source
  aspect_labels.csv  : review_id, aspect, sentiment
"""
from pathlib import Path
import csv

# (text, rating 1-5, sarcastic 0/1, [(aspect, sentiment), ...])
D = [
 ("Delivery 8 min me aa gayi, bahut fast service", 5, 0, [("delivery_speed","positive")]),
 ("Doodh expired tha, packet khula hua tha", 1, 0, [("product_quality","negative"),("packaging","negative")]),
 ("Wah bhai kya baat hai, 10 min bola aur 2 ghante lag gaye 👏 great job", 1, 1, [("delivery_speed","negative")]),
 ("Sabzi fresh thi aur price bhi theek tha", 4, 0, [("product_quality","positive"),("price","positive")]),
 ("App baar baar crash ho raha hai, payment fail ho gaya", 1, 0, [("app_support","negative")]),
 ("Delivery boy bahut polite tha, thanks bhaiya", 5, 0, [("delivery_partner","positive")]),
 ("Price thoda mehnga hai but delivery fast hai", 3, 0, [("price","negative"),("delivery_speed","positive")]),
 ("Fruits sade hue the, aam bilkul kharab", 1, 0, [("product_quality","negative")]),
 ("Packaging bahut acchi thi, kuch bhi leak nahi hua", 5, 0, [("packaging","positive")]),
 ("Customer care ne koi reply nahi diya, refund abhi tak nahi aaya", 1, 0, [("app_support","negative")]),
 ("Great, mera order kisi aur ko deliver kar diya, bahut badhiya service 🙂", 1, 1, [("delivery_speed","negative")]),
 ("Delivery time normal tha, 20 minute ke aas paas", 3, 0, [("delivery_speed","neutral")]),
 ("Bread taza tha, eggs bhi theek the", 4, 0, [("product_quality","positive")]),
 ("Handling fee aur surge fee lagakar bill double ho gaya, loot machi hai", 2, 0, [("price","negative")]),
 ("Delivery partner rude tha, gate pe chilla raha tha", 1, 0, [("delivery_partner","negative")]),
 ("App ka UI simple hai, order karna easy hai", 4, 0, [("app_support","positive")]),
 ("Wow superb, expired chips bhej diye, thank you so much Blinkit", 1, 1, [("product_quality","negative")]),
 ("Bottle leak ho gayi thi, pura bag geela", 2, 0, [("packaging","negative")]),
 ("Prices average hain, na sasta na mehnga", 3, 0, [("price","neutral")]),
 ("Rider ne call karke address confirm kiya, nice guy", 5, 0, [("delivery_partner","positive")]),
 ("Order 45 minute late aaya, raat ke 11 baje", 1, 0, [("delivery_speed","negative")]),
 ("Refund 2 din me aa gaya, support helpful tha", 4, 0, [("app_support","positive")]),
 ("Kya baat hai, 3 baar cancel hua order, genius app 👏👏", 1, 1, [("app_support","negative")]),
 ("Paneer ki quality mast thi", 5, 0, [("product_quality","positive")]),
 ("Packet thoda dented tha but andar saman theek tha", 3, 0, [("packaging","neutral"),("product_quality","positive")]),
 ("Sasta aur accha, paisa vasool", 5, 0, [("price","positive")]),
 ("Delivery wale bhaiya ne doorstep tak laaya, thank you", 5, 0, [("delivery_partner","positive")]),
 ("App me search kaam nahi karta, bahut irritating", 2, 0, [("app_support","negative")]),
 ("Delivery toh 10 min me ho gayi lekin doodh phat gaya tha", 2, 0, [("delivery_speed","positive"),("product_quality","negative")]),
 ("Bahut accha experience overall", 5, 0, [("general","positive")]),
 ("Bakwas service, kabhi order mat karna", 1, 0, [("general","negative")]),
 ("Theek thaak hai, kuch khaas nahi", 3, 0, [("general","neutral")]),
 ("Wah, 1 ghanta wait karwa ke bola out of stock, amazing 😂", 1, 1, [("delivery_speed","negative"),("app_support","negative")]),
 ("Fresh vegetables, timely delivery, no complaints", 5, 0, [("product_quality","positive"),("delivery_speed","positive")]),
 ("Delivery fee 30 rupees extra le liye, pehle free tha", 2, 0, [("price","negative")]),
 ("Bhaiya late aaye but polite the", 3, 0, [("delivery_speed","negative"),("delivery_partner","positive")]),
 ("OTP nahi aa raha login ke liye, app useless", 1, 0, [("app_support","negative")]),
 ("Packaging normal thi, plastic bag me aaya", 3, 0, [("packaging","neutral")]),
 ("Thank you Zepto, fruits me se keede nikle, kya quality hai 👌", 1, 1, [("product_quality","negative")]),
 ("Milk packet seal tha, fresh tha, good", 4, 0, [("packaging","positive"),("product_quality","positive")]),
 ("Delivery slow thi aaj, 35 minute", 2, 0, [("delivery_speed","negative")]),
 ("Discount coupon kaam kar gaya, achha offer", 4, 0, [("price","positive")]),
 ("Rider ne galat gate pe chhod diya aur chala gaya", 1, 0, [("delivery_partner","negative")]),
 ("Tracking map accurate hai, live location dikhta hai", 4, 0, [("app_support","positive")]),
 ("Great job, sabzi ki jagah mujhe kisi aur ka order mila, best app 🙃", 1, 1, [("product_quality","negative"),("delivery_speed","negative")]),
 ("Eggs tute hue the, box crushed tha", 1, 0, [("product_quality","negative"),("packaging","negative")]),
 ("Price Blinkit se kam hai yahan", 4, 0, [("price","positive")]),
 ("Delivery boy ne 2 baar call kiya, thoda confuse tha but theek hai", 3, 0, [("delivery_partner","neutral")]),
 ("Payment UPI se smooth ho gaya", 4, 0, [("app_support","positive")]),
 ("Delivery on time, as expected", 4, 0, [("delivery_speed","positive")]),
 ("Chips ka packet half empty tha, quantity kam", 2, 0, [("product_quality","negative")]),
 ("Bahut mehnga, MRP se zyada charge kar rahe hain", 1, 0, [("price","negative")]),
 ("Support chat bot sirf ghoomata hai, koi human nahi milta", 1, 0, [("app_support","negative")]),
 ("Wah wah, 'instant delivery' 50 minute me, kya speed hai 🚀", 1, 1, [("delivery_speed","negative")]),
 ("Curd fresh tha, taste accha tha", 5, 0, [("product_quality","positive")]),
 ("Packing average, ek bag me sab daal diya", 3, 0, [("packaging","neutral")]),
 ("Delivery partner bahut helpful tha, lift kharab thi phir bhi upar aaya", 5, 0, [("delivery_partner","positive")]),
 ("App slow hai, loading me time lagta hai", 2, 0, [("app_support","negative")]),
 ("Rates theek hain, offers bhi milte rehte hain", 4, 0, [("price","positive")]),
 ("Superb, rider ne mera order raste me kha liya lagta hai, 1 ghante se wait 😂", 1, 1, [("delivery_speed","negative"),("delivery_partner","negative")]),
 ("Order 12 min me aaya, sab kuch fresh", 5, 0, [("delivery_speed","positive"),("product_quality","positive")]),
 ("Wrong item deliver hua, replacement bhi nahi mila", 1, 0, [("product_quality","negative"),("app_support","negative")]),
 ("Delivery charge zero tha aaj, nice", 4, 0, [("price","positive")]),
 ("Bhaiya ka behaviour theek tha, normal", 3, 0, [("delivery_partner","neutral")]),
 ("Notification aata hai jab order aa jata hai, useful feature", 4, 0, [("app_support","positive")]),
 ("Amazing, bread me fungus tha, thank you for the free science experiment 🙏", 1, 1, [("product_quality","negative")]),
 ("Ice cream pighal ke aayi, packaging me ice nahi tha", 2, 0, [("product_quality","negative"),("packaging","negative")]),
 ("Delivery 15 min, theek hai", 3, 0, [("delivery_speed","neutral")]),
 ("Prices kabhi kabhi badh jaate hain raat ko", 3, 0, [("price","neutral")]),
 ("Delivery uncle bahut sweet the, bachhe ko toffee di", 5, 0, [("delivery_partner","positive")]),
 ("Cart me item add nahi ho raha, bug hai", 2, 0, [("app_support","negative")]),
 ("Fantastic, 3rd time cancel kiya order, really professional service", 1, 1, [("app_support","negative")]),
 ("Vegetables kachhe the, ripe nahi", 2, 0, [("product_quality","negative")]),
 ("Sealed packet, no damage, good packing", 5, 0, [("packaging","positive")]),
 ("Bahut sasta mila aaj, 20 percent off", 5, 0, [("price","positive")]),
 ("Rider raste me 20 minute ruka raha, map pe dikh raha tha", 2, 0, [("delivery_partner","negative"),("delivery_speed","negative")]),
 ("Customer support ne turant refund kar diya, impressed", 5, 0, [("app_support","positive")]),
 ("Kya baat hai, mehnge daam pe expired doodh, wah", 1, 1, [("price","negative"),("product_quality","negative")]),
 ("Delivery thodi late thi, 25 minute lage", 3, 0, [("delivery_speed","negative")]),
 ("Fruits ka size chhota tha but fresh the", 3, 0, [("product_quality","neutral")]),
 ("Packet open condition me aaya, seal tuta hua", 1, 0, [("packaging","negative")]),
 ("Thoda mehnga hai local shop se", 3, 0, [("price","negative")]),
 ("Delivery wala polite tha, time pe aaya", 5, 0, [("delivery_partner","positive"),("delivery_speed","positive")]),
 ("App update ke baad bahut smooth chal raha hai", 4, 0, [("app_support","positive")]),
 ("Best app ever, 2 ghante baad bola rider nahi mila, 😂😂", 1, 1, [("delivery_speed","negative")]),
 ("Paneer taza tha, doodh bhi theek", 4, 0, [("product_quality","positive")]),
 ("Packaging accha, sab alag alag bags me", 4, 0, [("packaging","positive")]),
 ("Handling charges kyu lagte hain samajh nahi aata", 2, 0, [("price","negative")]),
 ("Delivery boy ne saman gate pe rakh diya bina bataye", 2, 0, [("delivery_partner","negative")]),
 ("Chat support ne 5 min me solve kar diya", 5, 0, [("app_support","positive")]),
 ("Delivery 30 min, usual time", 3, 0, [("delivery_speed","neutral")]),
 ("Doodh ka taste ajeeb tha, lagta hai purana tha", 2, 0, [("product_quality","negative")]),
 ("Lovely, bottle leak ho ke pura order bheeg gaya, superb packing 👏", 1, 1, [("packaging","negative")]),
 ("Rates market jaise hi hain", 3, 0, [("price","neutral")]),
 ("Rider ko address samajh nahi aaya, 3 baar call", 2, 0, [("delivery_partner","negative")]),
 ("App crash hota hai checkout pe, order nahi ho pata", 1, 0, [("app_support","negative")]),
 ("Jaldi aa gaya, fresh tha, happy", 5, 0, [("delivery_speed","positive"),("product_quality","positive")]),
 ("Package ki condition theek thi, normal", 3, 0, [("packaging","neutral")]),
 ("Offers bahut acche milte hain, cashback bhi", 5, 0, [("price","positive")]),
 ("Delivery partner ne mask nahi pehna tha, careless", 2, 0, [("delivery_partner","negative")]),
 ("Refund process easy hai app me", 4, 0, [("app_support","positive")]),
 ("Wonderful, order deliver dikhaya but kuch aaya hi nahi, magic ✨", 1, 1, [("delivery_speed","negative"),("app_support","negative")]),
 ("Order sahi aaya, koi issue nahi", 4, 0, [("general","positive")]),
 ("Average service, improve karo", 3, 0, [("general","neutral")]),
 ("Worst experience, never again", 1, 0, [("general","negative")]),
 ("Sabzi me se badbu aa rahi thi, sadi hui", 1, 0, [("product_quality","negative")]),
 ("Box strong tha, kuch toota nahi", 4, 0, [("packaging","positive")]),
 ("Thoda costly but convenient", 3, 0, [("price","negative")]),
 ("Rider polite, on time, perfect", 5, 0, [("delivery_partner","positive"),("delivery_speed","positive")]),
 ("App me address save nahi hota, har baar type karo", 2, 0, [("app_support","negative")]),
 ("Haha great, 'fresh' ande 1 hafte purane the, thanks for the surprise", 1, 1, [("product_quality","negative")]),
 ("Delivery 10 min, zabardast speed", 5, 0, [("delivery_speed","positive")]),
 ("Prices kabhi kam kabhi zyada, depends", 3, 0, [("price","neutral")]),
 ("Delivery boy ne galat order diya phir sahi laaya, theek hai", 3, 0, [("delivery_partner","neutral")]),
 ("Support se baat karna mushkil hai, option hi nahi dikhta", 2, 0, [("app_support","negative")]),
 ("Fresh fruits, nice packing, fast delivery, sab mast", 5, 0, [("product_quality","positive"),("packaging","positive"),("delivery_speed","positive")]),
 ("Delivery late, saman kharab, price mehnga, pathetic", 1, 0, [("delivery_speed","negative"),("product_quality","negative"),("price","negative")]),
 ("Excellent service, 40 min wait aur phir cancel, keep it up 🙂", 1, 1, [("delivery_speed","negative"),("app_support","negative")]),
 ("Packaging me bubble wrap tha, safe aaya", 4, 0, [("packaging","positive")]),
 ("Rider ne mujhe hi daant diya late aane pe, wah", 1, 1, [("delivery_partner","negative")]),
 ("Milk 2 packet order kiya, 1 hi aaya", 2, 0, [("product_quality","negative")]),
 ("Surge fee raat ko lagta hai, otherwise theek", 3, 0, [("price","neutral")]),
 ("Order time pe aaya, bhaiya polite, item fresh", 5, 0, [("delivery_speed","positive"),("delivery_partner","positive"),("product_quality","positive")]),
 ("Login problem, OTP late aata hai", 2, 0, [("app_support","negative")]),
 ("Delivery timing theek hai, 20-25 min", 3, 0, [("delivery_speed","neutral")]),
 ("Quality acchi hai, Blinkit se better", 4, 0, [("product_quality","positive")]),
 ("Bag phata hua tha, saman gir sakta tha", 2, 0, [("packaging","negative")]),
 ("Genius, delivery fee + handling fee + surge, 100 ka saman 180 ka 👏", 1, 1, [("price","negative")]),
 ("Delivery partner ne saman sahi se handover kiya", 4, 0, [("delivery_partner","positive")]),
 ("App ka dark mode accha hai, UI clean", 4, 0, [("app_support","positive")]),
 ("Order 1 ghanta late, cold food, bakwas", 1, 0, [("delivery_speed","negative"),("product_quality","negative")]),
 ("Packing fine, product fine, price fine, 4 star", 4, 0, [("packaging","positive"),("product_quality","positive"),("price","positive")]),
 ("Rider behaviour average, na accha na bura", 3, 0, [("delivery_partner","neutral")]),
 ("Support ticket 1 hafte se pending, koi response nahi", 1, 0, [("app_support","negative")]),
 ("Sach me best, 'fresh' sabzi mujhe hi fridge se taazi lagi 😂 nahi", 1, 1, [("product_quality","negative")]),
 ("Delivery jaldi hui, 9 minute", 5, 0, [("delivery_speed","positive")]),
 ("Thoda mehnga lekin quality acchi", 4, 0, [("price","negative"),("product_quality","positive")]),
 ("Packaging perfect, cold items cold aaye", 5, 0, [("packaging","positive")]),
 ("Rider ne doorbell nahi bajayi, order bahar pada raha", 2, 0, [("delivery_partner","negative")]),
 ("App theek chal raha hai, koi crash nahi", 4, 0, [("app_support","positive")]),
 ("Normal delivery, normal packing, normal price", 3, 0, [("delivery_speed","neutral"),("packaging","neutral"),("price","neutral")]),
 ("Brilliant, 15 min bola, 90 min ke baad 'rider assigned', kya system hai", 1, 1, [("delivery_speed","negative"),("app_support","negative")]),
 ("Fresh doodh, sealed packet, time pe, sab badhiya", 5, 0, [("product_quality","positive"),("packaging","positive"),("delivery_speed","positive")]),
 ("Item missing tha order se, refund nahi mila", 1, 0, [("product_quality","negative"),("app_support","negative")]),
 ("Delivery guy friendly tha, smile ke saath diya", 5, 0, [("delivery_partner","positive")]),
 ("Price list me 50 tha, bill me 65, galat charge", 1, 0, [("price","negative")]),
 ("Packet thoda geela tha but product theek", 3, 0, [("packaging","neutral"),("product_quality","positive")]),
 ("Love this app, 10 min me doodh aa jata hai", 5, 0, [("app_support","positive"),("delivery_speed","positive")]),
 ("Congratulations team, aaj phir late, consistency toh hai 😂", 1, 1, [("delivery_speed","negative")]),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "demo"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "reviews.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "text", "rating", "sarcastic", "source"])
        for i, (t, r, s, _) in enumerate(D, 1):
            w.writerow([f"demo_{i:03d}", t, r, s, "SYNTHETIC_DEMO"])
    with open(out / "aspect_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "aspect", "sentiment"])
        for i, (_, _, _, labs) in enumerate(D, 1):
            for a, s in labs:
                w.writerow([f"demo_{i:03d}", a, s])
    n_inst = sum(len(x[3]) for x in D)
    print(f"wrote {len(D)} SYNTHETIC reviews, {n_inst} aspect instances, "
          f"{sum(x[2] for x in D)} sarcastic -> {out}")


if __name__ == "__main__":
    main()
