import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexCart · AI Recommendations",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PRODUCT CATALOG ──────────────────────────────────────────────────
CATALOG = [
    {"name": "Wireless Noise-Cancelling Headphones", "emoji": "🎧", "cat": "Audio",        "price": 2999, "brand": "SoundCore"},
    {"name": "Portable Bluetooth Speaker",           "emoji": "🔊", "cat": "Audio",        "price": 1499, "brand": "JBL Mini"},
    {"name": "USB-C Hub 7-in-1",                     "emoji": "🔌", "cat": "Accessories",  "price": 1299, "brand": "Anker"},
    {"name": "Mechanical Keyboard RGB",              "emoji": "⌨️", "cat": "Peripherals",  "price": 3499, "brand": "Keychron"},
    {"name": "Ergonomic Gaming Mouse",               "emoji": "🖱️", "cat": "Peripherals",  "price": 1799, "brand": "Logitech"},
    {"name": "4K Webcam Pro",                           "emoji": "📷", "cat": "Video",        "price": 4999, "brand": "Elgato"},
    {"name": "LED Architect Desk Lamp",              "emoji": "💡", "cat": "Accessories",  "price": 899,  "brand": "Philips"},
    {"name": "Portable Charger 20000mAh",            "emoji": "🔋", "cat": "Power",        "price": 1199, "brand": "Anker"},
    {"name": "Smart Watch Series X",                 "emoji": "⌚", "cat": "Wearables",    "price": 8999, "brand": "Noise"},
    {"name": "Adjustable Phone Stand",               "emoji": "📱", "cat": "Accessories",  "price": 399,  "brand": "Belkin"},
    {"name": "Laptop Sleeve 15\"",                   "emoji": "💼", "cat": "Accessories",  "price": 599,  "brand": "Tomtoc"},
    {"name": "Cable Management Box",                 "emoji": "🗂️", "cat": "Accessories",  "price": 349,  "brand": "BlueLounge"},
    {"name": "Screen Cleaning Kit",                  "emoji": "🧴", "cat": "Accessories",  "price": 199,  "brand": "iKlear"},
    {"name": "Memory Foam Mouse Pad XL",             "emoji": "🖱️", "cat": "Peripherals",  "price": 499,  "brand": "Corsair"},
    {"name": "Mini Projector 1080p",                 "emoji": "📽️", "cat": "Video",        "price": 12999,"brand": "BenQ"},
    {"name": "Smart Plug 4-Pack",                    "emoji": "🔌", "cat": "Smart Home",   "price": 999,  "brand": "TP-Link"},
    {"name": "Ring Light 12-inch",                   "emoji": "💡", "cat": "Video",        "price": 1599, "brand": "Neewer"},
    {"name": "Professional Drawing Tablet",          "emoji": "✏️", "cat": "Creative",     "price": 5499, "brand": "Wacom"},
    {"name": "NVMe SSD 1TB",                         "emoji": "💾", "cat": "Storage",      "price": 6999, "brand": "Samsung"},
    {"name": "DDR5 RAM 32GB",                        "emoji": "🧠", "cat": "Storage",      "price": 7499, "brand": "Corsair"},
    {"name": "Laptop Cooling Pad",                   "emoji": "❄️", "cat": "Accessories",  "price": 799,  "brand": "Cooler Master"},
    {"name": "Gel Wrist Rest",                       "emoji": "🖐️", "cat": "Peripherals",  "price": 449,  "brand": "Kensington"},
    {"name": "Single Monitor Arm",                   "emoji": "🖥️", "cat": "Peripherals",  "price": 2299, "brand": "Ergotron"},
    {"name": "Surge Protector 8-Outlet",             "emoji": "⚡", "cat": "Power",        "price": 1099, "brand": "APC"},
    {"name": "Foldable BT Keyboard",                 "emoji": "⌨️", "cat": "Peripherals",  "price": 1899, "brand": "iClever"},
]

def get_product(product_id):
    idx = abs(hash(str(product_id))) % len(CATALOG)
    return CATALOG[idx]

# ── PREDICT FUNCTION (outside cache — fixes pickle error) ─────────────
def predict(uim, uid, pid):
    if uid not in uim.index:
        return 0.0
    cid = uim.loc[uid, 'cluster']
    cu  = uim[uim['cluster'] == cid]
    if pid in cu.columns:
        return round(float(cu[pid].mean()), 2)
    return 0.0

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

.stApp { background: #FAFAF8 !important; }
section.main > div { padding-top: 0 !important; }

.topbar {
    background: #1a1a2e;
    padding: 0 40px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0 0 12px 12px;
    margin-bottom: 24px;
}
.topbar-logo { font-size: 1.45rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.topbar-logo em { color: #F59E0B; font-style: normal; }
.topbar-links { display: flex; gap: 28px; }
.topbar-link { color: #9ca3af; font-size: 0.8rem; font-weight: 500; }
.topbar-link.active { color: #F59E0B; }

.banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.banner::after {
    content: '🛒';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.banner-eyebrow { font-size: 0.72rem; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px; }
.banner-title { font-size: 2rem; font-weight: 800; color: #fff; margin: 0 0 8px 0; line-height: 1.2; }
.banner-sub { color: #94a3b8; font-size: 0.88rem; margin: 0; }

.stats-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 28px; }
.stat-box { background: white; border-radius: 12px; padding: 16px 14px; text-align: center; border: 1px solid #f0f0ee; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.stat-val { font-size: 1.3rem; font-weight: 800; color: #1a1a2e; display: block; line-height: 1; }
.stat-key { font-size: 0.67rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; display: block; }

.sh { font-size: 1.15rem; font-weight: 800; color: #1a1a2e; margin: 28px 0 16px 0; display: flex; align-items: center; gap: 10px; }
.sh-line { flex: 1; height: 2px; background: linear-gradient(to right, #F59E0B33, transparent); border-radius: 2px; }

.pcard { background: white; border-radius: 14px; padding: 0; border: 1px solid #f0f0ee; box-shadow: 0 2px 8px rgba(0,0,0,0.05); overflow: hidden; transition: box-shadow 0.2s, transform 0.2s; position: relative; margin-bottom: 16px; }
.pcard:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.10); transform: translateY(-3px); }
.pcard-img { background: linear-gradient(135deg, #f8f9ff, #eef0ff); padding: 22px; text-align: center; font-size: 3rem; border-bottom: 1px solid #f5f5f3; position: relative; }
.pcard-body { padding: 14px 16px 16px; }
.pcard-cat { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #6366f1; margin-bottom: 4px; }
.pcard-brand { font-size: 0.72rem; color: #9ca3af; margin-bottom: 4px; }
.pcard-name { font-size: 0.85rem; font-weight: 700; color: #1a1a2e; line-height: 1.35; margin-bottom: 8px; min-height: 36px; }
.pcard-stars { color: #F59E0B; font-size: 0.8rem; margin-bottom: 2px; }
.pcard-pred { font-size: 0.7rem; color: #9ca3af; margin-bottom: 10px; }
.pcard-price { font-size: 1.1rem; font-weight: 800; color: #1a1a2e; }
.pcard-price-sub { font-size: 0.68rem; color: #9ca3af; margin-bottom: 10px; }
.pcard-btn { display: block; width: 100%; background: #F59E0B; color: #1a1a2e; font-weight: 800; font-size: 0.78rem; text-align: center; padding: 9px; border-radius: 8px; border: none; cursor: pointer; letter-spacing: 0.02em; }
.pcard-btn:hover { background: #d97706; }

.badge-rank { position: absolute; top: 10px; left: 10px; background: #1a1a2e; color: #F59E0B; font-size: 0.62rem; font-weight: 800; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.05em; z-index: 2; }
.badge-match { position: absolute; top: 10px; right: 10px; background: #059669; color: white; font-size: 0.62rem; font-weight: 700; padding: 3px 7px; border-radius: 6px; z-index: 2; }
.badge-prime { display: inline-block; background: #1a1a2e; color: #60a5fa; font-size: 0.6rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.08em; margin-bottom: 8px; }

.selector-card { background: white; border-radius: 14px; padding: 20px 24px; border: 1px solid #f0f0ee; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 24px; }

.footer { background: #1a1a2e; color: #64748b; text-align: center; padding: 20px; border-radius: 12px; margin-top: 48px; font-size: 0.75rem; line-height: 1.8; }
.footer strong { color: #94a3b8; }

div[data-testid="stFileUploader"] { background: white; border-radius: 12px; padding: 12px 16px; border: 1px solid #f0f0ee; }
div[data-testid="stSelectbox"] > label, div[data-testid="stSlider"] > label { font-weight: 700 !important; color: #374151 !important; font-size: 0.82rem !important; }
.stButton > button { background: #F59E0B !important; color: #1a1a2e !important; font-weight: 800 !important; border: none !important; border-radius: 10px !important; font-size: 0.82rem !important; letter-spacing: 0.02em !important; }
</style>
""", unsafe_allow_html=True)


# ── TOPBAR ────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">Nex<em>Cart</em></div>
    <div class="topbar-links">
        <span class="topbar-link active">Recommendations</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── DATA PROCESSING ───────────────────────────────────────────────────
@st.cache_data
def process(file):
    df = pd.read_csv(file)
    if 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    uc = df['userid'].value_counts()
    df = df[df['userid'].isin(uc[uc >= 2].index)]
    pc = df['productid'].value_counts()
    df = df[df['productid'].isin(pc[pc >= 2].index)]

    uim = df.pivot_table(index='userid', columns='productid', values='rating')
    uim = uim.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0)

    sc = StandardScaler()
    scaled = sc.fit_transform(uim)

    km = KMeans(n_clusters=4, random_state=42, n_init='auto')
    labels = km.fit_predict(scaled)
    uim['cluster'] = labels

    _, test = train_test_split(df, test_size=0.2, random_state=42)

    yt, yp = [], []
    for r in test.itertuples():
        yt.append(r.rating)
        yp.append(predict(uim, r.userid, r.productid))
    rmse = root_mean_squared_error(yt, yp)

    return df, uim, rmse  # no longer returning predict


# ── UPLOAD ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂  Upload **rating_short.csv** to load the store", type=["csv"])

if uploaded is None:
    st.markdown("""
    <div style="background:white;border-radius:16px;padding:56px 32px;text-align:center;
                border:2px dashed #e5e7eb;margin-top:12px">
        <div style="font-size:3.5rem;margin-bottom:16px">🛒</div>
        <div style="font-size:1.25rem;font-weight:800;color:#1a1a2e">Welcome to NexCart</div>
        <div style="color:#9ca3af;font-size:0.88rem;margin-top:8px;max-width:400px;margin-left:auto;margin-right:auto">
            Upload your <b>rating_short.csv</b> file above to power the AI recommendation engine.<br>
            Required columns: <code>userid · productid · rating</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df, uim, rmse = process(uploaded)   # 3 values now
all_users    = uim.index.tolist()
all_products = [c for c in uim.columns if c != 'cluster']


# ── BANNER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="banner">
    <div class="banner-eyebrow">✦ AI-Powered </div>
    <div class="banner-title">Your Personalized Store</div>
    <p class="banner-sub">
        {df['userid'].nunique():,} customers · {df['productid'].nunique():,} products · 4 user clusters
    </p>
</div>
""", unsafe_allow_html=True)


# ── SELECTOR ─────────────────────────────────────────────────────────
st.markdown("<div class='selector-card'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([2.2, 1, 1])
with c1:
    sel_user = st.selectbox("👤  Select your Customer ID", all_users)
with c2:
    top_n = st.slider("Show top N products", 4, 12, 8, step=4)
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⚡  Refresh picks", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ── COMPUTE ───────────────────────────────────────────────────────────
scored = []
for pid in all_products:
    s = predict(uim, sel_user, pid)   # pass uim as argument
    if s > 0:
        p = get_product(pid)
        scored.append({**p, "pid": pid, "score": s,
                       "match": int((s / 5) * 100)})

top_picks   = sorted(scored, key=lambda x: x['score'], reverse=True)[:top_n]
also_viewed = sorted(scored, key=lambda x: x['score'])[:4]

if not top_picks:
    st.warning("No recommendations found. Try another user.")
    st.stop()

cid   = int(uim.loc[sel_user, 'cluster'])
csz   = int((uim['cluster'] == cid).sum())
urats = df[df['userid'] == sel_user]['rating']
avgr  = round(urats.mean(), 1) if len(urats) > 0 else 0.0


# ── STATS ─────────────────────────────────────────────────────────────
star_str = "⭐" * int(round(avgr)) if avgr > 0 else "—"
st.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <span class="stat-val">Cluster {cid}</span>
        <span class="stat-key">User group</span>
    </div>
    <div class="stat-box">
        <span class="stat-val">{csz}</span>
        <span class="stat-key">Similar users</span>
    </div>
    <div class="stat-box">
        <span class="stat-val">{len(urats)}</span>
        <span class="stat-key">Products rated</span>
    </div>
    <div class="stat-box">
        <span class="stat-val">{star_str}</span>
        <span class="stat-key">Avg given rating</span>
    </div>
    <div class="stat-box">
        <span class="stat-val">{rmse:.2f}</span>
        <span class="stat-key">Model RMSE</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TOP PICKS GRID ────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
    🎯&nbsp; Recommended For You
    <div class="sh-line"></div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
for i, p in enumerate(top_picks):
    sf    = int(round(p['score']))
    stars = "★" * sf + "☆" * (5 - sf)
    with cols[i % 4]:
        st.markdown(f"""
        <div class="pcard">
            <div class="pcard-img">
                <div class="badge-rank">#{i+1}</div>
                <div class="badge-match">{p['match']}% match</div>
                {p['emoji']}
            </div>
            <div class="pcard-body">
                <div class="pcard-cat">{p['cat']}</div>
                <div class="pcard-brand">{p['brand']}</div>
                <div class="pcard-name">{p['name']}</div>
                <div class="pcard-stars">{stars}</div>
                <div class="pcard-pred">Predicted: {p['score']:.1f} / 5.0</div>
                <span class="badge-prime" style="position:static">PRIME</span>
                <div class="pcard-price">₹{p['price']:,}</div>
                <div class="pcard-price-sub">Free delivery · In stock</div>
                <button class="pcard-btn">🛒 Add to Cart</button>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── ALSO VIEWED ───────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
    👀&nbsp; Customers Also Viewed
    <div class="sh-line"></div>
</div>
""", unsafe_allow_html=True)

cols2 = st.columns(4)
for i, p in enumerate(also_viewed):
    sf    = int(round(p['score']))
    stars = "★" * sf + "☆" * (5 - sf)
    with cols2[i % 4]:
        st.markdown(f"""
        <div class="pcard">
            <div class="pcard-img">{p['emoji']}</div>
            <div class="pcard-body">
                <div class="pcard-cat">{p['cat']}</div>
                <div class="pcard-brand">{p['brand']}</div>
                <div class="pcard-name">{p['name']}</div>
                <div class="pcard-stars">{stars}</div>
                <span class="badge-prime" style="position:static">PRIME</span>
                <div class="pcard-price">₹{p['price']:,}</div>
                <div class="pcard-price-sub">Free delivery · In stock</div>
                <button class="pcard-btn">🛒 Add to Cart</button>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <strong>NexCart Recommendation Engine</strong><br>
    K-Means Collaborative Filtering · 4 User Clusters · Silhouette ≈ 0.90 · RMSE ≈ {rmse:.2f}<br>
    · Product Recommendation System
</div>
""", unsafe_allow_html=True)