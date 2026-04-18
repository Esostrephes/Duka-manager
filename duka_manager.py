

import os, re, json, asyncio, logging, base64, io, uuid, textwrap, hashlib
from collections import defaultdict, Counter 
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np
import requests
import aiohttp
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from supabase import create_client, Client
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
@dataclass
class Config:
    supabase_url:          str = os.getenv("SUPABASE_URL", "")
    supabase_key:          str = os.getenv("SUPABASE_KEY", "")
    story_score_threshold:    float = float(os.getenv("STORY_SCORE_THRESHOLD", "7.0"))
    channel_link:     str = os.getenv("CHANNEL_LINK", "")
    channel_live: bool = os.getenv("CHANNEL_LIVE", "false").lower()==True
    openai_api_key:        str = os.getenv("OPENAI_API_KEY", "")
    mpesa_consumer_key:    str = os.getenv("MPESA_CONSUMER_KEY", "")
    mpesa_consumer_secret: str = os.getenv("MPESA_CONSUMER_SECRET", "")
    mpesa_shortcode:       str = os.getenv("MPESA_SHORTCODE", "")
    mpesa_passkey:         str = os.getenv("MPESA_PASSKEY", "")
    mpesa_callback_url:    str = os.getenv("MPESA_CALLBACK_URL", "")
    cron_secret:           str = os.getenv("CRON_SECRET", "change_this_in_production")
    data_sources:          List[str] = None
    meta_access_token:     str = os.getenv("META_ACCESS_TOKEN", "")
    meta_phone_number_id:  str = os.getenv("META_PHONE_NUMBER_ID", "")
    Whatsapp_channel_id:   str = os.getenv("WHATSAPP_CHANNEL_ID", "")
    data_sources:     List[str] = None

    def __post_init__(self):
        raw = os.getenv("DATA_SOURCES", "")
        self.data_sources = [s.strip() for s in raw.split(",") if s.strip()]

config = Config()

# ══════════════════════════════════════════════
# DATABASE MANAGER
# ══════════════════════════════════════════════
class DatabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(config.supabase_url, config.supabase_key)

    async def execute(self, table: str, operation: str, **kwargs):
        loop = asyncio.get_event_loop()
        max_retries = 3

        def _run():
            if operation == "select":
                query = self.supabase.table(table).select("*")
                if kwargs.get("eq"):
                    for k, v in kwargs["eq"].items():
                        query = query.eq(k, v)
                if kwargs.get("order"):
                    query = query.order(
                        kwargs["order"]["column"],
                        desc=kwargs["order"].get("desc", False)
                    )
                return query.execute()
            elif operation == "insert":
                return self.supabase.table(table).insert(kwargs["data"]).execute()
            elif operation == "update":
                return (
                    self.supabase.table(table)
                    .update(kwargs["data"])
                    .eq(kwargs["match"]["key"], kwargs["match"]["value"])
                    .execute()
                )
            elif operation == "delete":
                return (
                    self.supabase.table(table)
                    .delete()
                    .eq(kwargs["match"]["key"], kwargs["match"]["value"])
                    .execute()
                )
            raise ValueError(f"Unknown operation: {operation}")

        for attempt in range(max_retries):
            try:
                return await loop.run_in_executor(None, _run)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

db = DatabaseManager()

# ══════════════════════════════════════════════
# NLP ENGINE
# ══════════════════════════════════════════════
_NAME_STOPWORDS = {
    "Add","Sell","Buy","View","Show","List","Debt","Pay","Sale","Stock",
    "Product","Expense","Alert","Report","Summary","Forecast","Compare",
    "Insight","Inventory","Payment","Customer","Units","Pieces","Kilogram",
    "Litre","KES","KSh","Cash","Mpesa","Na","Kwa","Leo","Jana",
}

class NLPEngine:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.intent_patterns = {
            "add_product": [r"\badd product\b",r"\bnew product\b",r"\badd stock\b",r"\bstock up\b",r"\bongeza bidhaa\b"],
            "sell":        [r"\bsold\b",r"\bsell\b",r"\bsale\b",r"\bcustomer bought\b",r"\buliuza\b"],
            "debt":        [r"\bdebt\b",r"\bowed\b",r"\bowes\b",r"\bcredit\b",r"\bdeni\b",r"\bkopa\b"],
            "payment":     [r"\bpaid\b",r"\bpay\b",r"\bpayment\b",r"\breceived\b",r"\blipia\b"],
            "view":        [r"\bshow\b",r"\bview\b",r"\blist\b",r"\breport\b",r"\bsummary\b",r"\bonyesha\b"],
            "snapshot":    [r"\bhow is business\b",r"\bbiashara yako\b",r"\bleo biashara\b",
                            r"\bperformance\b",r"\btoday.{0,10}business\b",r"\bvipi biashara\b",
                            r"\bhali ya biashara\b",r"\bcheck business\b"],
            "insight":     [r"\binsight\b",r"\badvice\b",r"\bsuggestion\b",r"\bushauri\b",r"\bimprove\b"],
            "expense":     [r"\bexpense\b",r"\bspent\b",r"\bgharama\b",r"\bmatumizi\b"],
            "alert":       [r"\balert\b",r"\bwarning\b",r"\blow stock\b",r"\bexpiring\b",r"\bonyo\b"],
            "compare":     [r"\bcompare\b",r"\bcompetitor\b",r"\bmarket\b",r"\bother shops\b",r"\blinganisha\b"],
            "forecast":    [r"\bforecast\b",r"\bpredict\b",r"\btrend\b",r"\btabiri\b"],
            "monthly_report": [r"\bmonthly report\b",r"\bbalance book\b",r"\bripoti ya mwezi\b",r"\bmonthly balance\b"],
        }

    async def parse_message(self, message: str) -> Dict:
        msg_lower = message.lower()
        intent = None
        for intent_name, patterns in self.intent_patterns.items():
            if any(re.search(p, msg_lower) for p in patterns):
                intent = intent_name
                break

        entities = self.extract_entities(message)

        if not intent:
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            "You are a Kenyan duka (shop) management bot assistant. "
                            "Respond ONLY with valid JSON: {\"intent\": \"...\", \"entities\": {...}}. "
                            "Intents: add_product, sell, debt, payment, view, snapshot, insight, "
                            "expense, alert, compare, forecast, monthly_report. "
                            "Entities: name, amount, quantity, date, category. "
                            "Support Swahili and English."
                        )},
                        {"role": "user", "content": f"Message: {message}"}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)
                intent = parsed.get("intent", "unknown")
                entities.update(parsed.get("entities", {}))
            except Exception as e:
                log.warning(f"OpenAI parse failed: {e}")
                intent = "unknown"

        return {"intent": intent or "unknown", "entities": entities, "original_message": message}

    def extract_entities(self, message: str) -> Dict:
        entities = {}

        money_match = re.search(
            r"(?:KES|KSh|ksh|sh)[\s=]*(\d[\d,]*(?:\.\d{1,2})?)"
            r"|(?<!\w)(\d[\d,]*\.\d{2})(?!\s*(?:kg|pcs|pieces|units|litres|ltrs?))",
            message, re.IGNORECASE
        )
        if money_match:
            raw = (money_match.group(1) or money_match.group(2) or "").replace(",", "")
            try:
                entities["amount"] = float(raw)
            except ValueError:
                pass

        qty_match = re.search(r"(\d+)\s*(?:kg|pcs|pieces|units|litres?|ltrs?)", message, re.IGNORECASE)
        if qty_match:
            entities["quantity"] = int(qty_match.group(1))
        elif not entities.get("quantity"):
            # fallback: plain integer that isn't already the amount
            plain = re.search(r"\b(\d+)\b", message)
            if plain:
                val = int(plain.group(1))
                if val != entities.get("amount"):
                    entities["quantity"] = val

        for cand in re.findall(r"\b([A-Z][a-z]{2,})\b", message):
            if cand not in _NAME_STOPWORDS:
                entities["name"] = cand
                break

        date_match = re.search(
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(today|tomorrow|yesterday|next week|this month)",
            message, re.IGNORECASE
        )
        if date_match:
            entities["date"] = date_match.group(0)

        return entities

nlp = NLPEngine()

# ══════════════════════════════════════════════
# BUSINESS ANALYTICS
# ══════════════════════════════════════════════
class BusinessAnalytics:
    def __init__(self, shop_id: str):
        self.shop_id = shop_id

    async def calculate_metrics(self, since_date: Optional[date] = None) -> Dict:
        """Full metrics; optionally filtered from a start date."""
        txns = await db.execute("transactions", "select", eq={"shop_id": self.shop_id})
        all_txns = txns.data

        if since_date:
            all_txns = [
                t for t in all_txns
                if dateparser.parse(t["created_at"]).date() >= since_date
            ]

        sales    = [t for t in all_txns if t["type"] == "sale"]
        expenses = [t for t in all_txns if t["type"] == "expense"]

        total_revenue  = sum(s["amount"] for s in sales)
        total_expenses = sum(e["amount"] for e in expenses)
        profit         = total_revenue - total_expenses
        profit_margin  = (profit / total_revenue * 100) if total_revenue > 0 else 0

        products = await db.execute("products", "select", eq={"shop_id": self.shop_id})
        inventory_value = sum(
            p.get("quantity", 0) * (p.get("cost_price") or 0) for p in products.data
        )
        turnover = (total_revenue / inventory_value) if inventory_value > 0 else 0

        debts = await db.execute("debts", "select", eq={"shop_id": self.shop_id, "status": "pending"})
        debt_aging = {"0-30": 0, "31-60": 0, "61-90": 0, "90+": 0}
        for d in debts.data:
            age = (datetime.now() - dateparser.parse(d["created_at"])).days
            if age <= 30:   debt_aging["0-30"]  += d["amount"]
            elif age <= 60: debt_aging["31-60"] += d["amount"]
            elif age <= 90: debt_aging["61-90"] += d["amount"]
            else:           debt_aging["90+"]   += d["amount"]

        return {
            "revenue":             total_revenue,
            "expenses":            total_expenses,
            "profit":              profit,
            "profit_margin":       profit_margin,
            "inventory_value":     inventory_value,
            "inventory_turnover":  turnover,
            "total_debt":          sum(d["amount"] for d in debts.data),
            "debt_aging":          debt_aging,
            "customer_count":      len(set(t.get("reference_id") for t in sales if t.get("reference_id"))),
            "average_transaction": total_revenue / len(sales) if sales else 0,
            "transaction_count":   len(sales),
        }

    async def today_snapshot(self) -> Dict:
        """Metrics for today only."""
        return await self.calculate_metrics(since_date=date.today())

    async def identify_leaks(self) -> List[Dict]:
        metrics = await self.calculate_metrics()
        leaks = []
        rev = metrics["revenue"] or 1

        if metrics["total_debt"] / rev > 0.3:
            leaks.append({
                "type": "high_debt", "severity": "high",
                "description": f"Deni ni KES {metrics['total_debt']:,.0f} ({metrics['total_debt']/rev*100:.0f}% ya mapato)",
                "suggestion": "Fanya follow-up na wateja haraka. Weka ukomo wa mikopo.",
            })
        if metrics["inventory_turnover"] < 4:
            leaks.append({
                "type": "slow_inventory", "severity": "medium",
                "description": f"Bidhaa zinazunguka mara {metrics['inventory_turnover']:.1f} kwa mwezi",
                "suggestion": "Fanya promotion kwa bidhaa zinazokaa sana ghalani.",
            })
        if metrics["profit_margin"] < 20:
            leaks.append({
                "type": "low_margin", "severity": "high",
                "description": f"Margin ni {metrics['profit_margin']:.1f}% — chini ya 20%",
                "suggestion": "Angalia bei zako au pata wafanyabiashara wa bei nafuu.",
            })
        return leaks

    async def generate_insights(self) -> List[Dict]:
        insights = []
        txns = await db.execute("transactions", "select", eq={"shop_id": self.shop_id})

        daily: Dict = defaultdict(float)
        for t in txns.data:
            if t["type"] == "sale":
                daily[dateparser.parse(t["created_at"]).date()] += t["amount"]

        vals = [daily[k] for k in sorted(daily.keys())]
        if len(vals) >= 14:
            r_avg = sum(vals[-7:])  / 7
            p_avg = sum(vals[-14:-7]) / 7
            if p_avg > 0:
                if r_avg > p_avg * 1.1:
                    insights.append({
                        "type": "growth", "severity": "positive",
                        "title": "Mauzo Yanakua! 📈",
                        "description": f"Mauzo yameongezeka {(r_avg/p_avg-1)*100:.1f}% wiki hii",
                        "suggestion": "Tumia momentum hii — fanya promotion au ongeza stock.",
                    })
                elif r_avg < p_avg * 0.9:
                    insights.append({
                        "type": "decline", "severity": "high",
                        "title": "Mauzo Yanapungua 📉",
                        "description": f"Mauzo yameshuka {(p_avg/r_avg-1)*100:.1f}% wiki hii",
                        "suggestion": "Angalia washindani wapya au uliza wateja maoni.",
                    })

        products = await db.execute("products", "select", eq={"shop_id": self.shop_id})
        for p in products.data:
            if p.get("quantity", 0) > p.get("reorder_level", 10) * 3:
                insights.append({
                    "type": "excess_stock", "severity": "medium",
                    "title": f"Stock Nyingi: {p['name']}",
                    "description": f"Una {p['quantity']} units ghalani",
                    "suggestion": f"Fanya discount kwa {p['name']} kupunguza stock.",
                })
        return insights

    async def compare_with_market(self) -> Dict:
        market = await db.execute("market_data", "select")
        if not market.data:
            return {"comparisons": [], "recommendations": [], "message": "Hakuna data ya soko bado."}

        products = await db.execute("products", "select", eq={"shop_id": self.shop_id})
        comparisons, recommendations = [], []

        for p in products.data:
            prices = [m["price"] for m in market.data if m["product_name"].lower() == p["name"].lower()]
            if not prices:
                continue
            avg = sum(prices) / len(prices)
            diff = p.get("price", 0) - avg
            pct  = (diff / avg * 100) if avg else 0
            comparisons.append({
                "product": p["name"], "your_price": p.get("price", 0),
                "market_avg": avg, "difference": diff, "percentage_diff": pct,
            })
            if abs(pct) > 10:
                direction = "punguza" if pct > 0 else "panda"
                recommendations.append(
                    f"{direction.capitalize()} bei ya {p['name']} kwa {abs(pct):.0f}% kulingana na soko"
                )

        return {"comparisons": comparisons, "recommendations": recommendations}

    async def forecast_sales(self, days: int = 7) -> Dict:
        txns = await db.execute("transactions", "select", eq={"shop_id": self.shop_id})
        daily: Dict = defaultdict(float)
        for t in txns.data:
            if t["type"] == "sale":
                daily[dateparser.parse(t["created_at"]).date()] += t["amount"]

        if len(daily) < 7:
            return {"error": "Inahitaji angalau siku 7 za data ili kufanya utabiri."}

        days_list = sorted(daily.keys())
        origin = days_list[0]
        X = np.array([(d - origin).days for d in days_list]).reshape(-1, 1)
        y = np.array([daily[d] for d in days_list])
        model = LinearRegression().fit(X, y)

        last = days_list[-1]
        future = [last + timedelta(days=i) for i in range(1, days + 1)]
        X_f    = np.array([(d - origin).days for d in future]).reshape(-1, 1)
        preds  = np.maximum(model.predict(X_f), 0)

        return {
            "forecast":       [{"date": d.isoformat(), "predicted_sales": round(float(v), 2)}
                               for d, v in zip(future, preds)],
            "total_forecast": round(float(sum(preds)), 2),
            "trend":          round(float(model.coef_[0]), 2),
      }

# MARKET DATA SCRAPER
# ══════════════════════════════════════════════
class MarketDataScraper:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def scrape_data(self) -> int:
        all_data = []
        for src in config.data_sources:
            try:
                if "kamis" in src.lower():
                    all_data.extend(await self._scrape_kamis(src))
                elif "cheki" in src.lower():
                    all_data.extend(await self._scrape_cheki(src))
                else:
                    all_data.extend(await self._scrape_generic(src))
            except Exception as e:
                log.error(f"Scrape failed for {src}: {e}")

        for item in all_data:
            await db.execute("market_data", "insert", data={
                "source": item["source"], "product_name": item["product"],
                "price": item["price"], "location": item.get("location"),
                "raw_data": item.get("raw_data"),
            })
        return len(all_data)

    async def _get_html(self, url: str) -> str:
        s = await self.session()
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.text()

    async def _scrape_kamis(self, url: str) -> List[Dict]:
        html = await self._get_html(url)
        soup = BeautifulSoup(html, "html.parser")
        out  = []
        for row in soup.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                try:
                    out.append({"source":"KAMIS","product":cols[0].text.strip(),
                                "price":float(cols[1].text.replace(",","").strip()),
                                "location":cols[2].text.strip() if len(cols)>2 else "Nairobi"})
                except ValueError:
                    pass
        return out

    async def _scrape_cheki(self, url: str) -> List[Dict]:
        html = await self._get_html(url)
        soup = BeautifulSoup(html, "html.parser")
        out  = []
        for item in soup.find_all("div", class_="listing-item"):
            pt = item.find("span", class_="price")
            if pt:
                try:
                    out.append({"source":"Cheki","product":"Vehicle",
                                "price":float(re.sub(r"[^\d.]","",pt.text)),
                                "raw_data":{"title": item.find("h3").text if item.find("h3") else ""}})
                except ValueError:
                    pass
        return out

    async def _scrape_generic(self, url: str) -> List[Dict]:
        html = await self._get_html(url)
        soup = BeautifulSoup(html, "html.parser")
        out  = []
        for tag in soup.find_all(string=re.compile(r"KES\s*\d", re.IGNORECASE)):
            m = re.search(r"KES\s*(\d[\d,]*(?:\.\d{1,2})?)", tag, re.IGNORECASE)
            if not m:
                continue
            try:
                price = float(m.group(1).replace(",",""))
            except ValueError:
                continue
            parent = getattr(tag, "parent", None)
            label  = None
            if parent:
                for el in parent.find_all(["h1","h2","h3","h4","strong","b"]):
                    txt = el.get_text(strip=True)
                    if txt and len(txt) < 80:
                        label = txt
                        break
            if not label:
                tt = soup.find("title")
                label = tt.get_text(strip=True)[:60] if tt else url
            out.append({"source":url,"product":label,"price":price,"raw_data":{"snippet":str(tag)[:200]}})
            if len(out) >= 10:
                break
        return out

scraper = MarketDataScraper()

# ══════════════════════════════════════════════
# M-PESA SERVICE
# ══════════════════════════════════════════════
class MpesaService:
    BASE = "https://sandbox.safaricom.co.ke"   # switch to api.safaricom.co.ke in prod

    async def _token(self) -> str:
        auth = base64.b64encode(
            f"{config.mpesa_consumer_key}:{config.mpesa_consumer_secret}".encode()
        ).decode()
        r = requests.get(f"{self.BASE}/oauth/v1/generate?grant_type=client_credentials",
                         headers={"Authorization": f"Basic {auth}"}, timeout=10)
        return r.json().get("access_token", "")

    async def stk_push(self, phone: str, amount: int, ref: str) -> Dict:
        token     = await self._token()
        ts        = datetime.now().strftime("%Y%m%d%H%M%S")
        password  = base64.b64encode(
            f"{config.mpesa_shortcode}{config.mpesa_passkey}{ts}".encode()
        ).decode()
        payload = {
            "BusinessShortCode": config.mpesa_shortcode,
            "Password": password, "Timestamp": ts,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": amount, "PartyA": phone, "PartyB": config.mpesa_shortcode,
            "PhoneNumber": phone, "CallBackURL": config.mpesa_callback_url,
            "AccountReference": ref, "TransactionDesc": "Duka Bot Subscription",
        }
        r = requests.post(f"{self.BASE}/mpesa/stkpush/v1/processrequest",
                          json=payload,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          timeout=15)
        return r.json()

mpesa = MpesaService()

# ══════════════════════════════════════════════
# BALANCE BOOK IMAGE BUILDER
# ══════════════════════════════════════════════
C = {
    "bg":"#FDFAF4","header_bg":"#1B4332","header_fg":"#FFFFFF",
    "row_even":"#F0F7F2","row_odd":"#FFFFFF","accent":"#2D6A4F",
    "danger":"#C1121F","warn":"#E9C46A","positive":"#40916C",
    "text":"#1A1A2E","subtext":"#555577","border":"#B7D5C2",
    "line_rev":"#2D6A4F","line_exp":"#C1121F","line_profit":"#E9C46A",
}

class BalanceBookImageBuilder:
    FIG_W, FIG_H, DPI = 12, 17, 130

    def build(self, shop_name: str, month_label: str,
              weekly_rows: List[Dict], daily_data: Dict[str, Dict],
              metrics: Dict, advice_msgs: List[str]) -> bytes:
        fig = plt.figure(figsize=(self.FIG_W, self.FIG_H), facecolor=C["bg"])
        gs  = gridspec.GridSpec(5, 1, figure=fig,
                                height_ratios=[0.10,0.12,0.28,0.32,0.18],
                                hspace=0.04, top=0.97, bottom=0.02, left=0.04, right=0.96)
        self._header(fig.add_subplot(gs[0]), shop_name, month_label)
        self._kpi_row(fig.add_subplot(gs[1]), metrics)
        self._table(fig.add_subplot(gs[2]), weekly_rows)
        self._charts(fig.add_subplot(gs[3]), daily_data)
        self._advice(fig.add_subplot(gs[4]), advice_msgs)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _header(self, ax, shop_name, month_label):
        ax.set_facecolor(C["header_bg"]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.axhline(0, color=C["positive"], linewidth=3)
        ax.text(0.5, 0.72, shop_name.upper(), ha="center", va="center",
                fontsize=18, fontweight="bold", color=C["header_fg"], transform=ax.transAxes)
        ax.text(0.5, 0.28, f"📊  DAFTARI LA BIASHARA  |  {month_label.upper()}",
                ha="center", va="center", fontsize=9, color="#A8D5B5",
                transform=ax.transAxes, fontfamily="monospace")

    def _kpi_row(self, ax, metrics):
        ax.set_facecolor(C["bg"]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        kpis = [
            ("💰 Mapato",  f"KES {metrics.get('revenue',0):,.0f}",       C["accent"]),
            ("💸 Matumizi",f"KES {metrics.get('expenses',0):,.0f}",       C["danger"]),
            ("📈 Faida",   f"KES {metrics.get('profit',0):,.0f}",         C["positive"]),
            ("🎯 Margin",  f"{metrics.get('profit_margin',0):.1f}%",       C["warn"]),
            ("💳 Deni",    f"KES {metrics.get('total_debt',0):,.0f}",     C["danger"]),
        ]
        box_w = 0.18
        gap   = (1 - box_w * len(kpis)) / (len(kpis) + 1)
        for i, (label, value, colour) in enumerate(kpis):
            x = gap + i * (box_w + gap)
            ax.add_patch(FancyBboxPatch((x,0.08), box_w, 0.84, boxstyle="round,pad=0.02",
                linewidth=1.5, edgecolor=colour, facecolor=colour+"18",
                transform=ax.transAxes, clip_on=False))
            cx = x + box_w / 2
            ax.text(cx, 0.70, label, ha="center", va="center", fontsize=7.5,
                    color=C["subtext"], transform=ax.transAxes)
            ax.text(cx, 0.30, value, ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=colour, transform=ax.transAxes)

    def _table(self, ax, weekly_rows):
        ax.set_facecolor(C["bg"]); ax.axis("off")
        if not weekly_rows:
            ax.text(0.5,0.5,"Hakuna data", ha="center", va="center",
                    fontsize=11, color=C["subtext"]); return

        cols   = ["Wiki","Mapato (KES)","Matumizi (KES)","Faida (KES)","Margin %","Txns"]
        col_w  = [0.10,0.20,0.20,0.20,0.16,0.08]
        row_h  = 1.0 / (len(weekly_rows) + 2)
        xs = []
        cx = 0.01
        for w in col_w:
            xs.append(cx); cx += w

        # header
        ax.add_patch(FancyBboxPatch((0, 1.0-row_h), 1.0, row_h, boxstyle="square,pad=0",
            facecolor=C["header_bg"], edgecolor="none", transform=ax.transAxes, clip_on=False))
        for j,(col,x) in enumerate(zip(cols,xs)):
            ax.text(x+col_w[j]/2, 1.0-row_h*0.5, col, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=C["header_fg"],
                    transform=ax.transAxes, zorder=5)

        # rows
        for i, row in enumerate(weekly_rows):
            yt = 1.0 - row_h*(i+1)
            ax.add_patch(FancyBboxPatch((0, yt-row_h), 1.0, row_h, boxstyle="square,pad=0",
                facecolor=C["row_even"] if i%2==0 else C["row_odd"],
                edgecolor=C["border"], linewidth=0.4,
                transform=ax.transAxes, clip_on=False))
            prof = row.get("profit",0)
            pc   = C["positive"] if prof >= 0 else C["danger"]
            vals = [row.get("week",f"W{i+1}"), f"{row.get('revenue',0):,.0f}",
                    f"{row.get('expenses',0):,.0f}", f"{prof:,.0f}",
                    f"{row.get('margin',0):.1f}%", str(row.get("transactions",0))]
            colours = [C["text"],C["accent"],C["danger"],pc,pc,C["text"]]
            cy = yt - row_h/2
            for j,(v,x) in enumerate(zip(vals,xs)):
                ax.text(x+col_w[j]/2, cy, v, ha="center", va="center",
                        fontsize=7.8, color=colours[j], transform=ax.transAxes, zorder=5)

        # totals
        yt = 1.0 - row_h*(len(weekly_rows)+1)
        tr = sum(r.get("revenue",0) for r in weekly_rows)
        te = sum(r.get("expenses",0) for r in weekly_rows)
        tp = tr-te; tt = sum(r.get("transactions",0) for r in weekly_rows)
        tm = (tp/tr*100) if tr else 0
        ax.add_patch(FancyBboxPatch((0,yt-row_h),1.0,row_h, boxstyle="square,pad=0",
            facecolor=C["accent"]+"22", edgecolor=C["accent"], linewidth=1,
            transform=ax.transAxes, clip_on=False))
        tvals   = ["JUMLA",f"{tr:,.0f}",f"{te:,.0f}",f"{tp:,.0f}",f"{tm:.1f}%",str(tt)]
        tc      = C["positive"] if tp>=0 else C["danger"]
        tcolours= [C["text"],C["accent"],C["danger"],tc,tc,C["text"]]
        ty = yt - row_h/2
        for j,(v,x) in enumerate(zip(tvals,xs)):
            ax.text(x+col_w[j]/2, ty, v, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=tcolours[j],
                    transform=ax.transAxes, zorder=5)

    def _charts(self, ax, daily_data):
        ax.set_visible(False)
        fig  = ax.get_figure()
        bbox = ax.get_position()
        lax  = fig.add_axes([bbox.x0, bbox.y0, bbox.width*0.47, bbox.height])
        rax  = fig.add_axes([bbox.x0+bbox.width*0.53, bbox.y0, bbox.width*0.47, bbox.height])

        dates = sorted(daily_data.keys())
        if not dates:
            return
        revs  = [daily_data[d].get("revenue",0)  for d in dates]
        exps  = [daily_data[d].get("expenses",0) for d in dates]
        profs = [r-e for r,e in zip(revs,exps)]
        x     = np.arange(len(dates))
        step  = max(1, len(dates)//6)
        fmt   = matplotlib.ticker.FuncFormatter(
            lambda v,_: f"{v/1000:.0f}k" if abs(v)>=1000 else f"{v:.0f}")

        # left: bars
        lax.set_facecolor(C["bg"])
        lax.bar(x, revs, color=C["accent"], alpha=0.85, width=0.7, zorder=3)
        lax.bar(x, [-e for e in exps], color=C["danger"], alpha=0.55, width=0.7, zorder=3)
        lax.set_xticks(x[::step])
        lax.set_xticklabels([d[5:] for d in dates[::step]], fontsize=6, rotation=45,
                             ha="right", color=C["subtext"])
        lax.tick_params(axis="y", labelsize=6, colors=C["subtext"])
        lax.set_title("Mapato vs Matumizi (Kila Siku)", fontsize=8,
                       color=C["text"], pad=4, fontweight="bold")
        lax.spines[:].set_color(C["border"])
        lax.grid(axis="y", color=C["border"], linewidth=0.5, linestyle="--", zorder=0)
        lax.yaxis.set_major_formatter(fmt)

        # right: lines
        rax.set_facecolor(C["bg"])
        rax.plot(x, revs,  color=C["line_rev"],    linewidth=2,   label="Mapato", zorder=4)
        rax.plot(x, exps,  color=C["line_exp"],    linewidth=1.5, label="Matumizi",
                 linestyle="--", zorder=4)
        rax.plot(x, profs, color=C["line_profit"], linewidth=2,   label="Faida",
                 linestyle=":", zorder=4)
        rax.fill_between(x, 0, profs, where=[p>=0 for p in profs], alpha=0.12, color=C["positive"])
        rax.fill_between(x, 0, profs, where=[p<0  for p in profs], alpha=0.12, color=C["danger"])
        rax.set_xticks(x[::step])
        rax.set_xticklabels([d[5:] for d in dates[::step]], fontsize=6, rotation=45,
                              ha="right", color=C["subtext"])
        rax.tick_params(axis="y", labelsize=6, colors=C["subtext"])
        rax.set_title("Mwenendo wa Biashara", fontsize=8, color=C["text"], pad=4, fontweight="bold")
        rax.spines[:].set_color(C["border"])
        rax.grid(color=C["border"], linewidth=0.4, linestyle="--", zorder=0)
        rax.legend(fontsize=6, loc="upper left", facecolor=C["bg"], edgecolor=C["border"])
        rax.yaxis.set_major_formatter(fmt)

    def _advice(self, ax, advice_msgs):
        ax.set_facecolor(C["bg"]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        if not advice_msgs:
            ax.text(0.5,0.5,"✅ Biashara iko sawa — endelea hivyo!",
                    ha="center",va="center",fontsize=10,color=C["positive"],
                    transform=ax.transAxes,fontweight="bold"); return

        ax.text(0.01,0.96,"💬 USHAURI / ADVICE", ha="left",va="top",
                fontsize=8,fontweight="bold",color=C["header_bg"],transform=ax.transAxes)
        bh = 0.80 / max(len(advice_msgs),1)
        for i,msg in enumerate(advice_msgs):
            yt = 0.88 - i*bh
            colour = (C["danger"]    if any(w in msg for w in ["⚠️","📉","🔴"])
                      else C["positive"] if any(w in msg for w in ["🎉","💪","🌟","🏆","🚀"])
                      else C["accent"])
            ax.add_patch(FancyBboxPatch((0.01,yt-bh+0.01),0.98,bh-0.015,
                boxstyle="round,pad=0.015", facecolor=colour+"15",
                edgecolor=colour, linewidth=0.8,
                transform=ax.transAxes, clip_on=True))
            ax.text(0.025,yt-bh/2, textwrap.fill(msg,width=110),
                    ha="left",va="center",fontsize=7,color=C["text"],
                    transform=ax.transAxes)
        ax.text(0.99,0.01,
                f"Imezalishwa {datetime.now().strftime('%d %b %Y %H:%M')} | Duka Bot",
                ha="right",va="bottom",fontsize=5.5,color=C["subtext"],
                style="italic",transform=ax.transAxes)

_image_builder = BalanceBookImageBuilder()

# ══════════════════════════════════════════════
# ADVICE ENGINE  (Swahili + English mix)
# ══════════════════════════════════════════════
class SmartAdviceEngine:
    """
    Gamified crowdsourcing + LLM story scoring + WhatsApp Channel queue.

    Supabase tables required:
      crowd_profiles  (shop_id PK, location, shop_type, revenue, customers,
                       challenge, points, streak, last_share, submissions, reputation)
      crowd_tips      (id PK, shop_id, tip, location, shop_type,
                       upvotes, downvotes, reactions, score, featured, channel_posted,
                       created_at)
      channel_queue   (id PK, tip_id, story_text, score, status, created_at)
                       status: pending | posted | rejected
    """

    _COMMANDS = [
        (["/share", "/s"],      "_cmd_share"),
        (["/tip", "/t"],        "_cmd_tip"),
        (["/feature", "/post"], "_cmd_feature"),
        (["/verify", "/v"],     "_cmd_verify"),
        (["/upvote"],           "_cmd_upvote"),
        (["/downvote"],         "_cmd_downvote"),
        (["/benchmark", "/b"],  "_cmd_benchmark"),
        (["/profile", "/my"],   "_cmd_profile"),
        (["/crowdhelp", "/ch"], "_cmd_help"),
    ]

    def match_command(self, message: str) -> Optional[tuple]:
        msg = message.lower().strip()
        for prefixes, handler in self._COMMANDS:
            for p in prefixes:
                if msg == p or msg.startswith(p + " "):
                    return handler, message.strip()
        return None

    async def _cmd_share(self, phone: str, message: str) -> str:
        body = re.sub(r"^/s(?:hare)?\s*", "", message, flags=re.IGNORECASE).strip()
        data = {}
        for pair in re.findall(r"(\w+)=([^|]+)", body):
            data[pair[0].strip().lower()] = pair[1].strip()
        if not data.get("revenue"):
            nums = re.findall(r"\d{3,}", body)
            if nums:
                data["revenue"] = nums[0]

        if not data:
            return (
                "❌ Mfano:\n"
                "/share mapato=12000|wateja=35|mahali=Nairobi Eastlands"
                "|aina=butchery|changamoto=bei ya nyama juu\n\n"
                "Sehemu: mapato, wateja, mahali, aina, changamoto"
            )

        existing = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        prev = existing.data[0] if existing.data else {}

        location  = data.get("mahali",    prev.get("location",  "unknown"))
        shop_type = data.get("aina",      prev.get("shop_type", "duka"))
        revenue   = self._to_float(data.get("mapato",    prev.get("revenue")))
        customers = self._to_float(data.get("wateja",    prev.get("customers")))
        challenge = data.get("changamoto", prev.get("challenge"))

        last_share = prev.get("last_share")
        streak = prev.get("streak", 0)
        if last_share:
            days_since = (datetime.now() - dateparser.parse(last_share)).days
            streak = (streak + 1) if days_since == 1 else (1 if days_since > 1 else streak)
        else:
            streak = 1

        points      = prev.get("points", 0) + 30
        submissions = prev.get("submissions", 0) + 1

        row = {
            "shop_id":     phone,
            "location":    location,
            "shop_type":   shop_type,
            "revenue":     revenue,
            "customers":   customers,
            "challenge":   challenge,
            "points":      points,
            "streak":      streak,
            "last_share":  datetime.now().isoformat(),
            "submissions": submissions,
            "reputation":  prev.get("reputation", 1.0),
        }

        if existing.data:
            await db.execute("crowd_profiles", "update",
                data=row, match={"key": "shop_id", "value": phone})
        else:
            await db.execute("crowd_profiles", "insert", data=row)

        await db.execute("shops", "update",
            data={"location": location, "shop_type": shop_type},
            match={"key": "id", "value": phone})

        avg_rev = await self._avg_metric("revenue", location)
        top_ch  = await self._top_challenge(location)

        return (
            f"✅ Asante! Data imehifadhiwa bila jina.\n\n"
            f"⭐ Pointi zako: {points}  🔥 Streak: {streak} siku\n\n"
            f"📊 Wastani wa eneo lako ({location}):\n"
            f"  💰 Mapato: KES {avg_rev:,.0f}\n"
            f"  ⚠️  Changamoto: {top_ch}\n\n"
            f"Tuma /tip [hadithi yako] kushiriki uzoefu wako!\n"
            f"Au /benchmark kuona jinsi unavyolinganisha."
        )

    async def _cmd_tip(self, phone: str, message: str) -> str:
        tip_text = re.sub(r"^/t(?:ip)?\s*", "", message, flags=re.IGNORECASE).strip()

        if len(tip_text) < 15:
            return (
                "Toa uzoefu mrefu zaidi (angalau herufi 15).\n"
                "Mfano: /tip Niliongeza packaging safi kwa ofisi → mauzo +25%"
            )

        profile = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        prof = profile.data[0] if profile.data else {}

        tip_id = hashlib.md5(f"{phone}{tip_text}{datetime.now().isoformat()}".encode()).hexdigest()[:10]

        await db.execute("crowd_tips", "insert", data={
            "id":             tip_id,
            "shop_id":        phone,
            "tip":            tip_text,
            "location":       prof.get("location", "unknown"),
            "shop_type":      prof.get("shop_type", "duka"),
            "upvotes":        0,
            "reactions":      0,
            "score":          0.0,
            "featured":       False,
            "channel_posted": False,
            "created_at":     datetime.now().isoformat(),
        })

        new_points = prof.get("points", 0) + 50
        if profile.data:
            await db.execute("crowd_profiles", "update",
                data={"points": new_points},
                match={"key": "shop_id", "value": phone})

        asyncio.create_task(self._score_and_queue(tip_id, tip_text, prof))

        return (
            f"🙏 Tip imepokewa! ID: *{tip_id}*\n\n"
            f"⭐ Pointi zako: {new_points}\n\n"
            f"Bot itaangalia kama hadithi yako inafaa kushirikiwa\n"
            f"na jamii kwenye Channel. Utajulishwa!\n\n"
            f"Tuma /feature {tip_id} kama unataka kushirikiwa mapema."
        )

    async def _cmd_feature(self, phone: str, message: str) -> str:
        parts = message.split()
        if len(parts) < 2:
            return "Tumia: /feature <tip_id>"

        tip_id = parts[1]
        tips = await db.execute("crowd_tips", "select", eq={"id": tip_id})
        if not tips.data:
            return "❌ Tip haikupatikana."
        if tips.data[0]["shop_id"] != phone:
            return "❌ Hii si tip yako."
        if tips.data[0].get("channel_posted"):
            return "✅ Tip hii tayari imechapishwa kwenye Channel!"

        profile = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        prof = profile.data[0] if profile.data else {}
        asyncio.create_task(self._score_and_queue(tip_id, tips.data[0]["tip"], prof, force=True))

        new_pts = prof.get("points", 0) + 100
        if profile.data:
            await db.execute("crowd_profiles", "update",
                data={"points": new_pts},
                match={"key":"shop_id", "value": phone})

        return (
            f"🎉 Ombi lako limepokewa!\n\n"
            f"Bot itaangalia ubora wa hadithi yako na kuichapisha\n"
            f"kwenye Channel ikipita kiwango.\n\n"
            f"⭐ Pointi zako: {new_pts}\n"
            f"📢 Channel: {config.channel_link}"
        )

    async def _score_and_queue(self, tip_id: str, tip_text: str, prof: Dict, force: bool = False):
        try:
            client = AsyncOpenAI(api_key=config.openai_api_key)
            scoring_prompt = f"""
You are evaluating a story/tip from a Kenyan small business owner (duka, butchery, or shop).
Score it on these 4 axes, each from 1 to 10:

1. SPECIFICITY — has real numbers, named products, before/after, specific location
2. ACTIONABILITY — another shopkeeper can immediately replicate this
3. LOCAL_RELEVANCE — references Kenyan realities: M-Pesa, boda bodas, inflation, specific towns, local products
4. GROWTH_SIGNAL — promotes business growth OR warns about a real cautionary risk (both qualify equally)

Story: "{tip_text}"
Location: {prof.get("location", "unknown")}
Shop type: {prof.get("shop_type", "duka")}

Respond ONLY with valid JSON, no extra text:
{{"specificity": 0, "actionability": 0, "local_relevance": 0, "growth_signal": 0, "reason": "short reason"}}
"""
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            scores = json.loads(raw)

            axes = ["specificity", "actionability", "local_relevance", "growth_signal"]
            avg_score = sum(scores.get(a, 0) for a in axes) / 4

            await db.execute("crowd_tips", "update",
                data={"score": round(avg_score, 2)},
                match={"key": "id", "value": tip_id})

            log.info(f"Tip {tip_id} scored {avg_score:.1f}/10 — {scores.get('reason', '')}")

            if avg_score >= config.story_score_threshold or force:
                await self._queue_for_channel(tip_id, tip_text, avg_score, prof)

        except Exception as e:
            log.error(f"Scoring failed for tip {tip_id}: {e}")

    async def _queue_for_channel(self, tip_id: str, tip_text: str, score: float, prof: Dict):
        story = self._format_channel_story(tip_text, prof, score)

        await db.execute("channel_queue", "insert", data={
            "id":         str(uuid.uuid4()),
            "tip_id":     tip_id,
            "story_text": story,
            "score":      round(score, 2),
            "status":     "pending",
            "created_at": datetime.now().isoformat(),
        })

        await db.execute("crowd_tips", "update",
            data={"featured": True},
            match={"key": "id", "value": tip_id})

        if config.channel_live:
            await self._post_to_channel(story, tip_id)
        else:
            log.info(f"Tip {tip_id} queued (CHANNEL_LIVE=false). Score: {score:.1f}")

        tips_r = await db.execute("crowd_tips", "select", eq={"id": tip_id})
        if tips_r.data:
            owner_phone = tips_r.data[0].get("shop_id", "")
            if owner_phone:
                await self._notify_owner(owner_phone, tip_id, score)

    def _format_channel_story(self, tip_text: str, prof: Dict, score: float) -> str:
        location  = prof.get("location", "Kenya")
        shop_type = prof.get("shop_type", "Duka")
        return (
            f"💡 *HADITHI YA BIASHARA — {shop_type.upper()} | {location.upper()}*\n\n"
            f"{tip_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Imepigwa kura na wafanyabiashara wenzako\n"
            f"🤝 Jiunge: {config.channel_link}\n"
            f"💬 Shiriki uzoefu wako → DM Smart Duka Bot"
        )

    async def _post_to_channel(self, story_text: str, tip_id: str):
        url = f"https://graph.facebook.com/v19.0/{config.meta_phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to":                config.whatsapp_channel_id,
            "type":              "text",
            "text":              {"body": story_text},
        }
        headers = {
            "Authorization": f"Bearer {config.meta_access_token}",
            "Content-Type":  "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                    result = await r.json()
                    if r.status == 200:
                        await db.execute("crowd_tips", "update",
                            data={"channel_posted": True},
                            match={"key": "id", "value": tip_id})
                        await db.execute("channel_queue", "update",
                            data={"status": "posted"},
                            match={"key": "tip_id", "value": tip_id})
                        log.info(f"Tip {tip_id} posted to channel ✅")
                    else:
                        log.error(f"Meta API error for tip {tip_id}: {result}")
        except Exception as e:
            log.error(f"Channel post failed for tip {tip_id}: {e}")

    async def _notify_owner(self, phone: str, tip_id: str, score: float):
        prof_r = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        pts = prof_r.data[0].get("points", 0) if prof_r.data else 0
        msg = (
            f"🎉 *Hongera!* Hadithi yako imepita kiwango!\n\n"
            f"ID: {tip_id}  |  Score: {score:.1f}/10\n"
            f"⭐ Pointi zako: {pts}\n\n"
            + (f"✅ Imechapishwa kwenye Channel!\n📢 {config.channel_link}"
               if config.channel_live else
               "⏳ Itachapishwa Channel hivi karibuni.")
        )
        try:
            loop = asyncio.get_event_loop()
            twilio_client = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)
            await loop.run_in_executor(
                None,
                lambda: twilio_client.messages.create(
                    body=msg,
                    from_=config.twilio_whatsapp_number,
                    to=f"whatsapp:{phone}",
                )
            )
        except Exception as e:
            log.error(f"Owner notify failed for {phone}: {e}")

    async def _cmd_verify(self, phone: str, message: str) -> str:
        parts = message.split()
        if len(parts) < 2:
            return "Tumia: /verify <tip_id>"
        tip_id = parts[1]
        tips = await db.execute("crowd_tips", "select", eq={"id": tip_id})
        if not tips.data:
            return "❌ Tip haikupatikana."
        t = tips.data[0]
        status = (
            "✅ Imechapishwa kwenye Channel" if t.get("channel_posted") else
            "⏳ Inasubiri Channel"            if t.get("featured")       else
            "🔍 Bado haijapimwa"
        )
        return (
            f"🔍 *Thibitisha Tip:*\n\n"
            f"📝 \"{t['tip']}\"\n\n"
            f"📍 {t['location']}  |  🏪 {t['shop_type']}\n"
            f"👍 {t['upvotes']}  |  ❤️ {t['reactions']}  |  📊 Score: {t.get('score', 0):.1f}/10\n"
            f"📢 {status}\n\n"
            f"Je, inakusaidia?\n"
            f"/upvote {tip_id}  au  /downvote {tip_id}"
        )

    async def _cmd_upvote(self, phone: str, message: str) -> str:
        return await self._vote(message, "upvotes")

    async def _cmd_downvote(self, phone: str, message: str) -> str:
        return await self._vote(message, "downvotes")

    async def _vote(self, message: str, field: str) -> str:
        parts = message.split()
        if len(parts) < 2:
            return f"Tumia: /{field.rstrip('s')} <tip_id>"
        tip_id = parts[1]
        tips = await db.execute("crowd_tips", "select", eq={"id": tip_id})
        if not tips.data:
            return "❌ Tip haikupatikana."
        new_val = tips.data[0].get(field, 0) + 1
        await db.execute("crowd_tips", "update",
            data={field: new_val}, match={"key": "id", "value": tip_id})
        icon = "👍" if field == "upvotes" else "👎"
        return f"✅ Kura imehesabiwa! {icon} {new_val}"

    async def _cmd_benchmark(self, phone: str, message: str) -> str:
        prof_r = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        if not prof_r.data:
            return "📊 Shiriki data kwanza:\n/share mapato=5000|wateja=45|mahali=Nairobi"

        me        = prof_r.data[0]
        location  = me.get("location", "unknown")
        shop_type = me.get("shop_type", "duka")
        my_rev    = me.get("revenue")   or 0
        my_cust   = me.get("customers") or 0

        avg_rev  = await self._avg_metric("revenue",   location)
        avg_cust = await self._avg_metric("customers", location)
        count    = await self._peer_count(location)
        top_ch   = await self._top_challenge(location)

        return (
            f"📊 *Benchmark — {shop_type} | {location}*\n"
            f"_(Biashara {count} sawa nawe)_\n\n"
            f"💰 Mapato:  Wewe KES {my_rev:,.0f}  |  Wastani KES {avg_rev:,.0f}  {self._diff_label(my_rev, avg_rev)}\n"
            f"👥 Wateja:  Wewe {my_cust:.0f}  |  Wastani {avg_cust:.0f}  {self._diff_label(my_cust, avg_cust)}\n"
            f"⚠️  Changamoto kubwa: {top_ch}\n\n"
            f"Tuma /tip [uzoefu wako] kusaidia wenzako!"
        )

    async def _cmd_profile(self, phone: str, message: str) -> str:
        prof_r = await db.execute("crowd_profiles", "select", eq={"shop_id": phone})
        if not prof_r.data:
            return "Bado huna data. Tuma /share kuanza!"
        me = prof_r.data[0]
        tips_r = await db.execute("crowd_tips", "select", eq={"shop_id": phone})
        posted = sum(1 for t in tips_r.data if t.get("channel_posted"))
        return (
            f"👤 *Profaili Yako*\n\n"
            f"📍 {me.get('location', '?')}  |  🏪 {me.get('shop_type', '?')}\n"
            f"⭐ Pointi: {me.get('points', 0)}\n"
            f"🔥 Streak: {me.get('streak', 0)} siku\n"
            f"📤 Tips ulizotuma: {len(tips_r.data)}\n"
            f"📢 Zilizochapishwa Channel: {posted}\n\n"
            f"Tuma /tip kushiriki hadithi yako!"
        )

    async def _cmd_help(self, phone: str, message: str) -> str:
        return (
            f"🤝 *Smart Duka — Crowdsourcing*\n\n"
            f"📊 /share mapato=5000|wateja=45|mahali=Meru|aina=butchery\n"
            f"   _Shiriki takwimu zako (bila jina)_\n\n"
            f"💡 /tip [hadithi/uzoefu wako]\n"
            f"   _Bot itapima ubora na kuchapisha Channel_\n\n"
            f"🌟 /feature <tip_id> — omba kuchapishwa haraka\n"
            f"🔍 /verify <tip_id>  — angalia tip ya mtu\n"
            f"👍 /upvote <tip_id>  au  /downvote <tip_id>\n"
            f"📈 /benchmark        — linganisha na wenzako\n"
            f"👤 /profile          — pointi na streak yako\n\n"
            f"📢 Channel: {config.channel_link}\n"
            f"_Data yote huhifadhiwa bila jina. Bure kabisa._"
        )

    async def get_top_tips(self, location: str, shop_type: str, limit: int = 3) -> List[str]:
        rows = await db.execute("crowd_tips", "select", eq={"location": location})
        if not rows.data:
            rows = await db.execute("crowd_tips", "select", eq={"shop_type": shop_type})
        sorted_tips = sorted(
            rows.data,
            key=lambda t: (t.get("score", 0) * 0.5 + t.get("reactions", 0) * 0.3 + t.get("upvotes", 0) * 0.2),
            reverse=True,
        )
        return [t["tip"] for t in sorted_tips[:limit]]

    async def flush_channel_queue(self) -> int:
        queue = await db.execute("channel_queue", "select", eq={"status": "pending"})
        posted = 0
        for item in queue.data:
            await self._post_to_channel(item["story_text"], item["tip_id"])
            posted += 1
            await asyncio.sleep(2)
        return posted

    @staticmethod
    def _to_float(val) -> Optional[float]:
        try:
            return float(str(val).replace(",", "")) if val is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _diff_label(mine: float, avg: float) -> str:
        if not avg or not mine:
            return ""
        pct = (mine - avg) / avg * 100
        if pct >= 5:  return f"🔺 +{pct:.0f}%"
        if pct <= -5: return f"🔻 {pct:.0f}%"
        return "✅ Sawa"

    async def _avg_metric(self, field: str, location: str) -> float:
        rows = await db.execute("crowd_profiles", "select", eq={"location": location})
        vals = [r[field] for r in rows.data if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    async def _peer_count(self, location: str) -> int:
        rows = await db.execute("crowd_profiles", "select", eq={"location": location})
        return len(rows.data)

    async def _top_challenge(self, location: str) -> str:
        rows = await db.execute("crowd_profiles", "select", eq={"location": location})
        challenges = [r["challenge"] for r in rows.data if r.get("challenge")]
        return Counter(challenges).most_common(1)[0][0] if challenges else "Bei ya juu"


advice_engine = SmartAdviceEngine()

# ══════════════════════════════════════════════
# SUPABASE STORAGE UPLOADER
# ══════════════════════════════════════════════
class StorageUploader:
    BUCKET = "balance-reports"

    def __init__(self):
        self.sb = create_client(config.supabase_url, config.supabase_key)

    def _ensure_bucket(self):
        try:
            self.sb.storage.create_bucket(
                self.BUCKET,
                options={"public": True, "fileSizeLimit": 5_000_000}
            )
        except Exception:
            pass

    async def upload(self, png_bytes: bytes, shop_id: str, label: str) -> str:
        loop = asyncio.get_event_loop()
        def _up():
            self._ensure_bucket()
            filename = f"{shop_id}/{label.replace(' ','_')}_{uuid.uuid4().hex[:8]}.png"
            self.sb.storage.from_(self.BUCKET).upload(
                path=filename, file=png_bytes,
                file_options={"content-type":"image/png","upsert":"true"}
            )
            return f"{config.supabase_url}/storage/v1/object/public/{self.BUCKET}/{filename}"
        return await loop.run_in_executor(None, _up)

_storage = StorageUploader()

# ══════════════════════════════════════════════
# BALANCE BOOK REPORTER
# ══════════════════════════════════════════════
class BalanceBookReporter:
    def __init__(self, shop_id: str, phone: str):
        self.shop_id   = shop_id
        self.phone     = phone
        self.analytics = BusinessAnalytics(shop_id)
        self.twilio    = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)

    async def _shop_name(self) -> str:
        r = await db.execute("shops", "select", eq={"id": self.shop_id})
        return r.data[0]["name"] if r.data else self.shop_id

    async def _txns_for_month(self, year: int, month: int) -> List[Dict]:
        r     = await db.execute("transactions","select",eq={"shop_id":self.shop_id},
                                 order={"column":"created_at","desc":False})
        first = date(year, month, 1)
        last  = date(year+1,1,1)-timedelta(1) if month==12 else date(year,month+1,1)-timedelta(1)
        return [t for t in r.data if first <= dateparser.parse(t["created_at"]).date() <= last]

    def _weekly_rows(self, txns: List[Dict]) -> List[Dict]:
        weeks: Dict = defaultdict(lambda:{"revenue":0,"expenses":0,"transactions":0})
        for t in txns:
            w = (dateparser.parse(t["created_at"]).date().day - 1) // 7 + 1
            if t["type"]=="sale":
                weeks[w]["revenue"]+=t["amount"]; weeks[w]["transactions"]+=1
            elif t["type"]=="expense":
                weeks[w]["expenses"]+=t["amount"]
        rows = []
        for w in sorted(weeks):
            r=weeks[w]["revenue"]; e=weeks[w]["expenses"]; p=r-e
            rows.append({"week":f"Wiki {w}","revenue":r,"expenses":e,"profit":p,
                         "margin":(p/r*100) if r else 0,"transactions":weeks[w]["transactions"]})
        return rows

    def _daily_data(self, txns: List[Dict]) -> Dict[str,Dict]:
        daily: Dict = defaultdict(lambda:{"revenue":0,"expenses":0})
        for t in txns:
            k = dateparser.parse(t["created_at"]).date().isoformat()
            if t["type"]=="sale":    daily[k]["revenue"]  +=t["amount"]
            elif t["type"]=="expense": daily[k]["expenses"]+=t["amount"]
        return dict(daily)

    async def _low_stock(self) -> List[str]:
        r = await db.execute("products","select",eq={"shop_id":self.shop_id})
        return [p["name"] for p in r.data if p.get("quantity",0) < p.get("reorder_level",10)]

    async def send_monthly_report(self, year: int = None, month: int = None) -> bool:
        now   = datetime.now()
        year  = year  or now.year
        month = month or now.month
        label = datetime(year, month, 1).strftime("%B %Y")
        log.info(f"Generating monthly report for {self.shop_id} — {label}")

        try:
            shop_name  = await self._shop_name()
            txns       = await self._txns_for_month(year, month)
            metrics    = await self.analytics.calculate_metrics()
            insights   = await self.analytics.generate_insights()
            market     = await self.analytics.compare_with_market()
            low_stock  = await self._low_stock()
            advice     = _advice_engine.generate(metrics, insights, market, low_stock)
            weekly     = self._weekly_rows(txns)
            daily      = self._daily_data(txns)

            png        = _image_builder.build(shop_name, label, weekly, daily, metrics, advice)
            url        = await _storage.upload(png, self.shop_id, label)

            caption = self._caption(shop_name, label, metrics, advice)
            loop    = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.twilio.messages.create(
                body=caption, media_url=[url],
                from_=config.twilio_whatsapp_number, to=f"whatsapp:{self.phone}"
            ))
            log.info(f"Monthly report sent to {self.phone}")
            return True
        except Exception as e:
            log.error(f"Monthly report failed: {e}", exc_info=True)
            return False

    def _caption(self, shop_name, label, metrics, advice):
        p    = metrics.get("profit",0)
        icon = "📈" if p >= 0 else "📉"
        lines = [
            f"{icon} *{shop_name} — {label}*","",
            f"💰 Mapato:  KES {metrics.get('revenue',0):,.0f}",
            f"💸 Matumizi: KES {metrics.get('expenses',0):,.0f}",
            f"📊 Faida:   KES {p:,.0f}  ({metrics.get('profit_margin',0):.1f}%)",
            f"💳 Deni:    KES {metrics.get('total_debt',0):,.0f}","",
        ]
        if advice:
            lines += ["💬 *Ushauri:*"] + [f"  {a}" for a in advice[:3]]
        lines.append("\n_Angalia picha kwa ripoti kamili_")
        return "\n".join(lines)

# ══════════════════════════════════════════════
# ON-DEMAND DAILY SNAPSHOT (text, instant)
# ══════════════════════════════════════════════
async def build_daily_snapshot(shop_id: str) -> str:
    """
    Returns a rich text snapshot for today.
    Called when a trader asks 'how is business today' etc.
    No image — instant text response.
    """
    analytics = BusinessAnalytics(shop_id)
    today     = await analytics.today_snapshot()
    leaks     = await analytics.identify_leaks()
    insights  = await analytics.generate_insights()
    market    = await analytics.compare_with_market()

    products  = await db.execute("products","select",eq={"shop_id":shop_id})
    low_stock = [p["name"] for p in products.data
                 if p.get("quantity",0) < p.get("reorder_level",10)]

    profit = today.get("profit",0)
    icon   = "📈" if profit >= 0 else "📉"

    lines = [
        f"{icon} *Leo Biashara — {datetime.now().strftime('%d %b %Y')}*","",
        f"💰 Mapato:      KES {today.get('revenue',0):,.0f}",
        f"💸 Matumizi:    KES {today.get('expenses',0):,.0f}",
        f"📊 Faida:       KES {profit:,.0f}",
        f"🛒 Miamala:     {today.get('transaction_count',0)}",
        f"💳 Deni Lote:   KES {today.get('total_debt',0):,.0f}","",
    ]

    if low_stock:
        lines += [f"⚠️ *Stock Chini:* {', '.join(low_stock[:4])}",""]

    if leaks:
        lines.append("🔍 *Matatizo:*")
        for lk in leaks[:2]:
            lines.append(f"  • {lk['description']}")
        lines.append("")

    if insights:
        lines.append("💡 *Fursa/Mwenendo:*")
        for ins in insights[:2]:
            lines.append(f"  • {ins['title']}: {ins['description']}")
        lines.append("")

    recs = (market.get("recommendations") or [])[:2]
    if recs:
        lines.append("📊 *Soko:*")
        for r in recs:
            lines.append(f"  • {r}")
        lines.append("")

    lines.append("_Andika 'ripoti ya mwezi' kupata ripoti kamili na picha._")
    return "\n".join(lines)

# ══════════════════════════════════════════════
# WHATSAPP BOT
# ══════════════════════════════════════════════
class WhatsAppBusinessBot:

    def _extract_product_name(self, message: str) -> str:
        for word in message.split():
            clean = word.strip(".,!?")
            if clean and clean not in _NAME_STOPWORDS and len(clean) > 2 and clean[0].isupper():
                return clean
        tokens = message.split()
        return tokens[1] if len(tokens) > 1 else "Unknown"

    @staticmethod
    def _extract_phone(message: str) -> Optional[str]:
        m = re.search(r"(\+?254\d{9}|07\d{8}|01\d{8})", message)
        return m.group(1) if m else None

    async def process_message(self, phone: str, message: str) -> str:
        try:
            shop = await self._get_or_create_shop(phone)
            if not await self._is_active(shop["id"]):
                if "pay" in message.lower():
                    return await self._handle_payment(phone, shop)
                return "Usajili wako umekwisha. Jibu *PAY* kuhuisha."
                
            crowd = advice_engine.match_command(message)
            if crowd:
                handler_name, raw = crowd
                return await getattr(advice_engine, handler_name)(phone, raw)

            parsed    = await nlp.parse_message(message)
            shop_id   = shop["id"]
            intent    = parsed["intent"]
            entities  = parsed["entities"]
            analytics = BusinessAnalytics(shop_id)

            handlers = {
                "add_product":    self._handle_add_product,
                "sell":           self._handle_sale,
                "debt":           self._handle_debt,
                "payment":        self._handle_payment_received,
                "view":           self._handle_view,
                "snapshot":       self._handle_snapshot,
                "insight":        self._handle_insights,
                "expense":        self._handle_expense,
                "alert":          self._handle_alerts,
                "compare":        self._handle_comparison,
                "forecast":       self._handle_forecast,
                "monthly_report": self._handle_monthly_report,
            }

            handler = handlers.get(intent, self._handle_unknown)

            # pass analytics to handlers that need it
            if intent in ("view","insight","compare","forecast","snapshot","monthly_report"):
                return await handler(shop_id, entities, message, analytics)
            return await handler(shop_id, entities, message)

        except Exception as e:
            log.error(f"process_message error: {e}", exc_info=True)
            return "Samahani, kuna hitilafu. Tafadhali jaribu tena."

    # ── shop helpers ─────────────────────────
    async def _get_or_create_shop(self, phone: str) -> Dict:
        r = await db.execute("shops","select",eq={"id":phone})
        if not r.data:
            await db.execute("shops","insert",data={
                "id":phone,"name":f"Duka {phone[-6:]}",
                "created_at":datetime.now().isoformat()})
            await db.execute("subscriptions","insert",data={
                "shop_id":phone,"plan":"trial",
                "expires_at":(datetime.now()+timedelta(days=14)).isoformat()})
            return {"id":phone,"name":f"Duka {phone[-6:]}"}
        return r.data[0]

    async def _is_active(self, shop_id: str) -> bool:
        r = await db.execute("subscriptions","select",eq={"shop_id":shop_id})
        if not r.data: return False
        return dateparser.parse(r.data[0]["expires_at"]) > datetime.now()

    # ── inventory ────────────────────────────
    async def _handle_add_product(self, shop_id, entities, message):
        name  = entities.get("name") or self._extract_product_name(message)
        qty   = int(entities.get("quantity",0))
        price = float(entities.get("amount",0))

        ex = await db.execute("products","select",eq={"shop_id":shop_id,"name":name})
        if ex.data:
            prev    = ex.data[0]
            new_qty = prev["quantity"] + qty
            await db.execute("products","update",
                data={"quantity":new_qty,"price":price,"updated_at":datetime.now().isoformat()},
                match={"key":"id","value":prev["id"]})
            await db.execute("inventory_movements","insert",data={
                "shop_id":shop_id,"product_id":prev["id"],"type":"purchase",
                "quantity":qty,"previous_quantity":prev["quantity"],"new_quantity":new_qty})
            return f"✅ *{name}* imesasishwa: +{qty} units. Jumla: {new_qty}. Bei: KES {price:,.0f}"

        await db.execute("products","insert",data={
            "shop_id":shop_id,"name":name,"quantity":qty,
            "price":price,"cost_price":round(price*0.7,2)})
        return f"✅ *{name}* imeongezwa: {qty} units @ KES {price:,.0f}"

    async def _handle_sale(self, shop_id, entities, message):
        name     = entities.get("name") or self._extract_product_name(message)
        qty      = int(entities.get("quantity",1))
        amount   = float(entities.get("amount",0))
        customer = entities.get("customer_name","Mteja")

        r = await db.execute("products","select",eq={"shop_id":shop_id,"name":name})
        if not r.data:
            return f"❌ Bidhaa *{name}* haipatikani. Iongeze kwanza."
        p = r.data[0]
        if p["quantity"] < qty:
            return f"❌ Stock haitoshi! Kuna {p['quantity']} tu za {name}."
        if amount == 0:
            amount = p["price"] * qty

        new_qty = p["quantity"] - qty
        await db.execute("products","update",
            data={"quantity":new_qty,"updated_at":datetime.now().isoformat()},
            match={"key":"id","value":p["id"]})
        await db.execute("transactions","insert",data={
            "shop_id":shop_id,"type":"sale","amount":amount,
            "category":name,"description":f"Uliuza {qty} {name}","reference_id":customer})
        await db.execute("inventory_movements","insert",data={
            "shop_id":shop_id,"product_id":p["id"],"type":"sale",
            "quantity":-qty,"previous_quantity":p["quantity"],"new_quantity":new_qty})
        return f"✅ Mauzo: {qty}× *{name}* = KES {amount:,.0f}\nMteja: {customer}"

    async def _handle_debt(self, shop_id, entities, message):
        name   = entities.get("name")
        amount = entities.get("amount")
        if not name or not amount:
            return "Tafadhali andika: *Deni [jina] [kiasi]*"

        ex = await db.execute("debts","select",
            eq={"shop_id":shop_id,"customer_name":name,"status":"pending"})
        if ex.data:
            debt      = ex.data[0]
            new_amount= debt["amount"] + amount
            await db.execute("debts","update",
                data={"amount":new_amount,"updated_at":datetime.now().isoformat()},
                match={"key":"id","value":debt["id"]})
            return f"⚠️ Deni la {name} limesasishwa: +KES {amount:,.0f}. Jumla: KES {new_amount:,.0f}"

        phone = self._extract_phone(message)
        await db.execute("debts","insert",data={
            "shop_id":shop_id,"customer_name":name,"customer_phone":phone or "",
            "amount":amount,"original_amount":amount,
            "due_date":(datetime.now()+timedelta(days=7)).date().isoformat()})
        if phone:
            await reminder_system.send_reminder({
                "customer_name":name,"customer_phone":phone,
                "amount":amount,"created_at":datetime.now().isoformat(),"reminder_count":0}, 0)
        return f"✅ Deni limerekodiwa: {name} anadaiwa KES {amount:,.0f}"

    async def _handle_payment_received(self, shop_id, entities, message):
        name   = entities.get("name")
        amount = entities.get("amount")
        if not name or not amount:
            return "Tafadhali andika: *Malipo [jina] [kiasi]*"

        r = await db.execute("debts","select",
            eq={"shop_id":shop_id,"customer_name":name,"status":"pending"})
        if not r.data:
            return f"Hakuna deni la {name}."
        debt = r.data[0]

        await db.execute("debt_payments","insert",data={
            "debt_id":debt["id"],"amount":amount,
            "payment_method":"cash","reference":"WhatsApp"})
        remaining = debt["amount"] - amount
        if remaining <= 0:
            await db.execute("debts","update",
                data={"status":"paid","amount":0,"updated_at":datetime.now().isoformat()},
                match={"key":"id","value":debt["id"]})
            return f"✅ {name} amelipa KES {amount:,.0f} kikamilifu. Deni limefutwa! 🎉"
        await db.execute("debts","update",
            data={"amount":remaining,"updated_at":datetime.now().isoformat()},
            match={"key":"id","value":debt["id"]})
        return f"✅ Malipo ya KES {amount:,.0f} kutoka {name}. Baki: KES {remaining:,.0f}"

    async def _handle_expense(self, shop_id, entities, message):
        amount   = entities.get("amount")
        category = entities.get("category","Nyingine")
        if not amount:
            return "Tafadhali andika kiasi: *Gharama 500 usafiri*"
        await db.execute("transactions","insert",data={
            "shop_id":shop_id,"type":"expense",
            "amount":amount,"category":category,"description":message})
        return f"✅ Gharama imerekodiwa: {category} — KES {amount:,.0f}"

    # ── view / analytics ─────────────────────
    async def _handle_view(self, shop_id, entities, message, analytics):
        msg = message.lower()
        if "debt" in msg or "deni" in msg:
            debts = await db.execute("debts","select",eq={"shop_id":shop_id,"status":"pending"})
            if not debts.data: return "Hakuna madeni yanayosubiri 📝"
            lines = ["*Madeni Yanayosubiri*\n"]
            for d in debts.data:
                age = (datetime.now() - dateparser.parse(d["created_at"])).days
                lines.append(f"👤 {d['customer_name']}: KES {d['amount']:,.0f} ({age} siku)")
            lines.append(f"\n*Jumla: KES {sum(d['amount'] for d in debts.data):,.0f}*")
            return "\n".join(lines)

        if "stock" in msg or "inventory" in msg or "bidhaa" in msg:
            products = await db.execute("products","select",eq={"shop_id":shop_id})
            if not products.data: return "Hakuna bidhaa ghalani."
            lines = ["*Hali ya Stock*\n"]
            for p in products.data:
                status = "⚠️ CHINI" if p["quantity"] < p.get("reorder_level",10) else "✓ SAWA"
                lines.append(f"📦 {p['name']}: {p['quantity']} units @ KES {p['price']:,.0f}  {status}")
            return "\n".join(lines)

        m = await analytics.calculate_metrics()
        return (
            f"*Muhtasari wa Biashara*\n\n"
            f"💰 Mapato:      KES {m['revenue']:,.0f}\n"
            f"💸 Matumizi:    KES {m['expenses']:,.0f}\n"
            f"📈 Faida:       KES {m['profit']:,.0f}\n"
            f"🎯 Margin:      {m['profit_margin']:.1f}%\n"
            f"📦 Thamani Stock: KES {m['inventory_value']:,.0f}\n"
            f"💳 Deni Lote:  KES {m['total_debt']:,.0f}\n"
            f"🛒 Mauzo ya Kawaida: KES {m['average_transaction']:,.0f}"
        )

    async def _handle_snapshot(self, shop_id, entities, message, analytics):
        """Instant today's performance — always ready."""
        return await build_daily_snapshot(shop_id)

    async def _handle_insights(self, shop_id, entities, message, analytics):
        leaks    = await analytics.identify_leaks()
        insights = await analytics.generate_insights()
        lines    = ["*📊 Ripoti ya Akili ya Biashara*\n"]
        if leaks:
            lines.append("*⚠️ Matatizo ya Fedha:*")
            for lk in leaks:
                lines += [f"• {lk['description']}", f"  💡 {lk['suggestion']}"]
        if insights:
            lines.append("\n*💡 Fursa:*")
            for ins in insights:
                lines.append(f"• {ins['title']}: {ins['description']}")
                if ins.get("suggestion"):
                    lines.append(f"  💡 {ins['suggestion']}")
        if not leaks and not insights:
            lines.append("✓ Kila kitu kiko sawa! Hongera! 🎉")

        # ── enrich with top crowd tips ─────────
        shop_r    = await db.execute("shops", "select", eq={"id": shop_id})
        shop_d    = shop_r.data[0] if shop_r.data else {}
        location  = shop_d.get("location", "unknown")
        shop_type = shop_d.get("shop_type", "duka")
        top_tips  = await advice_engine.get_top_tips(location, shop_type)
        if top_tips:
            lines.append("\n*🤝 Wafanyabiashara Wenzako Wanasema:*")
            for tip in top_tips:
                lines.append(f"• {tip}")
            lines.append(f"\n_Tuma /tip kushiriki uzoefu wako!_")

        return "\n".join(lines)
    async def _handle_alerts(self, shop_id, entities, message, *_):
        products  = await db.execute("products","select",eq={"shop_id":shop_id})
        low_stock = [p for p in products.data if p["quantity"] < p.get("reorder_level",10)]
        if not low_stock:
            return "✓ Hakuna tahadhari za stock! Bidhaa zote zipo sawa."
        lines = ["*⚠️ Tahadhari za Stock Chini*\n"]
        for p in low_stock:
            suggest = p.get("reorder_level",10)*2 - p["quantity"]
            lines += [f"📦 {p['name']}: {p['quantity']} zimebaki (agiza kwa {p.get('reorder_level',10)})",
                      f"   💡 Agiza takriban {suggest} units\n"]
        return "\n".join(lines)

    async def _handle_comparison(self, shop_id, entities, message, analytics):
        comp = await analytics.compare_with_market()
        if not comp.get("comparisons"):
            return comp.get("message","Hakuna data ya soko bado.")
        lines = ["*📈 Ulinganisho na Soko*\n"]
        for c in comp["comparisons"][:5]:
            icon = "🔺" if c["difference"]>0 else "🔻"
            dir_ = "juu ya" if c["difference"]>0 else "chini ya"
            lines += [f"*{c['product']}*",
                      f"  Bei yako: KES {c['your_price']:,.0f}",
                      f"  Bei ya soko: KES {c['market_avg']:,.0f}",
                      f"  {icon} {abs(c['percentage_diff']):.1f}% {dir_} soko\n"]
        if comp.get("recommendations"):
            lines.append("*💡 Mapendekezo:*")
            for r in comp["recommendations"][:3]:
                lines.append(f"• {r}")
        return "\n".join(lines)

    async def _handle_forecast(self, shop_id, entities, message, analytics):
        fc = await analytics.forecast_sales(7)
        if "error" in fc:
            return f"❌ {fc['error']}"
        lines = ["*📈 Utabiri wa Mauzo (Siku 7)*\n"]
        for d in fc["forecast"]:
            ds = dateparser.parse(d["date"]).strftime("%a, %d %b")
            lines.append(f"📅 {ds}: KES {d['predicted_sales']:,.0f}")
        lines.append(f"\n*Jumla ya utabiri: KES {fc['total_forecast']:,.0f}*")
        lines.append("\n📈 *Mwenendo: Ukuaji unatarajiwa!*" if fc["trend"]>0
                     else "\n📉 *Mwenendo: Mauzo yanaweza kupungua — fikiria promotion.*")
        return "\n".join(lines)

    async def _handle_monthly_report(self, shop_id, entities, message, analytics):
        """Triggered by trader manually; queues async image generation + send."""
        reporter = BalanceBookReporter(shop_id, shop_id)
        asyncio.create_task(reporter.send_monthly_report())
        return ("📊 *Ripoti ya Mwezi* inaandaliwa...\n"
                "Utapata picha kwenye WhatsApp ndani ya dakika moja. ⏳")

    # ── payment ──────────────────────────────
    async def _handle_payment(self, phone: str, shop: Dict) -> str:
        try:
            r = await mpesa.stk_push(phone, 500, f"duka_{phone}")
            if r.get("ResponseCode") == "0":
                return "✅ Ombi la M-Pesa limetumwa. Ingiza PIN kuhuisha kwa siku 30 (KES 500)."
            return "❌ Imeshindwa kutuma ombi la malipo. Jaribu tena."
        except Exception as e:
            log.error(f"STK push failed: {e}")
            return "Huduma ya malipo haipo sasa hivi."

    async def _handle_unknown(self, shop_id, entities, message):
        return (
            "*Amri Zinazopatikana:*\n\n"
            "📦 *Stock*\n"
            "  • Ongeza bidhaa Sugar 50 120\n"
            "  • Uliuza Sugar 5\n"
            "  • Angalia stock\n\n"
            "💰 *Madeni*\n"
            "  • Deni John 500\n"
            "  • Malipo John 300\n"
            "  • Angalia madeni\n\n"
            "📊 *Uchambuzi*\n"
            "  • Vipi biashara leo  ← hali ya leo\n"
            "  • Ushauri\n"
            "  • Linganisha\n"
            "  • Tabiri\n"
            "  • Tahadhari\n"
            "  • Ripoti ya mwezi  ← ripoti + picha\n\n"
            "💳 *Nyingine*\n"
            "  • PAY — huisha usajili\n"
            "  • Gharama 500 usafiri\n\n"
            "_Mfano: Deni John 500  |  Uliuza Sugar 2 400_"
        )

bot = WhatsAppBusinessBot()

# ══════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════
app = FastAPI(title="Duka WhatsApp Manager")

# ── Twilio webhook validation ─────────────────
async def _validate_twilio(request: Request):
    validator = RequestValidator(config.twilio_auth_token)
    form      = await request.form()
    sig       = request.headers.get("X-Twilio-Signature","")
    if not validator.validate(str(request.url), dict(form), sig):
        raise HTTPException(403, "Invalid Twilio signature")
    return dict(form)

# ── Cron auth ────────────────────────────────
def _cron_auth(secret: str = Form(...)):
    if secret != config.cron_secret:
        raise HTTPException(403, "Unauthorized")

# ── Endpoints ────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request, form=Depends(_validate_twilio)):
    phone   = form.get("From","").replace("whatsapp:","")
    message = (form.get("Body","") or "").strip()
    if not phone or not message:
        raise HTTPException(400, "Missing data")
    response_text = await bot.process_message(phone, message)
    resp = MessagingResponse()
    resp.message(response_text)
    return Response(content=str(resp), media_type="application/xml")

@app.post("/mpesa/callback")
async def mpesa_callback(request: Request):
    try:
        data     = await request.json()
        callback = data.get("Body",{}).get("stkCallback",{})
        if callback.get("ResultCode") == 0:
            # Extract phone and extend subscription by 30 days
            items  = {i["Name"]:i.get("Value") for i in
                      callback.get("CallbackMetadata",{}).get("Item",[])}
            phone  = str(items.get("PhoneNumber",""))
            if phone:
                await db.execute("subscriptions","update",
                    data={"expires_at":(datetime.now()+timedelta(days=30)).isoformat()},
                    match={"key":"shop_id","value":phone})
                log.info(f"Subscription extended for {phone}")
        return {"status":"received"}
    except Exception as e:
        log.error(f"M-Pesa callback error: {e}")
        return {"status":"error"}

@app.post("/cron/reminders", dependencies=[Depends(_cron_auth)])
async def cron_reminders():
    sent = await reminder_system.follow_up_debts()
    return {"reminders_sent": sent}

@app.post("/cron/scrape", dependencies=[Depends(_cron_auth)])
async def cron_scrape():
    count = await scraper.scrape_data()
    return {"records_scraped": count}
    
@app.post("/cron/flush-channel", dependencies=[Depends(_cron_auth)])
async def cron_flush_channel():
    """Flush pending channel queue — run after CHANNEL_LIVE=true is set."""
    posted = await advice_engine.flush_channel_queue()
    return {"posted": posted}
    
@app.post("/cron/monthly-report", dependencies=[Depends(_cron_auth)])
async def cron_monthly_report(background_tasks: BackgroundTasks):
    """Run on the last day of each month via external cron (e.g. cron-job.org)."""
    shops = await db.execute("shops","select")
    async def _send_all():
        for shop in shops.data:
            reporter = BalanceBookReporter(shop["id"], shop["id"])
            await reporter.send_monthly_report()
    background_tasks.add_task(_send_all)
    return {"status": "monthly reports queued", "shop_count": len(shops.data)}

@app.post("/report/{shop_id}")
async def on_demand_report(shop_id: str, background_tasks: BackgroundTasks):
    """Manually trigger a report for one shop (admin use)."""
    reporter = BalanceBookReporter(shop_id, shop_id)
    background_tasks.add_task(reporter.send_monthly_report)
    return {"status": "queued", "shop_id": shop_id}

@app.get("/health")
async def health():
    return {"status":"ok","time":datetime.now().isoformat()}

@app.on_event("shutdown")
async def shutdown():
    await scraper.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
