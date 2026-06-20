"""OCR and rule-based field extraction for ReceiptFlow.

This module is UI-independent: it only handles image preprocessing,
OCR text extraction, and parsing the OCR text into structured fields.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime

import cv2
import pytesseract


def find_tesseract_cmd() -> str:
    """Locate the Tesseract OCR binary on the current machine.

    Resolution order:
    1. The TESSERACT_CMD environment variable, if it points to a real file.
    2. "tesseract" found on the system PATH (works out of the box on most
       macOS/Linux installs, and on Windows if it was added to PATH).
    3. A handful of common per-OS install locations, as a fallback.
    4. Plain "tesseract" — pytesseract will raise a clear error if it is
       still not found, and the user can fix the path from the app sidebar.
    """
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and Path(env_path).exists():
        return env_path

    found = shutil.which("tesseract")
    if found:
        return found

    if sys.platform.startswith("win"):
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    else:
        candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

    for c in candidates:
        if Path(c).exists():
            return c

    return "tesseract"


pytesseract.pytesseract.tesseract_cmd = find_tesseract_cmd()

#Known store name patterns
KNOWN_STORES = re.compile(
    r"wal.?mart|costco|trader\s*joe|whole\s*foods|7.eleven|tesco|mydin|"
    r"giant|aeon|parkson|99\s*speedmart|kk\s*super|lotus|carrefour",
    re.IGNORECASE
)

#Lines that are clearly NOT store names
SKIP_HEADER = re.compile(
    r"always low price|save money|low price|open \d|supercenter|"
    r"receipt|invoice|bill|tax|official|welcome|thank|tel:|fax:|phone:|"
    r"cashier|operator|pos|terminal|reg|till|table|no\.|www\.|http|"
    r"GST|SST|registration|manager|survey|feedback|see back|id #|"
    r"give us|co\. no|business|\(\d{3}\)|\d{3}[-\s]\d{3}",
    re.IGNORECASE
)

#Lines that are summary rows, not items
SKIP_ITEM = re.compile(
    r"^(total|grand\s*total|\*+\s*total|sub.?total|net\s*sales|"
    r"cash|change\s*due|change|balance|tender|cash\s*tend|debit\s*tend|"
    r"tax[\s\d]|gst|sst|discount\s*given|discount|disc|rounding|tip|service\s*charge|"
    r"thank|please\s*come|receipt|cashier|operator|pos|dine|hall|table|welcome|"
    r"jumlah|bayaran|baki|diskaun|terima\s*kasih|pelanggan|pekerja|"
    r"served\s*by|prepared\s*by|void|refund|exchange|items?\s*sold|"
    r"eft\s*debit|us\s*debit|network|terminal|ref\s*#|appr|aid\s*[a-f0-9]|"
    r"member|check/member|savings?\s*catch|scan\s*with|store\s*receipt|"
    r"sold\s*items?|paid|net\s*sales|subtotal|open\s*\d|closed\s*bill|"
    r"low prices|every\s*day|introducing)(\W|$)",
    re.IGNORECASE
)

MONTHS_MY = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "januari":1,"februari":2,"mac":3,"april":4,"mei":5,
    "julai":7,"ogos":8,"september":9,"oktober":10,"november":11,"disember":12,
}

#Currency detection
def _detect_currency(full_text: str) -> str:
    if re.search(r"\bRM\b|\bMYR\b", full_text): return "MYR"
    if re.search(r"\bIDR\b|Rp\.?\s*\d", full_text): return "IDR"
    if re.search(r"\bSGD\b|S\$", full_text): return "SGD"
    if re.search(r"\bUSD\b|\bUS\$\b", full_text): return "USD"
    if re.search(r"\$", full_text): return "USD"
    return "MYR"


def preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
#Upscale small images
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=15)
#Auto threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_text(image_path: str) -> str:
    img = preprocess(image_path)
    try:
        text = pytesseract.image_to_string(img, lang="eng+chi_sim", config="--psm 4 --oem 3")
    except pytesseract.TesseractError:
        #The "chi_sim" language pack isn't installed on this machine.
        #Fall back to English-only OCR instead of failing the whole scan.
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 4 --oem 3")
    return text


def _find_amount(pattern: str, text: str):
    """Search for an amount pattern, return float or None."""
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        raw = m.group(1).replace(",", "").strip()
        try:
            return float(raw)
        except ValueError:
            return None
    return None


#--------------------------------------------------------------------------
# Field-level extraction helpers. Each one takes the already-split lines
# (or the joined full text) and returns just the piece of the result it is
# responsible for, so parse_text() only has to assemble the pieces.
#--------------------------------------------------------------------------

def _extract_store(lines: list) -> tuple:
    """Return (store_name, store_address)."""
    store_name = None
    for line in lines[:10]:
        if KNOWN_STORES.search(line):
            store_name = line.strip()
            break
    if not store_name:
        for line in lines[:8]:
            clean = line.strip()
            if (len(clean) >= 3
                    and not SKIP_HEADER.search(clean)
                    and not re.match(r"^[\d\(\+]", clean)
                    and not re.search(r"\d{5,}", clean)):
                store_name = clean
                break

    store_address = None
    if store_name and store_name in lines:
        idx = lines.index(store_name)
        addr_lines = []
        for line in lines[idx+1:idx+5]:
            if re.search(r"\d{4,5}|\bjalan\b|\blorong\b|road|street|ave|blvd|dr\b|st\b|floor|level|, [a-z]{2}", line, re.IGNORECASE):
                addr_lines.append(line)
            elif addr_lines:
                break
        if addr_lines:
            store_address = ", ".join(addr_lines)

    return store_name, store_address


def _extract_date(full: str):
    date_pats = [
        (r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", "ymd"),
        (r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", "dmy"),
        (r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b", "dmy2"),
        (r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+(\d{4})", "dmonthy"),
    ]
    for pat, fmt in date_pats:
        m = re.search(pat, full, re.IGNORECASE)
        if m:
            g = m.groups()
            try:
                if fmt == "ymd":
                    return f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
                elif fmt == "dmy":
                    return f"{g[2]}-{int(g[1]):02d}-{int(g[0]):02d}"
                elif fmt == "dmy2":
                    year = int(g[2]); year += 2000 if year < 50 else 1900
                    return f"{year}-{int(g[1]):02d}-{int(g[0]):02d}"
                elif fmt == "dmonthy":
                    mon = MONTHS_MY.get(g[1].lower()[:3], 1)
                    return f"{g[2]}-{mon:02d}-{int(g[0]):02d}"
            except Exception:
                continue
    return None


def _extract_time(full: str):
    time_m = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM|am|pm)?(?!\d)", full)
    if not time_m:
        return None
    h_val, m_val = int(time_m.group(1)), time_m.group(2)
    ampm = (time_m.group(3) or "").upper()
    if 0 <= h_val <= 23 and 0 <= int(m_val) <= 59:
        if ampm == "PM" and h_val < 12: h_val += 12
        elif ampm == "AM" and h_val == 12: h_val = 0
        elif re.search(r"下午", full) and h_val < 12: h_val += 12
        return f"{h_val:02d}:{m_val}"
    return None


def _extract_cashier(full: str):
    cm = re.search(r"(?:Cashier|Operator|Served\s*by|Pekerja|Staff|OP#?)[:\s]+([^\n\r,#]{2,30})", full, re.IGNORECASE)
    if cm:
        val = cm.group(1).strip()
        if not re.search(r"\d{4,}", val):
            return val
    return None


def _extract_receipt_no(full: str):
    rcpt_pats = [
        r"Rcpt#?[:\s]*([A-Z0-9][\w\-]{2,20})",
        r"(?:Receipt|Invoice|Resit)\s*(?:No\.?)?[:\s#]*([A-Z0-9][\w\-]{2,20})",
        r"TC#\s*([\d\s]{10,})",
        r"(?:Trans|Txn)[:\s#]*([A-Z0-9][\w\-]{3,20})",
    ]
    for pat in rcpt_pats:
        m = re.search(pat, full, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if not re.match(r"^\d{4}$", val):
                return val
    return None


def _extract_amounts(full: str) -> dict:
    """Return subtotal/discount/total/tax/tax_type/rounding/payment_method."""
    amounts = {
        "subtotal": None, "discount": None, "total": None,
        "tax": None, "tax_type": None, "payment_method": None,
    }

    AMT = r"\$?\s*([\d,]+(?:\.\d{1,2})?)"

    amounts["total"] = (
        _find_amount(rf"(?:Grand\s*Total|\*+\s*Total)[^\d$]*{AMT}", full) or
        _find_amount(rf"(?:^|\n)\s*(?:Total|TOTAL|Jumlah)\s*:?[^\d$]*{AMT}", full) or
        _find_amount(rf"(?:Amount\s*Due|Amount\s*Paid|Amaun)[^\d$]*{AMT}", full)
    )
    amounts["subtotal"] = _find_amount(
        rf"(?:Subtotal|Sub.?total|Net\s*Sales|Sub\s*Jumlah)[^\d$]*{AMT}", full
    )
    amounts["discount"] = _find_amount(
        rf"(?:Discount\s*Given|Discount|Disc|Diskaun|Rebate|You\s*Saved|Savings)[^\d$]*{AMT}", full
    )

    #Tax: collect ALL tax lines, sum them
    tax_lines = re.findall(
        r"(?:TAX\s*\d*|GST|SST|Service\s*Tax|Cukai)\s*(?:[\d.]+\s*%)?\s*\$?\s*([\d,]+\.\d{2})",
        full, re.IGNORECASE
    )
    if tax_lines:
        try:
            amounts["tax"] = round(sum(float(x.replace(",","")) for x in tax_lines), 2)
        except Exception:
            pass
        if re.search(r"GST", full, re.IGNORECASE): amounts["tax_type"] = "GST"
        elif re.search(r"SST", full, re.IGNORECASE): amounts["tax_type"] = "SST"
        else: amounts["tax_type"] = "TAX"

    #Rounding
    rounding = _find_amount(rf"(?:Rounding|Round)[^\d\-]*(-?[\d,]+\.\d{{2}})", full)
    if rounding is not None:
        amounts["rounding"] = rounding

    #Payment method
    pay_patterns = [
        (r"\bCash\b|\bTunai\b",                              "Cash"),
        (r"Visa",                                               "Visa"),
        (r"Mastercard|Master\s*Card",                          "Mastercard"),
        (r"(?:US\s*)?Debit|EFT\s*Debit",                      "Debit Card"),
        (r"Credit|Kad\s*Kredit",                               "Credit Card"),
        (r"TNG|Touch\s*['`]?n?\s*Go|eWallet|e-Wallet",       "TNG"),
        (r"GrabPay|Grab\s*Pay",                                "GrabPay"),
        (r"Boost",                                              "Boost"),
        (r"ShopeePay|Shopee\s*Pay",                            "ShopeePay"),
        (r"DuitNow|Duit\s*Now",                                "DuitNow"),
        (r"MAE|Maybank\s*QR",                                  "MAE"),
        (r"Online\s*Banking|FPX",                              "Online Banking"),
        (r"\bQR\b",                                            "QR"),
    ]
    for pat, label in pay_patterns:
        if re.search(pat, full, re.IGNORECASE):
            amounts["payment_method"] = label
            break

    return amounts


_ITEM_STD       = re.compile(r"^(.+?)\s{2,}(?:RM\s*|\$\s*)?(\d[\d,]*\.\d{2})(?:\s*[A-Z])?\s*$")
_ITEM_WMT       = re.compile(r"^([A-Z][A-Z0-9\s\#\.\&\-\'\*@]{1,28}?)\s{2,}\d{8,}\s*[A-Z]?\s+(\d[\d,]*\.\d{2})\s*[A-Z]?\s*$")
_ITEM_COS       = re.compile(r"^[A-Z]\s+(\d{6}\s+)(.+?)\s{2,}(\d[\d,]*\.\d{2})\s*[A-Z]?\s*$")
_ITEM_QTY_FIRST = re.compile(r"^(\d+)\s+(.+?)\s{2,}(\d[\d,]*)\s*$")
_QTY_LINE       = re.compile(r"^(\d+)\s*[@xX]\s*(\$?[\d,]+(?:\.\d{1,2})?)\s*$")
_INLINE_QTY     = re.compile(r"^(.+?)\s+(\d+)\s*[xX@]\s*(\$?[\d,]+\.\d{2})\s+(\$?[\d,]+\.\d{2})\s*$")
_ITEM_TJ        = re.compile(r"^([A-Z][A-Z\s\.\/\&\-\'\d]{3,35})\s+(\$?[\d]+\.\d{2})\s*$")


def _clean_price(s: str) -> float:
    return float(s.replace("$","").replace(",","").strip())


def _extract_items(lines: list) -> list:
    """Walk the OCR lines and pull out individual purchased items.

    A handful of receipt layouts are tried per line (inline qty/price,
    quantity-on-its-own-line, Walmart/Costco-style codes, a generic
    "name ... price" fallback, etc). A `pending` item is carried across
    lines so a name/price split across two OCR lines is reassembled.
    """
    items = []
    pending = None

    def flush(p):
        if p: items.append(p)
        return None

    for line in lines:
        if len(line) < 3:
            pending = flush(pending); continue
        if SKIP_ITEM.search(line):
            pending = flush(pending); continue

        m = _INLINE_QTY.match(line)
        if m:
            name = m.group(1).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending)
                try:
                    items.append({
                        "name": name, "quantity": int(m.group(2)),
                        "unit_price": _clean_price(m.group(3)),
                        "total_price": _clean_price(m.group(4)),
                    })
                except Exception: pass
                continue

        m = _QTY_LINE.match(line)
        if m and pending:
            try:
                pending["quantity"]   = int(m.group(1))
                pending["unit_price"] = _clean_price(m.group(2))
            except Exception: pass
            pending = flush(pending)
            continue

        m = _ITEM_COS.match(line)
        if m:
            name = m.group(2).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending)
                try:
                    items.append({
                        "name": name, "quantity": 1,
                        "unit_price": _clean_price(m.group(3)),
                        "total_price": _clean_price(m.group(3)),
                    })
                except Exception: pass
                continue

        m = _ITEM_WMT.match(line)
        if m:
            name = m.group(1).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending)
                try:
                    items.append({
                        "name": name, "quantity": 1,
                        "unit_price": _clean_price(m.group(2)),
                        "total_price": _clean_price(m.group(2)),
                    })
                except Exception: pass
                continue

        m = _ITEM_QTY_FIRST.match(line)
        if m:
            qty_val, name = int(m.group(1)), m.group(2).strip()
            if not SKIP_ITEM.search(name) and not SKIP_HEADER.search(name):
                pending = flush(pending)
                try:
                    total = _clean_price(m.group(3))
                    items.append({
                        "name": name, "quantity": qty_val,
                        "unit_price": round(total / qty_val, 2),
                        "total_price": total,
                    })
                except Exception: pass
                continue

        m = _ITEM_STD.match(line)
        if m:
            name = m.group(1).strip()
            if SKIP_ITEM.search(name) or SKIP_HEADER.search(name):
                pending = flush(pending); continue
            try:
                price = _clean_price(m.group(2))
                if price > 99999: continue
            except Exception: continue
            pending = flush(pending)
            pending = {"name": name, "quantity": 1, "unit_price": price, "total_price": price}
            continue

        m = _ITEM_TJ.match(line)
        if m:
            name = m.group(1).strip()
            if SKIP_ITEM.search(name) or SKIP_HEADER.search(name):
                pending = flush(pending); continue
            try:
                price = _clean_price(m.group(2))
                if price > 99999: continue
            except Exception: continue
            pending = flush(pending)
            pending = {"name": name, "quantity": 1, "unit_price": price, "total_price": price}
            continue

        #Continuation line
        if pending:
            if not re.search(r"\d{2}:\d{2}|\d{4}[/\-]|\b\d+\.\d{2}\b|\d{6,}", line):
                pending["name"] += " " + line
            else:
                pending = flush(pending)

    flush(pending)
    return items


def parse_text(raw_text: str, source_file: str = "") -> dict:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    full  = "\n".join(lines)

    store_name, store_address = _extract_store(lines)
    amounts = _extract_amounts(full)
    items = _extract_items(lines)

    #Remove items that are actually total/summary amounts
    if amounts["total"]:
        items = [i for i in items if i["total_price"] != amounts["total"]]

    result = {
        "store_name":     store_name,
        "store_address":  store_address,
        "date":           _extract_date(full),
        "time":           _extract_time(full),
        "cashier":        _extract_cashier(full),
        "receipt_no":     _extract_receipt_no(full),
        "items":          items,
        "subtotal":       amounts["subtotal"],
        "discount":       amounts["discount"],
        "tax":            amounts["tax"],
        "tax_type":       amounts["tax_type"],
        "total":          amounts["total"],
        "payment_method": amounts["payment_method"],
        "currency":       _detect_currency(full),
        "_source_file":   source_file,
        "_parsed_at":     datetime.now().isoformat(timespec="seconds"),
        "_raw_text":      raw_text,
    }

    if "rounding" in amounts:
        result["rounding"] = amounts["rounding"]

    return result


def scan_receipt(image_path: str) -> dict:
    return parse_text(extract_text(image_path), Path(image_path).name)
