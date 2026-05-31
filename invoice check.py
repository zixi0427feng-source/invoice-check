import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import csv
import re
import cv2
import pytesseract
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

COLORS = {
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "card":     "#0f3460",
    "accent":   "#e94560",
    "accent2":  "#f5a623",
    "text":     "#eaeaea",
    "text_dim": "#8892a4",
    "success":  "#4ecca3",
    "input_bg": "#0d2137",
    "btn_hover":"#c73652",
}

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
    r"give us|co\. no|business|\(\d{{3}}\)|\d{{3}}[-\s]\d{{3}}",
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

#Walmart-style barcode item line
WALMART_ITEM = re.compile(
    r"^([A-Z][A-Z0-9\s\#\.\\/\&\-\'\*]{1,30?}?)\s{{2,}}\d{{6,}}\s*[A-Z]?\s+\$?([\d,]+\.\d{{2}})\s*[A-Z]?\s*$"
)

MONTHS_MY = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "januari":1,"februari":2,"mac":3,"april":4,"mei":5,"jun":6,
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
    text = pytesseract.image_to_string(img, lang="eng+chi_sim", config="--psm 4 --oem 3")
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


def parse_text(raw_text: str, source_file: str = "") -> dict:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    full  = "\n".join(lines)

    result = {
        "store_name":     None,
        "store_address":  None,
        "date":           None,
        "time":           None,
        "cashier":        None,
        "receipt_no":     None,
        "items":          [],
        "subtotal":       None,
        "discount":       None,
        "tax":            None,
        "tax_type":       None,
        "total":          None,
        "payment_method": None,
        "currency":       _detect_currency(full),
        "_source_file":   source_file,
        "_parsed_at":     datetime.now().isoformat(timespec="seconds"),
        "_raw_text":      raw_text,
    }

#Store name
    for line in lines[:10]:
        if KNOWN_STORES.search(line):
            result["store_name"] = line.strip()
            break
    if not result["store_name"]:
        for line in lines[:8]:
            clean = line.strip()
            if (len(clean) >= 3
                    and not SKIP_HEADER.search(clean)
                    and not re.match(r"^[\d\(\+]", clean)
                    and not re.search(r"\d{5,}", clean)):
                result["store_name"] = clean
                break

#Store address
    if result["store_name"] and result["store_name"] in lines:
        idx = lines.index(result["store_name"])
        addr_lines = []
        for line in lines[idx+1:idx+5]:
            if re.search(r"\d{4,5}|\bjalan\b|\blorong\b|road|street|ave|blvd|dr\b|st\b|floor|level|, [a-z]{2}", line, re.IGNORECASE):
                addr_lines.append(line)
            elif addr_lines:
                break
        if addr_lines:
            result["store_address"] = ", ".join(addr_lines)

#Date
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
                    result["date"] = f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
                elif fmt == "dmy":
                    result["date"] = f"{g[2]}-{int(g[1]):02d}-{int(g[0]):02d}"
                elif fmt == "dmy2":
                    year = int(g[2]); year += 2000 if year < 50 else 1900
                    result["date"] = f"{year}-{int(g[1]):02d}-{int(g[0]):02d}"
                elif fmt == "dmonthy":
                    mon = MONTHS_MY.get(g[1].lower()[:3], 1)
                    result["date"] = f"{g[2]}-{mon:02d}-{int(g[0]):02d}"
                break
            except Exception:
                continue

#Time
    time_m = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM|am|pm)?(?!\d)", full)
    if time_m:
        h_val, m_val = int(time_m.group(1)), time_m.group(2)
        ampm = (time_m.group(3) or "").upper()
        if 0 <= h_val <= 23 and 0 <= int(m_val) <= 59:
            if ampm == "PM" and h_val < 12: h_val += 12
            elif ampm == "AM" and h_val == 12: h_val = 0
            elif re.search(r"下午", full) and h_val < 12: h_val += 12
            result["time"] = f"{h_val:02d}:{m_val}"

#Cashier
    cm = re.search(r"(?:Cashier|Operator|Served\s*by|Pekerja|Staff|OP#?)[:\s]+([^\n\r,#]{2,30})", full, re.IGNORECASE)
    if cm:
        val = cm.group(1).strip()
        if not re.search(r"\d{4,}", val):
            result["cashier"] = val

#Receipt number
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
                result["receipt_no"] = val
                break

#Amounts
    AMT = r"\$?\s*([\d,]+(?:\.\d{1,2})?)"

    result["total"] = (
        _find_amount(rf"(?:Grand\s*Total|\*+\s*Total)[^\d$]*{AMT}", full) or
        _find_amount(rf"(?:^|\n)\s*(?:Total|TOTAL|Jumlah)\s*:?[^\d$]*{AMT}", full) or
        _find_amount(rf"(?:Amount\s*Due|Amount\s*Paid|Amaun)[^\d$]*{AMT}", full)
    )
    result["subtotal"] = _find_amount(
        rf"(?:Subtotal|Sub.?total|Net\s*Sales|Sub\s*Jumlah)[^\d$]*{AMT}", full
    )
    result["discount"] = _find_amount(
        rf"(?:Discount\s*Given|Discount|Disc|Diskaun|Rebate|You\s*Saved|Savings)[^\d$]*{AMT}", full
    )

#Tax （ collect ALL tax lines, sum them ）
    tax_lines = re.findall(
        r"(?:TAX\s*\d*|GST|SST|Service\s*Tax|Cukai)\s*(?:[\d.]+\s*%)?\s*\$?\s*([\d,]+\.\d{2})",
        full, re.IGNORECASE
    )
    if tax_lines:
        try:
            result["tax"] = round(sum(float(x.replace(",","")) for x in tax_lines), 2)
        except Exception:
            pass
    #Label
        if re.search(r"GST", full, re.IGNORECASE): result["tax_type"] = "GST"
        elif re.search(r"SST", full, re.IGNORECASE): result["tax_type"] = "SST"
        else: result["tax_type"] = "TAX"

#Rounding
    rounding = _find_amount(rf"(?:Rounding|Round)[^\d\-]*(-?[\d,]+\.\d{{2}})", full)
    if rounding is not None:
        result["rounding"] = rounding

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
            result["payment_method"] = label
            break

#Items
    item_std = re.compile(r"^(.+?)\s{2,}(?:RM\s*|\$\s*)?(\d[\d,]*\.\d{2})(?:\s*[A-Z])?\s*$")
    item_wmt = re.compile(r"^([A-Z][A-Z0-9\s\#\.\&\-\'\*@]{1,28}?)\s{2,}\d{8,}\s*[A-Z]?\s+(\d[\d,]*\.\d{2})\s*[A-Z]?\s*$")
    item_cos = re.compile(r"^[A-Z]\s+(\d{6}\s+)(.+?)\s{2,}(\d[\d,]*\.\d{2})\s*[A-Z]?\s*$")
    item_qty_first = re.compile(r"^(\d+)\s+(.+?)\s{2,}(\d[\d,]*)\s*$")
    qty_line = re.compile(r"^(\d+)\s*[@xX]\s*(\$?[\d,]+(?:\.\d{1,2})?)\s*$")
    inline_qty = re.compile(r"^(.+?)\s+(\d+)\s*[xX@]\s*(\$?[\d,]+\.\d{2})\s+(\$?[\d,]+\.\d{2})\s*$")
    item_tj = re.compile(r"^([A-Z][A-Z\s\.\/\&\-\'\d]{3,35})\s+(\$?[\d]+\.\d{2})\s*$")

    def clean_price(s):
        return float(s.replace("$","").replace(",","").strip())

    pending = None

    def flush(p, items):
        if p: items.append(p)
        return None

    for line in lines:
        if len(line) < 3:
            pending = flush(pending, result["items"]); continue
        if SKIP_ITEM.search(line):
            pending = flush(pending, result["items"]); continue

        m = inline_qty.match(line)
        if m:
            name = m.group(1).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending, result["items"])
                try:
                    result["items"].append({
                        "name": name, "quantity": int(m.group(2)),
                        "unit_price": clean_price(m.group(3)),
                        "total_price": clean_price(m.group(4)),
                    })
                except Exception: pass
                continue

        m = qty_line.match(line)
        if m and pending:
            try:
                pending["quantity"]   = int(m.group(1))
                pending["unit_price"] = clean_price(m.group(2))
            except Exception: pass
            pending = flush(pending, result["items"])
            continue

        m = item_cos.match(line)
        if m:
            name = m.group(2).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending, result["items"])
                try:
                    result["items"].append({
                        "name": name, "quantity": 1,
                        "unit_price": clean_price(m.group(3)),
                        "total_price": clean_price(m.group(3)),
                    })
                except Exception: pass
                continue

        m = item_wmt.match(line)
        if m:
            name = m.group(1).strip()
            if not SKIP_ITEM.search(name):
                pending = flush(pending, result["items"])
                try:
                    result["items"].append({
                        "name": name, "quantity": 1,
                        "unit_price": clean_price(m.group(2)),
                        "total_price": clean_price(m.group(2)),
                    })
                except Exception: pass
                continue

        m = item_qty_first.match(line)
        if m:
            qty_val, name = int(m.group(1)), m.group(2).strip()
            if not SKIP_ITEM.search(name) and not SKIP_HEADER.search(name):
                pending = flush(pending, result["items"])
                try:
                    total = clean_price(m.group(3))
                    result["items"].append({
                        "name": name, "quantity": qty_val,
                        "unit_price": round(total / qty_val, 2),
                        "total_price": total,
                    })
                except Exception: pass
                continue

        m = item_std.match(line)
        if m:
            name = m.group(1).strip()
            if SKIP_ITEM.search(name) or SKIP_HEADER.search(name):
                pending = flush(pending, result["items"]); continue
            try:
                price = clean_price(m.group(2))
                if price > 99999: continue
            except Exception: continue
            pending = flush(pending, result["items"])
            pending = {"name": name, "quantity": 1, "unit_price": price, "total_price": price}
            continue

        m = item_tj.match(line)
        if m:
            name = m.group(1).strip()
            if SKIP_ITEM.search(name) or SKIP_HEADER.search(name):
                pending = flush(pending, result["items"]); continue
            try:
                price = clean_price(m.group(2))
                if price > 99999: continue
            except Exception: continue
            pending = flush(pending, result["items"])
            pending = {"name": name, "quantity": 1, "unit_price": price, "total_price": price}
            continue

    #Continuation line
        if pending:
            if not re.search(r"\d{2}:\d{2}|\d{4}[/\-]|\b\d+\.\d{2}\b|\d{6,}", line):
                pending["name"] += " " + line
            else:
                pending = flush(pending, result["items"])

    pending = flush(pending, result["items"])

#Remove items that are actually total/summary amounts
    if result["total"]:
        result["items"] = [i for i in result["items"] if i["total_price"] != result["total"]]

    return result

def scan_receipt(image_path: str) -> dict:
    return parse_text(extract_text(image_path), Path(image_path).name)


#GUI

C = {
    "bg":        "#0e0e12",
    "sidebar":   "#13131a",
    "card":      "#1a1a24",
    "card2":     "#1f1f2e",
    "border":    "#2a2a3d",
    "accent":    "#7c6af7",      # soft violet
    "accent2":   "#f0a500",      # warm amber
    "green":     "#3ecf8e",      # mint green
    "red":       "#f45b5b",
    "text":      "#e8e8f0",
    "dim":       "#6b6b8a",
    "input":     "#111118",
    "hover":     "#8f7ffb",
    "scan_from": "#7c6af7",
    "scan_to":   "#c084fc",
}

FONT_TITLE  = ("Segoe UI", 15, "bold")
FONT_LABEL  = ("Segoe UI", 8)
FONT_BODY   = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_SCAN   = ("Segoe UI", 11, "bold")
FONT_HEADER = ("Segoe UI", 10, "bold")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def _lerp_color(c1, c2, t):
    r1,g1,b1 = _hex_to_rgb(c1)
    r2,g2,b2 = _hex_to_rgb(c2)
    return f"#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}"


class ReceiptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Receipt Scanner")
        self.root.geometry("1200x760")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        self.camera         = None
        self.camera_running = False
        self.camera_frame   = None
        self.current_path   = None
        self.batch_paths    = None
        self.records        = []
        self._scan_anim_step = 0
        self._scan_anim_id   = None

        self._styles()
        self._build_ui()
        self._check_tesseract()

#Styles

    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")

    #Notebook
        s.configure("App.TNotebook", background=C["card"], borderwidth=0, tabmargins=[0,0,0,0])
        s.configure("App.TNotebook.Tab",
                    background=C["card2"], foreground=C["dim"],
                    padding=[14, 7], font=FONT_BODY,
                    borderwidth=0, focuscolor=C["card"])
        s.map("App.TNotebook.Tab",
              background=[("selected", C["card"]), ("active", C["card"])],
              foreground=[("selected", C["accent"]), ("active", C["text"])])

        s.configure("Side.TNotebook", background=C["sidebar"], borderwidth=0, tabmargins=[0,0,0,0])
        s.configure("Side.TNotebook.Tab",
                    background=C["sidebar"], foreground=C["dim"],
                    padding=[10, 6], font=FONT_BODY, borderwidth=0)
        s.map("Side.TNotebook.Tab",
              background=[("selected", C["card"]), ("active", C["card2"])],
              foreground=[("selected", C["accent"]), ("active", C["text"])])

    #Treeview
        s.configure("App.Treeview",
                    background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], rowheight=30,
                    font=FONT_BODY, borderwidth=0)
        s.configure("App.Treeview.Heading",
                    background=C["card2"], foreground=C["dim"],
                    font=("Segoe UI", 8, "bold"), relief="flat",
                    borderwidth=0, padding=[8,6])
        s.map("App.Treeview",
              background=[("selected", C["accent"])],
              foreground=[("selected", "#ffffff")])

    #Scrollbar
        s.configure("Thin.Vertical.TScrollbar",
                    background=C["card2"], troughcolor=C["card"],
                    borderwidth=0, arrowsize=0, width=6)
        s.configure("Thin.Horizontal.TScrollbar",
                    background=C["card2"], troughcolor=C["card"],
                    borderwidth=0, arrowsize=0, width=6)

    #Progressbar
        s.configure("Scan.Horizontal.TProgressbar",
                    background=C["accent"], troughcolor=C["card2"],
                    borderwidth=0, thickness=3)

        s.configure("TFrame", background=C["bg"])
        s.configure("Card.TFrame", background=C["card"])

#UI

    def _build_ui(self):
    #Titlebar
        bar = tk.Frame(self.root, bg=C["sidebar"], height=52)
        bar.pack(fill="x"); bar.pack_propagate(False)

    #Logo dot + title
        dot_frame = tk.Frame(bar, bg=C["sidebar"])
        dot_frame.pack(side="left", padx=(18,0), pady=14)
        tk.Canvas(dot_frame, width=10, height=10, bg=C["sidebar"],
                  highlightthickness=0).pack(side="left")
        self._draw_dot(dot_frame.winfo_children()[0])

        tk.Label(bar, text="Receipt Scanner", font=FONT_TITLE,
                 bg=C["sidebar"], fg=C["text"]).pack(side="left", padx=(10,0))
        tk.Label(bar, text="Tesseract OCR", font=("Segoe UI", 8),
                 bg=C["sidebar"], fg=C["dim"]).pack(side="left", padx=(8,0), pady=(4,0))

    #Separator line
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x")

    #Main layout
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

    #Left sidebar
        sidebar = tk.Frame(body, bg=C["sidebar"], width=300)
        sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

    #Vertical divider
        tk.Frame(body, bg=C["border"], width=1).pack(side="left", fill="y")

    #Right content
        content = tk.Frame(body, bg=C["bg"])
        content.pack(side="left", fill="both", expand=True)
        self._build_content(content)

    def _draw_dot(self, canvas):
        canvas.configure(width=10, height=10)
        canvas.create_oval(1, 1, 9, 9, fill=C["accent"], outline="")

#Sidebar

    def _build_sidebar(self, p):
        # Tesseract path
        sec = self._section(p, "TESSERACT PATH")
        self.tess_path = tk.StringVar(value=pytesseract.pytesseract.tesseract_cmd)
        path_frame = tk.Frame(sec, bg=C["input"], bd=0)
        path_frame.pack(fill="x", pady=(4,0))
        tk.Entry(path_frame, textvariable=self.tess_path,
                 font=("Consolas", 8), bg=C["input"], fg=C["accent2"],
                 insertbackground=C["accent2"], relief="flat",
                 bd=6, highlightthickness=0).pack(fill="x")

        tk.Frame(p, bg=C["border"], height=1).pack(fill="x", pady=10)

    #Image preview
        prev_label = tk.Label(p, text="PREVIEW", font=FONT_LABEL,
                              bg=C["sidebar"], fg=C["dim"])
        prev_label.pack(anchor="w", padx=16, pady=(0,6))

        prev_wrap = tk.Frame(p, bg=C["card"], padx=1, pady=1)
        prev_wrap.pack(fill="x", padx=12, pady=(0,10))
        self.preview = tk.Label(prev_wrap,
                                text="No image selected\n\nCapture or browse",
                                font=("Segoe UI", 9), bg=C["input"],
                                fg=C["dim"], height=13, anchor="center",
                                justify="center")
        self.preview.pack(fill="both", expand=True)

        tk.Frame(p, bg=C["border"], height=1).pack(fill="x", pady=(0,10))

    #Input tabs
        nb = ttk.Notebook(p, style="Side.TNotebook")
        nb.pack(fill="x", padx=12, pady=(0,10))

    #Camera tab
        ct = tk.Frame(nb, bg=C["card"]); nb.add(ct, text=" 📷  Camera ")
        ci = tk.Frame(ct, bg=C["card"], pady=10, padx=10); ci.pack(fill="x")
        self.cam_btn = self._pill_btn(ci, "▶  Start Camera", self._toggle_camera, C["card2"])
        self.cam_btn.pack(fill="x", pady=(0,6))
        self._pill_btn(ci, "⊙  Capture Photo", self._capture, C["accent"]).pack(fill="x")

    #File tab
        ft = tk.Frame(nb, bg=C["card"]); nb.add(ft, text=" 🖼  File ")
        fi = tk.Frame(ft, bg=C["card"], pady=10, padx=10); fi.pack(fill="x")
        self._pill_btn(fi, "⊕  Browse Image",    self._browse,          C["card2"]).pack(fill="x", pady=(0,6))
        self._pill_btn(fi, "⊕  Select Multiple", self._browse_multiple, C["card2"]).pack(fill="x")

    #Scan button
        scan_wrap = tk.Frame(p, bg=C["sidebar"]); scan_wrap.pack(fill="x", padx=12, pady=(0,10))
        self.scan_btn = tk.Button(scan_wrap,
                                  text="  Scan Receipt",
                                  font=FONT_SCAN,
                                  bg=C["accent"], fg="#ffffff",
                                  activebackground=C["hover"],
                                  activeforeground="#ffffff",
                                  relief="flat", bd=0,
                                  pady=12, cursor="hand2",
                                  command=self._start_scan)
        self.scan_btn.pack(fill="x")
        self._btn_hover_effect(self.scan_btn, C["accent"], C["hover"])

    #Progress bar
        self.progress = ttk.Progressbar(p, mode="indeterminate",
                                        style="Scan.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=12, pady=(0,4))

    #Status
        status_frame = tk.Frame(p, bg=C["card"], padx=12, pady=10)
        status_frame.pack(fill="x", padx=12, pady=(4,0))
        tk.Label(status_frame, text="STATUS", font=FONT_LABEL,
                 bg=C["card"], fg=C["dim"]).pack(anchor="w")
        self.status_var = tk.StringVar(value="Ready — Tesseract not checked yet")
        self.status_lbl = tk.Label(status_frame, textvariable=self.status_var,
                                   font=("Segoe UI", 8), bg=C["card"],
                                   fg=C["green"], anchor="w",
                                   wraplength=260, justify="left")
        self.status_lbl.pack(fill="x", pady=(4,0))

#Content area

    def _build_content(self, p):
    #Toolbar
        tb = tk.Frame(p, bg=C["sidebar"], height=46)
        tb.pack(fill="x"); tb.pack_propagate(False)

        tk.Label(tb, text="Results", font=FONT_HEADER,
                 bg=C["sidebar"], fg=C["text"]).pack(side="left", padx=18, pady=12)

        self.count_badge = tk.Label(tb, text="0", font=("Segoe UI", 8, "bold"),
                                    bg=C["accent"], fg="#fff",
                                    padx=8, pady=2)
        self.count_badge.pack(side="left", pady=16)

    #Right toolbar buttons
        btn_frame = tk.Frame(tb, bg=C["sidebar"])
        btn_frame.pack(side="right", padx=12, pady=8)
        self._tool_btn(btn_frame, "↓  Export CSV", self._export_csv, C["green"]).pack(side="right", padx=(6,0))
        self._tool_btn(btn_frame, "⊘  Clear",      self._clear,      C["card2"]).pack(side="right")

        tk.Frame(p, bg=C["border"], height=1).pack(fill="x")

    #Tabs
        nb = ttk.Notebook(p, style="App.TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

    #Summary
        t1 = tk.Frame(nb, bg=C["card"]); nb.add(t1, text="  Summary  ")
        self._build_summary_tab(t1)

    #Items
        t2 = tk.Frame(nb, bg=C["card"]); nb.add(t2, text="  Items  ")
        self._build_items_tab(t2)

    #JSON
        t3 = tk.Frame(nb, bg=C["card"]); nb.add(t3, text="  JSON  ")
        self._build_json_tab(t3)

    #Raw OCR
        t4 = tk.Frame(nb, bg=C["card"]); nb.add(t4, text="  Raw OCR  ")
        self._build_raw_tab(t4)

    #Analytics
        t5 = tk.Frame(nb, bg=C["card"]); nb.add(t5, text="  Analytics  ")
        self._build_analytics_tab(t5)


    def _build_analytics_tab(self, p):
        """Daily spending line chart."""
    #Top controls bar
        ctrl = tk.Frame(p, bg=C["card2"], pady=8, padx=16)
        ctrl.pack(fill="x")
        tk.Label(ctrl, text="Daily Spending", font=FONT_HEADER,
                 bg=C["card2"], fg=C["text"]).pack(side="left")
        self._tool_btn(ctrl, "↻  Refresh", self._refresh_chart, C["card"]).pack(side="right")

    #Summary stats row
        stats_row = tk.Frame(p, bg=C["card"], pady=10, padx=16)
        stats_row.pack(fill="x")
        self._stat_cards = {}
        for key, label in [("total_spent","Total Spent"),("avg_day","Avg / Day"),
                            ("max_day","Highest Day"),("receipts","Receipts")]:
            card = tk.Frame(stats_row, bg=C["card2"], padx=14, pady=8)
            card.pack(side="left", padx=(0,10))
            tk.Label(card, text=label, font=("Segoe UI",7), bg=C["card2"], fg=C["dim"]).pack(anchor="w")
            val_lbl = tk.Label(card, text="—", font=("Segoe UI",13,"bold"),
                               bg=C["card2"], fg=C["accent"])
            val_lbl.pack(anchor="w")
            self._stat_cards[key] = val_lbl

        tk.Frame(p, bg=C["border"], height=1).pack(fill="x")

    #Canvas for chart
        self.chart_canvas = tk.Canvas(p, bg=C["card"], highlightthickness=0,
                                       bd=0, cursor="crosshair")
        self.chart_canvas.pack(fill="both", expand=True, padx=16, pady=16)
        self.chart_canvas.bind("<Configure>", lambda e: self._refresh_chart())
        self.chart_canvas.bind("<Motion>", self._chart_hover)
        self._chart_data   = []   # list of (date_str, amount)
        self._chart_points = []   # list of (cx, cy, date_str, amount) for hover

    #Tooltip label
        self._tooltip = tk.Label(p, text="", font=("Segoe UI",8),
                                  bg=C["accent2"], fg=C["bg"],
                                  padx=8, pady=4, relief="flat")

    def _refresh_chart(self):
        """Aggregate records by date and redraw the line chart."""
        from collections import defaultdict

    #Aggregate
        daily = defaultdict(float)
        for r in self.records:
            date = r.get("date")
            total = r.get("total")
            if date and total:
                try: daily[date] += float(total)
                except (ValueError, TypeError): pass

        if not daily:
            self.chart_canvas.delete("all")
            w = self.chart_canvas.winfo_width() or 600
            h = self.chart_canvas.winfo_height() or 300
            self.chart_canvas.create_text(w//2, h//2,
                text="No data yet — scan some receipts first",
                fill=C["dim"], font=("Segoe UI", 10))
            for lbl in self._stat_cards.values(): lbl.configure(text="—")
            return

        sorted_dates = sorted(daily.keys())
        amounts      = [daily[d] for d in sorted_dates]
        self._chart_data = list(zip(sorted_dates, amounts))

    #Update stat cards
        total_spent = sum(amounts)
        avg_day     = total_spent / len(amounts)
        max_day     = max(amounts)
        receipts    = len(self.records)
        currency    = self.records[0].get("currency","MYR") if self.records else "MYR"
        self._stat_cards["total_spent"].configure(text=f"{currency} {total_spent:,.2f}")
        self._stat_cards["avg_day"].configure(text=f"{currency} {avg_day:,.2f}")
        self._stat_cards["max_day"].configure(text=f"{currency} {max_day:,.2f}")
        self._stat_cards["receipts"].configure(text=str(receipts))

        self._draw_chart(sorted_dates, amounts)

    def _draw_chart(self, dates, amounts):
        cv = self.chart_canvas
        cv.delete("all")
        W = cv.winfo_width()
        H = cv.winfo_height()
        if W < 50 or H < 50: return

        PAD_L, PAD_R, PAD_T, PAD_B = 64, 24, 24, 52
        n   = len(dates)
        max_val = max(amounts) if amounts else 1
        min_val = 0

        def cx(i):
            if n == 1: return PAD_L + (W - PAD_L - PAD_R) // 2
            return PAD_L + int(i / (n-1) * (W - PAD_L - PAD_R))

        def cy(v):
            ratio = (v - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            return PAD_T + int((1 - ratio) * (H - PAD_T - PAD_B))

    #Grid lines & Y labels
        y_steps = 5
        for i in range(y_steps + 1):
            v    = min_val + (max_val - min_val) * i / y_steps
            y    = cy(v)
            cv.create_line(PAD_L, y, W - PAD_R, y,
                           fill=C["border"], dash=(4,4), width=1)
            cv.create_text(PAD_L - 6, y, text=f"{v:,.0f}",
                           anchor="e", fill=C["dim"], font=("Segoe UI",7))

    # X labels
        step = max(1, n // 12)
        for i, d in enumerate(dates):
            if i % step == 0 or i == n-1:
                label = d[5:] if len(d) == 10 else d   # strip year → MM-DD
                cv.create_text(cx(i), H - PAD_B + 10,
                               text=label, fill=C["dim"],
                               font=("Segoe UI",7), angle=30 if n > 8 else 0)

    #Gradient fill under line
        pts = [(cx(i), cy(v)) for i, v in enumerate(amounts)]
        baseline = H - PAD_B
        for xi in range(PAD_L, W - PAD_R):
            seg = 0
            for k in range(len(pts)-1):
                if pts[k][0] <= xi <= pts[k+1][0]:
                    seg = k; break
            if len(pts) > 1 and pts[seg][0] != pts[seg+1][0]:
                t  = (xi - pts[seg][0]) / (pts[seg+1][0] - pts[seg][0])
                yi = int(pts[seg][1] + t * (pts[seg+1][1] - pts[seg][1]))
            else:
                yi = pts[seg][1] if pts else baseline
            alpha = int(40 + 60 * (baseline - yi) / max(baseline - PAD_T, 1))
            shade = f"#{max(0,min(255, alpha)):02x}{max(0,min(255,alpha//2)):02x}{max(0,min(255,180)):02x}"
            cv.create_line(xi, yi, xi, baseline, fill=shade, width=1)

    #Line
        if len(pts) > 1:
            flat = [coord for pt in pts for coord in pt]
            cv.create_line(*flat, fill=C["accent"], width=2, smooth=True)

    #Dots & store hover data
        self._chart_points = []
        for i, (x, y) in enumerate(pts):
            cv.create_oval(x-5, y-5, x+5, y+5,
                           fill=C["card"], outline=C["accent"], width=2)
            self._chart_points.append((x, y, dates[i], amounts[i]))

    #Axes
        cv.create_line(PAD_L, PAD_T, PAD_L, H-PAD_B, fill=C["border"], width=1)
        cv.create_line(PAD_L, H-PAD_B, W-PAD_R, H-PAD_B, fill=C["border"], width=1)

    def _chart_hover(self, event):
        """Show tooltip near nearest data point."""
        if not self._chart_points: return
        mx, my = event.x, event.y
    #Find nearest point within 30px
        nearest = None
        best    = 30
        for (px, py, date, amt) in self._chart_points:
            dist = ((mx-px)**2 + (my-py)**2) ** 0.5
            if dist < best:
                best = dist; nearest = (px, py, date, amt)
        if nearest:
            px, py, date, amt = nearest
            currency = self.records[0].get("currency","MYR") if self.records else "MYR"
            self._tooltip.configure(text=f"{date}   {currency} {amt:,.2f}")
            # Position tooltip near the dot, stay inside window
            tx = px + 12
            ty = py - 28
            self._tooltip.place(in_=self.chart_canvas, x=tx, y=ty)
            self._tooltip.lift()
        else:
            self._tooltip.place_forget()

    def _build_summary_tab(self, p):
        cols = ("File", "Store", "Date", "Time", "Receipt No", "Total", "Currency", "Payment")
        widths = [130, 160, 90, 70, 120, 75, 70, 110]
        f = tk.Frame(p, bg=C["card"]); f.pack(fill="both", expand=True, padx=1, pady=1)
        self.tree = ttk.Treeview(f, columns=cols, show="headings",
                                  selectmode="browse", style="App.Treeview")
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col, anchor="w")
            self.tree.column(col, width=w, anchor="w", minwidth=50)
        vsb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview, style="Thin.Vertical.TScrollbar")
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self.tree.xview, style="Thin.Horizontal.TScrollbar")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        f.grid_rowconfigure(0, weight=1); f.grid_columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        # Alternating row colors
        self.tree.tag_configure("odd",  background=C["card"])
        self.tree.tag_configure("even", background=C["card2"])

    def _build_items_tab(self, p):
        cols2 = ("Receipt File", "Item Name", "Qty", "Unit Price", "Total")
        widths2 = [140, 280, 55, 90, 90]
        f = tk.Frame(p, bg=C["card"]); f.pack(fill="both", expand=True, padx=1, pady=1)
        self.items_tree = ttk.Treeview(f, columns=cols2, show="headings", style="App.Treeview")
        for col, w in zip(cols2, widths2):
            self.items_tree.heading(col, text=col, anchor="w")
            self.items_tree.column(col, width=w, anchor="w", minwidth=40)
        vsb2 = ttk.Scrollbar(f, orient="vertical", command=self.items_tree.yview, style="Thin.Vertical.TScrollbar")
        self.items_tree.configure(yscrollcommand=vsb2.set)
        self.items_tree.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")
        f.grid_rowconfigure(0, weight=1); f.grid_columnconfigure(0, weight=1)
        self.items_tree.tag_configure("odd",  background=C["card"])
        self.items_tree.tag_configure("even", background=C["card2"])

    def _build_json_tab(self, p):
        f = tk.Frame(p, bg=C["card"]); f.pack(fill="both", expand=True, padx=1, pady=1)
        self.json_text = tk.Text(f, font=FONT_MONO,
                                  bg=C["input"], fg="#a0d8a0",
                                  insertbackground=C["green"],
                                  relief="flat", bd=0,
                                  wrap="none", padx=12, pady=10,
                                  selectbackground=C["accent"],
                                  highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=self.json_text.yview, style="Thin.Vertical.TScrollbar")
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self.json_text.xview, style="Thin.Horizontal.TScrollbar")
        self.json_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.json_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        f.grid_rowconfigure(0, weight=1); f.grid_columnconfigure(0, weight=1)

    def _build_raw_tab(self, p):
        f = tk.Frame(p, bg=C["card"]); f.pack(fill="both", expand=True, padx=1, pady=1)
        self.raw_text = tk.Text(f, font=FONT_MONO,
                                 bg=C["input"], fg=C["accent2"],
                                 insertbackground=C["accent2"],
                                 relief="flat", bd=0,
                                 wrap="word", padx=12, pady=10,
                                 selectbackground=C["accent"],
                                 highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=self.raw_text.yview, style="Thin.Vertical.TScrollbar")
        self.raw_text.configure(yscrollcommand=vsb.set)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        f.grid_rowconfigure(0, weight=1); f.grid_columnconfigure(0, weight=1)

#Widget helpers

    def _section(self, parent, label):
        """Labelled section block."""
        f = tk.Frame(parent, bg=C["sidebar"])
        f.pack(fill="x", padx=16, pady=(12,0))
        tk.Label(f, text=label, font=FONT_LABEL,
                 bg=C["sidebar"], fg=C["dim"]).pack(anchor="w")
        return f

    def _pill_btn(self, parent, text, cmd, color):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=FONT_BODY,
                        bg=color, fg=C["text"],
                        activebackground=C["border"],
                        activeforeground=C["text"],
                        relief="flat", bd=0,
                        pady=7, cursor="hand2",
                        highlightthickness=0)
        self._btn_hover_effect(btn, color, C["border"])
        return btn

    def _tool_btn(self, parent, text, cmd, color):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Segoe UI", 8, "bold"),
                        bg=color, fg=C["text"],
                        activebackground=C["hover"] if color == C["accent"] else C["border"],
                        activeforeground=C["text"],
                        relief="flat", bd=0,
                        padx=12, pady=5, cursor="hand2",
                        highlightthickness=0)
        return btn

    def _btn_hover_effect(self, btn, normal, hover):
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=normal))

#Tesseract check

    def _check_tesseract(self):
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tess_path.get()
            v = pytesseract.get_tesseract_version()
            self._set_status(f"✓ Tesseract {v} ready")
        except Exception:
            self._set_status("⚠ Tesseract not found. Update the path above.", "warning")

#Camera

    def _toggle_camera(self):
        if self.camera_running: self._stop_camera()
        else: self._start_camera()

    def _start_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot open camera."); return
        self.camera_running = True
        self.cam_btn.configure(text="■  Stop Camera", bg=C["red"])
        self._set_status("Camera active")
        self._cam_loop()

    def _stop_camera(self):
        self.camera_running = False
        if self.camera: self.camera.release(); self.camera = None
        self.cam_btn.configure(text="▶  Start Camera", bg=C["card2"])

    def _cam_loop(self):
        if not self.camera_running: return
        ret, frame = self.camera.read()
        if ret:
            self.camera_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb); img.thumbnail((276, 210))
            photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=photo, text=""); self.preview.image = photo
        self.root.after(33, self._cam_loop)

    def _capture(self):
        if not self.camera_running or self.camera_frame is None:
            messagebox.showwarning("Notice", "Please start the camera first"); return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(Path.home() / f"receipt_{ts}.jpg")
        cv2.imwrite(path, self.camera_frame)
        self.current_path = path
        self._set_status(f"Captured → {Path(path).name}")
        self._stop_camera()

#File browse

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Receipt Image",
            filetypes=[("Image files","*.jpg *.jpeg *.png *.webp *.gif"),("All files","*.*")])
        if path:
            self.current_path = path; self.batch_paths = None
            self._show_preview(path)
            self._set_status(f"Selected  {Path(path).name}")

    def _browse_multiple(self):
        paths = filedialog.askopenfilenames(
            title="Select Multiple Receipt Images",
            filetypes=[("Image files","*.jpg *.jpeg *.png *.webp *.gif"),("All files","*.*")])
        if paths:
            self.batch_paths  = list(paths)
            self.current_path = paths[0]
            self._show_preview(paths[0])
            self._set_status(f"Selected {len(paths)} images")

    def _show_preview(self, path):
        try:
            img = Image.open(path); img.thumbnail((276, 210))
            photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=photo, text=""); self.preview.image = photo
        except Exception as e:
            self.preview.configure(text=f"Preview error: {e}")

#Scan

    def _start_scan(self):
        pytesseract.pytesseract.tesseract_cmd = self.tess_path.get()
        targets = self.batch_paths or ([self.current_path] if self.current_path else None)
        if not targets:
            messagebox.showwarning("Notice", "Please select an image or capture a photo first"); return
        self.batch_paths = None
        self.scan_btn.configure(state="disabled", text="  Scanning…")
        self.progress.start(10)
        threading.Thread(target=self._worker, args=(targets,), daemon=True).start()

    def _worker(self, paths):
        for path in paths:
            self._set_status(f"Scanning  {Path(path).name} …")
            try:
                data = scan_receipt(path)
                self.records.append(data)
                self.root.after(0, self._update_tables, data)
                self.root.after(0, self._refresh_chart)
                self._set_status(
                    f"✓ {data.get('store_name') or Path(path).stem}"
                    f"   {data.get('currency','MYR')} {data.get('total','—')}"
                )
            except Exception as e:
                self._set_status(f"✗ {Path(path).name}: {e}", "error")
        self.root.after(0, self._done)

    def _done(self):
        self.scan_btn.configure(state="normal", text="  Scan Receipt")
        self.progress.stop()
        n = len(self.records)
        self.count_badge.configure(text=str(n))

#Update tables

    def _update_tables(self, data: dict):
        tag = "even" if len(self.tree.get_children()) % 2 == 0 else "odd"
        self.tree.insert("", "end", tags=(tag,), values=(
            data.get("_source_file",""), data.get("store_name",""),
            data.get("date",""),        data.get("time",""),
            data.get("receipt_no",""),  data.get("total",""),
            data.get("currency","MYR"), data.get("payment_method",""),
        ))
        for item in (data.get("items") or []):
            tag2 = "even" if len(self.items_tree.get_children()) % 2 == 0 else "odd"
            self.items_tree.insert("", "end", tags=(tag2,), values=(
                data.get("_source_file",""), item.get("name",""),
                item.get("quantity",""),     item.get("unit_price",""),
                item.get("total_price",""),
            ))
        d = {k: v for k, v in data.items() if k != "_raw_text"}
        self.json_text.insert("end", json.dumps(d, ensure_ascii=False, indent=2) + "\n\n")
        self.json_text.see("end")
        self.raw_text.insert("end", f"── {data.get('_source_file','')} ──\n{data.get('_raw_text','')}\n\n")
        self.raw_text.see("end")

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        idx = self.tree.index(sel[0])
        if idx < len(self.records):
            d = {k: v for k, v in self.records[idx].items() if k != "_raw_text"}
            self.json_text.delete("1.0","end")
            self.json_text.insert("end", json.dumps(d, ensure_ascii=False, indent=2))
            self.raw_text.delete("1.0","end")
            self.raw_text.insert("end", self.records[idx].get("_raw_text",""))

#Export / Clear

    def _export_csv(self):
        if not self.records:
            messagebox.showinfo("Notice","No results to export yet"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files","*.csv")],
            initialfile=f"receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not path: return
        rows = []
        for r in self.records:
            base = {k: v for k, v in r.items() if k not in ("items","_raw_text")}
            for item in (r.get("items") or [base]):
                row = base.copy()
                if r.get("items"):
                    row.update({"item_name": item.get("name"),
                                "item_qty": item.get("quantity"),
                                "item_unit_price": item.get("unit_price"),
                                "item_total": item.get("total_price")})
                rows.append(row)
        fields = list(rows[0].keys())
        for row in rows:
            for k in row:
                if k not in fields: fields.append(k)
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        self._set_status(f"✓ Exported {len(rows)} rows  →  {Path(path).name}")
        messagebox.showinfo("Exported", f"Saved to:\n{path}")

    def _clear(self):
        if not self.records: return
        if messagebox.askyesno("Confirm","Clear all scan results?"):
            self.records.clear()
            for t in [self.tree, self.items_tree]:
                for i in t.get_children(): t.delete(i)
            self.json_text.delete("1.0","end")
            self.raw_text.delete("1.0","end")
            self.count_badge.configure(text="0")
            self._set_status("Cleared")

#Status

    def _set_status(self, msg, level="normal"):
        color = {"normal": C["green"], "warning": C["accent2"], "error": C["red"]}.get(level, C["green"])
        self.status_var.set(msg)
        self.status_lbl.configure(fg=color)

    def on_close(self):
        self._stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ReceiptApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
