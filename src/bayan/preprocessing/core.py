"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""
import re
import unicodedata


PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""

    if not isinstance(text, str):
     raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789"
))
    # حذف التشكيل والتطويل
    text = re.sub(
        r"[\u0610-\u061A\u0640\u064B-\u065F\u0670\u06D6-\u06ED]",
          "",
         text, )
    # توحيد بعض أشكال الحروف العربية
    
    

    # تقليص تكرار الحرف أو الرمز إلى مرتين كحد أقصى
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
   
    # حذف المسافات الزائدة
    text = re.sub(r"\s+", " ", text).strip()

    # المحافظة على وسوم إخفاء البيانات بأحرف كبيرة
    text = text.replace("<phone>", "<PHONE>")
    text = text.replace("<national_id>", "<NATIONAL_ID>")

    return text



def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    if not isinstance(text, str):
     raise TypeError("text must be a string")
    # تحويل الأرقام العربية والفارسية إلى أرقام إنجليزية
    text = text.translate(str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789"
     ))

    phone_pattern = re.compile( r""" (?<!\d)(?:(?:\+966|00966|966)[\s().-]*5 | 0?5 )
    (?:[\s().-]*\d){8} (?!\d)  """, re.VERBOSE,)

    national_id_pattern = re.compile( r"(?<!\d)[12]\d{9}(?!\d)")

    # نخفي رقم الجوال أولًا حتى لا يتداخل مع نمط الهوية
    text = phone_pattern.sub("<PHONE>", text)
    text = national_id_pattern.sub("<NATIONAL_ID>", text)

    return text



def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    return normalize(mask_pii(text))
