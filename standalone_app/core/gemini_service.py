#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Tích hợp Gemini LLM cho Hệ thống Nhận diện Ngôn ngữ Ký hiệu.
Chức năng:
  - Nhận danh sách các từ đơn (cử chỉ nhận diện từ 2 model AI 1 tay và 2 tay).
  - Ghép các từ rời rạc thành câu tiếng Việt tự nhiên, chuẩn ngữ pháp, đúng ngữ cảnh.
  - Hỗ trợ nhiều phong cách: Tự nhiên, Lịch sự/Kính ngữ, Ngắn gọn, Nghi vấn, Song ngữ Anh-Việt.
  - Sử dụng REST API chính thức của Google Gemini thông qua thư viện requests.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
SUPPORTED_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Cache lưu trữ danh sách model thực sự khả dụng theo API Key
_MODELS_CACHE = {}


def normalize_model_name(name: str) -> str:
    """Loại bỏ tiền tố models/ nếu có."""
    m = (name or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    return m or DEFAULT_GEMINI_MODEL


def preprocess_vsl_tokens(words: List[str]) -> List[str]:
    """
    Tiền xử lý chuỗi từ ký hiệu thô:
    1. Gộp các từ trùng lặp liên tiếp do giữ tay quá lâu.
    2. Chuyển đổi nhãn kép 'có / đúng' thành 'có' (hoặc 'đúng' tùy ngữ cảnh).
    3. Ghép các chuỗi chữ số ký hiệu liên tiếp (ví dụ: '1', '00', '000' -> '100.000').
    """
    if not words:
        return []

    # 1. Gộp từ lặp liên tiếp
    deduped = []
    for w in words:
        w_clean = str(w).strip()
        if not w_clean:
            continue
        if not deduped or deduped[-1] != w_clean:
            deduped.append(w_clean)

    # 2. Ghép chuỗi số và làm sạch từ
    merged = []
    i = 0
    while i < len(deduped):
        token = deduped[i]
        # Kiểm tra nếu token là số hoặc chuỗi số '00', '000'
        if token.isdigit():
            num_tokens = [token]
            j = i + 1
            while j < len(deduped) and deduped[j].isdigit():
                num_tokens.append(deduped[j])
                j += 1
            if len(num_tokens) > 1:
                full_num_str = "".join(num_tokens)
                try:
                    val = int(full_num_str)
                    formatted_num = f"{val:,}".replace(",", ".")
                    merged.append(formatted_num)
                except ValueError:
                    merged.append(full_num_str)
                i = j
                continue
            else:
                merged.append(token)
        else:
            if token == "có / đúng":
                # Nhãn 'có / đúng': trong ngữ cảnh câu hỏi hoặc sở hữu thì mang nghĩa 'có'
                merged.append("có")
            else:
                merged.append(token)
        i += 1

    return merged


STYLE_PROMPTS = {
    "natural": (
        "Yêu cầu phong cách: TỰ NHIÊN, KHẨU NGỮ HÀNG NGÀY. "
        "Hãy viết như một câu nói chuyện đời thường mượt mà, thân thiện của người Việt Nam. "
        "Bổ sung hư từ, từ nối, đại từ chỉ định, trợ từ (nhé, với, ạ, này, thì, là, mà...). "
        "Tuyệt đối không ghép dịch thô cứng từng từ một."
    ),
    "polite": (
        "Yêu cầu phong cách: LỊCH SỰ, TRANG TRỌNG. "
        "Sử dụng kính ngữ tiếng Việt (như 'Dạ', 'Thưa', 'ạ', 'làm ơn', 'xin phép'...) "
        "để tạo thành câu giao tiếp nhã nhặn, tôn trọng đối phương."
    ),
    "concise": (
        "Yêu cầu phong cách: NGẮN GỌN, SÚC TÍCH. "
        "Tạo câu ngắn gọn nhất có thể nhưng vẫn giữ đầy đủ ý nghĩa và chuẩn ngữ pháp."
    ),
    "question": (
        "Yêu cầu phong cách: CÂU HỎI / NGHI VẤN. "
        "Tạo thành một câu hỏi tự nhiên (dùng 'hả', 'không', 'vậy ạ', 'ở đâu', 'như thế nào'...), "
        "kết thúc bằng dấu chấm hỏi '?'."
    ),
    "bilingual": (
        "Yêu cầu phong cách: SONG NGỮ VIỆT - ANH. "
        "Tạo câu tiếng Việt tự nhiên ở dòng 1, và bản dịch tiếng Anh chuẩn ở dòng 2."
    ),
}

SYSTEM_INSTRUCTION = """
Bạn là một chuyên gia ngôn ngữ học và phiên dịch viên Ngôn ngữ Ký hiệu Việt Nam (VSL - Vietnamese Sign Language) sang Tiếng Việt giao tiếp tự nhiên hàng đầu.

Bối cảnh:
Các từ nhận diện được trích xuất từ mô hình AI thị giác máy tính nhận diện cử chỉ bàn tay của người khiếm thính. Người sử dụng ký hiệu chỉ biểu thị các từ khóa cốt lõi (Keywords) và lược bỏ hầu hết ngữ pháp, hư từ, liên từ hoặc trợ từ.

Các nguyên tắc BẮT BUỘC khi tạo câu:
1. XỬ LÝ NHÃN KÉP CÓ DẤU GẠCH CHÉO '/':
   - Nhãn "có / đúng" biểu thị cử chỉ mang nghĩa "có" (sở hữu, tồn tại, có thể) hoặc "đúng" (chính xác, khẳng định, đồng ý).
   - Bạn PHẢI tự động chọn từ "có" hoặc "đúng" phù hợp với ngữ cảnh câu.
   - TUYỆT ĐỐI KHÔNG BAO GIỜ để nguyên văn chuỗi "có / đúng" trong câu đầu ra!
   - Ví dụ: ["bạn", "có / đúng", "cái gì"] -> "Bạn có cái gì" (chọn "có").

2. GHÉP CÁC CHỮ SỐ KÝ HIỆU (NUMBER CONCATENATION):
   - Người khiếm thính biểu thị các số lớn bằng chuỗi ký hiệu liên tiếp:
     Ví dụ: ["1", "00", "000"] -> ghép thành số "100.000" (hoặc "100.000 đồng" khi nói về giá cả).
     Ví dụ: ["5", "0", "000"] -> "50.000", ["2", "0"] -> "20".
   - Hãy ghép các token số liền nhau thành một con số hoàn chỉnh, rõ ràng.

3. VĂN PHONG TỰ NHIÊN, KHÔNG DỊCH THÔ CỨNG (NATURAL SPOKEN VIETNAMESE):
   - TUYỆT ĐỐI KHÔNG ghép thô từng từ một (word-by-word) khiến câu gượng ép.
   - Bổ sung các hư từ, từ nối, đại từ, trợ từ tình thái tiếng Việt cần thiết (ví dụ: "thì", "là", "mà", "ở", "với", "tiền", "đồng", "món", "của", "ạ", "nhé", "vậy", "thế", "được không", "giúp đỡ"...).
   - Nếu câu chứa từ để hỏi ("cái gì", "ở đâu", "khi nào", "ai", "tại sao", "bao nhiêu", "không") -> viết thành câu hỏi tự nhiên với dấu chấm hỏi "?" ở cuối.
   - Lọc bỏ các từ lặp lại liên tiếp do giữ tay quá lâu (ví dụ: ["xin chào", "xin chào"] -> "xin chào").

CÁC VÍ DỤ MẪU (FEW-SHOT EXAMPLES):
- Từ nhận diện: ["xin chào", "bạn", "có / đúng", "cái gì", "giá", "1", "00", "000", "không"]
  -> Câu hoàn chỉnh: "Xin chào, bạn có cái gì giá 100.000 không?"
- Từ nhận diện: ["tên", "bạn", "cái gì"]
  -> Câu hoàn chỉnh: "Bạn tên là gì vậy?"
- Từ nhận diện: ["giá", "cái gì", "bao nhiêu"]
  -> Câu hoàn chỉnh: "Cái này giá bao nhiêu tiền vậy?"
- Từ nhận diện: ["giá", "2", "00", "000"]
  -> Câu hoàn chỉnh: "Món này giá 200.000 đồng."
- Từ nhận diện: ["xin lỗi", "tôi", "muốn", "giúp", "bạn"]
  -> Câu hoàn chỉnh: "Xin lỗi, tôi rất muốn giúp đỡ bạn."
- Từ nhận diện: ["cảm ơn", "bạn", "rất", "tốt"]
  -> Câu hoàn chỉnh: "Cảm ơn bạn nhiều, bạn tốt bụng quá!"
- Từ nhận diện: ["ở đâu", "giúp", "chúng tôi", "làm ơn"]
  -> Câu hoàn chỉnh: "Có ai ở đây có thể làm ơn giúp chúng tôi không?"
- Từ nhận diện: ["bạn", "tại sao", "không", "giúp"]
  -> Câu hoàn chỉnh: "Tại sao bạn lại không giúp đỡ tôi?"
- Từ nhận diện: ["tạm biệt", "bạn", "yêu"]
  -> Câu hoàn chỉnh: "Tạm biệt bạn, tôi yêu quý bạn rất nhiều!"

ĐỊNH DẠNG ĐẦU RA:
CHỈ TRẢ VỀ DUY NHẤT một chuỗi JSON hợp lệ (không kèm Markdown block thừa):
{
  "sentence": "<câu tiếng Việt hoàn chỉnh, tự nhiên, đúng ngữ pháp>",
  "cleaned_words": ["từ1", "từ2", ...],
  "confidence_explanation": "<giải thích ngắn gọn 1 câu>",
  "english_translation": "<bản dịch tiếng Anh tương đương>"
}
"""


class GeminiSentenceBuilder:
    """
    Dịch vụ kết nối Google Gemini API để ghép từ ký hiệu thành câu tiếng Việt hoàn chỉnh.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GEMINI_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = normalize_model_name(model)

    def set_api_key(self, api_key: str):
        """Cập nhật Gemini API Key."""
        self.api_key = api_key.strip()

    def set_model(self, model: str):
        """Cập nhật Model Gemini."""
        self.model = normalize_model_name(model)

    def get_working_models_for_key(self, api_key: str) -> List[str]:
        """Tự động khám phá các model khả dụng cho API Key từ Google ModelService."""
        key = api_key.strip()
        if not key:
            return []
        if key in _MODELS_CACHE:
            return _MODELS_CACHE[key]

        discovered = []
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                models_data = resp.json().get("models", [])
                for m in models_data:
                    name = m.get("name", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods and name.startswith("models/"):
                        m_id = name.split("/")[-1]
                        discovered.append(m_id)
                if discovered:
                    # Sắp xếp ưu tiên: flash trước, pro sau
                    flash = [m for m in discovered if "flash" in m]
                    others = [m for m in discovered if "flash" not in m]
                    discovered = flash + others
                    _MODELS_CACHE[key] = discovered
        except Exception:
            pass

        return discovered

    def test_connection(self, api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Kiểm tra xem API Key và model có hoạt động hay không.
        """
        key = (api_key or self.api_key).strip()
        if not key:
            return {"valid": False, "message": "Chưa cung cấp Gemini API Key."}

        # Khám phá các model khả dụng cho key này
        available = self.get_working_models_for_key(key)
        preferred_model = normalize_model_name(model or self.model)
        if available and preferred_model not in available:
            preferred_model = available[0]

        target_model = preferred_model
        payload = {
            "contents": [
                {
                    "parts": [{"text": "Xin chào! Trả lời 'OK' nếu kết nối thành công."}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 10,
                "temperature": 0.1
            }
        }

        # Thử v1beta sau đó thử v1
        last_msg = ""
        for api_ver in ("v1beta", "v1"):
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{target_model}:generateContent?key={key}"
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    self.model = target_model
                    return {
                        "valid": True,
                        "message": f"Kết nối Gemini API thành công với model {target_model} ({api_ver})!",
                        "model": target_model
                    }
                else:
                    err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    last_msg = err_data.get("error", {}).get("message", resp.text)
            except Exception as e:
                last_msg = str(e)

        return {
            "valid": False,
            "message": f"Lỗi Gemini API: {last_msg}"
        }

    def combine_words_to_sentence(
        self,
        words: List[str],
        style: str = "natural",
        custom_prompt: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ghép danh sách từ đơn thành câu hoàn chỉnh bằng Gemini LLM.

        :param words: Danh sách các từ nhận diện (ví dụ: ['xin chào', 'tôi', 'muốn', 'giúp', 'bạn'])
        :param style: 'natural', 'polite', 'concise', 'question', 'bilingual'
        :param custom_prompt: Gợi ý hoặc yêu cầu bổ sung nếu có
        :param model_override: Ghi đè tên model cho lần gọi này
        :return: Dict chứa câu hoàn chỉnh, từ đã làm sạch, và thông tin chi tiết.
        """
        if not words:
            return {
                "success": False,
                "error": "Danh sách từ rỗng. Vui lòng thực hiện cử chỉ ký hiệu trước.",
                "sentence": "",
                "original_words": []
            }

        # Tiền xử lý từ ký hiệu (gộp lặp, đổi 'có / đúng' -> 'có', ghép số '1'+'00'+'000' -> '100.000')
        cleaned_tokens = preprocess_vsl_tokens(words)

        key = self.api_key.strip()
        if not key:
            # Fallback thông minh nếu chưa có API Key
            fallback_text = " ".join(cleaned_tokens).capitalize()
            is_question = any(q in fallback_text.lower() for q in ["cái gì", "ở đâu", "khi nào", "ai", "tại sao", "bao nhiêu", "không"])
            punct = "?" if is_question else "."
            if not fallback_text.endswith(('.', '?', '!')):
                fallback_text += punct
            return {
                "success": False,
                "error": "Chưa có Gemini API Key! Hãy cấu hình API Key ở khung cài đặt.",
                "sentence": fallback_text,
                "original_words": words,
                "cleaned_words": cleaned_tokens,
                "is_fallback": True
            }

        style_instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS["natural"])
        user_content = (
            f"Danh sách từ khóa ký hiệu nhận diện được: {json.dumps(cleaned_tokens, ensure_ascii=False)}\n"
            f"(Từ gốc thu thập từ cảm biến AI: {json.dumps(words, ensure_ascii=False)})\n"
            f"{style_instruction}\n"
            f"Hãy ghép các từ khóa trên thành một câu Tiếng Việt hoàn chỉnh, tự nhiên, đúng ngữ cảnh giao tiếp."
        )
        if custom_prompt:
            user_content += f"Yêu cầu bổ sung của người dùng: {custom_prompt}\n"

        # Tự động lấy danh sách model thực sự khả dụng từ Google hoặc danh sách dự phòng ưu tiên
        requested_model = normalize_model_name(model_override or self.model)
        if requested_model in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"):
            requested_model = "gemini-3.5-flash"

        discovered = self.get_working_models_for_key(key)
        if discovered:
            candidate_models = ([requested_model] if requested_model in discovered else []) + [m for m in discovered if m != requested_model]
        else:
            candidate_models = [
                requested_model,
                "gemini-3.5-flash",
                "gemini-3.8-flash",
                "gemini-1.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-pro"
            ]

        # Lọc trùng và loại bỏ các model không hỗ trợ
        seen = set()
        unique_candidates = [
            m for m in candidate_models 
            if m and m not in seen and not seen.add(m) and m not in ("gemini-2.5-flash", "gemini-2.5-pro")
        ]

        last_error = ""
        for current_model in unique_candidates:
            # Thử qua cả 2 endpoint v1beta và v1 để tương thích tối đa với API Key của người dùng
            for api_ver in ("v1beta", "v1"):
                url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{current_model}:generateContent?key={key}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": user_content}]
                        }
                    ],
                    "systemInstruction": {
                        "parts": [{"text": SYSTEM_INSTRUCTION}]
                    },
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 400,
                        "responseMimeType": "application/json"
                    }
                }

                try:
                    resp = requests.post(url, json=payload, timeout=12)
                    if resp.status_code == 200:
                        result_json = resp.json()
                        candidates = result_json.get("candidates", [])
                        if candidates:
                            content_parts = candidates[0].get("content", {}).get("parts", [])
                            raw_text = content_parts[0].get("text", "") if content_parts else ""

                            # Phân tích JSON kết quả
                            try:
                                parsed = json.loads(raw_text)
                                sentence = parsed.get("sentence", "").strip()
                                explanation = parsed.get("confidence_explanation", "")
                                english = parsed.get("english_translation", "")
                                cleaned = parsed.get("cleaned_words", cleaned_tokens)
                            except Exception:
                                sentence = raw_text.strip()
                                explanation = ""
                                english = ""
                                cleaned = cleaned_tokens

                            # Ghi nhớ model đã thành công cho các lần gọi sau
                            self.model = current_model

                            return {
                                "success": True,
                                "sentence": sentence,
                                "original_words": words,
                                "cleaned_words": cleaned,
                                "style": style,
                                "model": current_model,
                                "english_translation": english,
                                "explanation": explanation
                            }
                    else:
                        err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                        err_msg = err_data.get("error", {}).get("message", resp.text)
                        last_error = f"Gemini API Error ({resp.status_code}): {err_msg}"
                except Exception as e:
                    last_error = f"Lỗi ngoại lệ: {str(e)}"
                    continue

        return {
            "success": False,
            "error": last_error or "Tất cả các mô hình Gemini hiện đang bận hoặc quá tải.",
            "sentence": " ".join(cleaned_tokens).capitalize() + ".",
            "original_words": words,
            "cleaned_words": cleaned_tokens,
            "is_fallback": True
        }

