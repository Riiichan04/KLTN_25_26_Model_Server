import re
import emoji
import torch
import torch.nn.functional as F
import torch.distributions as dist

from config import aspect_columns, sentiment_labels
from data_processor import prepare_data_for_row, extract_opinion_phrase_gat
from schemas import AspectResult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản, chuyển emoji thành chữ và loại bỏ ký tự rác.
    """
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_single_prediction(text: str, model, annotator) -> list[AspectResult]:
    """
    Xử lý text qua model ABSA và trả về danh sách các AspectResult.
    """
    # 1. Tiền xử lý dữ liệu (Áp dụng logic Global Head Offset)
    cleaned_text = clean_text(text)
    output = annotator.annotate(cleaned_text)

    words = []
    heads = []
    word_offset = 0

    for sentence in output['sentences']:
        for word_info in sentence:
            words.append(word_info['form'])

            head_val = word_info['head']
            if head_val == 0:
                heads.append(-1)
            else:
                global_head_idx = (head_val - 1) + word_offset
                heads.append(global_head_idx)

        word_offset += len(sentence)

    text_segmented = " ".join(words)

    row_simulated = {
        'Review': text_segmented,
        'text_segmented': text_segmented,
        'heads': heads
    }

    input_ids, attention_mask, e_syn, e_sem, e_asp = prepare_data_for_row(row_simulated)

    # Đưa lên device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    e_syn = e_syn.to(device)
    e_sem = e_sem.to(device)
    e_asp = e_asp.to(device)
    batch_idx = torch.zeros(162, dtype=torch.long).to(device)

    extracted_aspects = []

    # 2. Suy luận (Inference)
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits, (edges_asp, gat_attn) = model(
                input_ids,
                attention_mask,
                e_syn,
                e_sem,
                e_asp,
                batch_idx
            )

            logits = logits[0]

            # Tính Softmax & Dự đoán
            probs = F.softmax(logits.float(), dim=-1)
            preds = logits.argmax(dim=-1)
            entropies = dist.Categorical(probs=probs).entropy()

            # 3. Trích xuất kết quả
            for asp_idx in range(34):
                pred_class = preds[asp_idx].item()
                asp_entropy = entropies[asp_idx].item()
                max_prob = probs[asp_idx].max().item()

                opinion, max_attn = extract_opinion_phrase_gat(
                    row_simulated['text_segmented'],
                    edges_asp,
                    gat_attn,
                    asp_idx,
                    window=3
                )

                if pred_class != 0 and asp_entropy < 0.7 and max_prob > 0.65 and max_attn > 0.15:
                    extracted_aspects.append(
                        AspectResult(
                            aspect=aspect_columns[asp_idx],
                            sentiment=sentiment_labels.get(pred_class, "Unknown"),
                            probability=round(max_prob, 4),
                            entropy=round(asp_entropy, 3),
                            attention=round(max_attn, 4),
                            opinion_word=opinion
                        )
                    )

    return extracted_aspects