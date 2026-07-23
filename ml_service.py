import torch
import torch.nn.functional as F
import torch.distributions as dist
from pyvi import ViTokenizer

from config import aspect_columns, sentiment_labels
from data_processor import prepare_data_for_row, extract_opinion_phrase_gat
from schemas import AspectResult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_single_prediction(text: str, model) -> list[AspectResult]:
    """
    Xử lý text qua model ABSA và trả về danh sách các AspectResult.
    """
    # 1. Tiền xử lý dữ liệu
    segmented_text = ViTokenizer.tokenize(text)
    words_count = len(segmented_text.split())
    dummy_heads = [-1] * words_count

    row_simulated = {
        'Review': segmented_text,
        'text_segmented': segmented_text,
        'heads': dummy_heads
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