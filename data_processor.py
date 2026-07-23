# row: Một dòng dữ liệu lấy ra từ DataFrame (chứa câu review, danh sách heads từ parser, và các nhãn).
# max_len=128: Chiều dài tối đa của câu sau khi token hóa.
# num_aspects=34: Số lượng khía cạnh.
import torch
from graph import build_syn_graph, build_sem_graph, build_asp_graph
from transformers import AutoTokenizer

# Khởi tạo phoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def prepare_data_for_row(row, max_len=128, num_aspects=34):
    # Mã hóa text bằng tokenizer
    # padding='max_length' sẽ nhồi thêm các token rỗng (padding) nếu câu ngắn hơn 128
    # truncation=True sẽ cắt bỏ phần đuôi nếu câu dài hơn 128.
    # return_tensors='pt' báo cho tokenizer trả về định dạng tensor của PyTorch thay vì list thuần.
    encoded = tokenizer(
        row['Review'],
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    # Mặc định khi đưa một câu vào tokenizer, nó sẽ trả về tensor có kích thước [1, 128]
    # squeeze(0) để ép phẳng về tensor [128]
    input_ids = encoded['input_ids'].squeeze(0)
    attention_mask = encoded['attention_mask'].squeeze(0)

    # Trong danh sách heads, thêm padding nếu câu < 128 từ, và cắt bớt nếu câu > 128 từ
    raw_heads = row['heads']
    safe_heads = []
    for i in range(max_len):
        if i < len(raw_heads):
            h = raw_heads[i]
            safe_heads.append(-1 if h >= max_len else h)
        else:
            safe_heads.append(-1)

    # Khởi tạo 3 đồ thị
    total_n = max_len + num_aspects
    e_syn = build_syn_graph(safe_heads, seq_len=max_len, total_nodes=total_n)
    e_sem = build_sem_graph(seq_len=max_len, total_nodes=total_n, window=2)
    e_asp = build_asp_graph(seq_len=max_len, num_aspects=num_aspects)

    return input_ids, attention_mask, e_syn, e_sem, e_asp


def extract_opinion_phrase_gat(text_segmented, edges_asp, gat_attn, aspect_idx, window=3):
    words = text_segmented.split()
    if len(words) == 0: return "", 0.0

    attn_weights = gat_attn.mean(dim=1)
    src, dst = edges_asp[0], edges_asp[1]
    target_node = 128 + aspect_idx

    mask = (dst == target_node) & (src < 128)
    word_indices = src[mask]
    weights = attn_weights[mask]

    valid_weights = torch.full((len(words),), -1e9)

    for w_idx, w_weight in zip(word_indices, weights):
        real_word_idx = w_idx.item() - 1
        if 0 <= real_word_idx < len(words):
            valid_weights[real_word_idx] = w_weight.item()

    if torch.all(valid_weights == -1e9): return "", 0.0

    # # Bổ sung các từ ngắt ngữ cảnh (chống lây lan attention sang khía cạnh kế bên)
    # stop_words = {'.', ',', '!', '?', ':', ';', '-', '...', '..', 'nhưng', 'tuy_nhiên', 'mặc_dù'}
    # for i, word in enumerate(words):
    #     if word in stop_words or word.strip() == "":
    #         valid_weights[i] = -1e9

    core_idx = torch.argmax(valid_weights).item()
    if valid_weights[core_idx] == -1e9: return "", 0.0

    max_attn = valid_weights[core_idx].item()
    related_indices = [core_idx]

    # ÉP ĐIỀU KIỆN KHẮT KHE HƠN: Từ lân cận phải có Attention > 40% đỉnh
    threshold = max_attn * 0.40

    # Quét trái
    for i in range(core_idx - 1, max(-1, core_idx - window - 1), -1):
        # if words[i] in stop_words: break
        if valid_weights[i].item() > threshold:
            related_indices.append(i)
        else:
            break  # Ngắt ngay nếu Attention rơi tự do

    # Quét phải
    for i in range(core_idx + 1, min(len(words), core_idx + window + 1)):
        # if words[i] in stop_words: break
        if valid_weights[i].item() > threshold:
            related_indices.append(i)
        else:
            break

    related_indices = sorted(list(set(related_indices)))
    opinion_phrase = " ".join([words[i] for i in related_indices])

    return opinion_phrase.replace("_", " "), max_attn