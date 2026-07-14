import torch
import torch.nn.functional as F
import torch.distributions as dist
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from vncorenlp import VnCoreNLP

from config import aspect_columns, sentiment_labels
from utils import prepare_data_for_row, extract_opinion_phrase_pure_attn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Đang tải mô hình lên {device}...")
    try:
        model = torch.load("model.pth", map_location=device)
        model.to(device)
        model.eval()
        ml_models["model"] = model
        print("✅ Tải mô hình AI thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình AI: {e}")

    print("Đang tải VnCoreNLP (Word Segmentation & Dependency Parsing)...")
    try:
        annotator = VnCoreNLP(
            "./libs/VnCoreNLP-1.2VnCoreNLP-1.1.1.jar", 
            annotators="wseg,pos,ner,parse", 
            max_heap_size='-Xmx2g'
        )
        ml_models["annotator"] = annotator
        print("✅ Tải VnCoreNLP thành công!")
    except Exception as e:
        print(f"Lỗi khi tải VnCoreNLP: {e}")
        
    yield 
    
    ml_models.clear()
    print("Đã giải phóng tài nguyên.")

app = FastAPI(lifespan=lifespan, title="ABSA Model API (KLTN)")

class PredictRequest(BaseModel):
    text: str

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    probability: float
    entropy: float
    attention: float
    opinion_word: str

class PredictResponse(BaseModel):
    original_text: str
    message: str
    results: List[AspectResult]

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = ml_models.get("model")
    annotator = ml_models.get("annotator")
    
    if not model or not annotator:
        raise HTTPException(status_code=503, detail="Server đang khởi động tài nguyên, vui lòng thử lại sau.")
        
    try:
        output = annotator.annotate(request.text)
        
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
            'Review': request.text, 
            'text_segmented': text_segmented, 
            'heads': heads
        }
        
        input_ids, attention_mask, e_syn, e_sem, e_asp = prepare_data_for_row(row_simulated)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        e_syn = e_syn.to(device)
        e_sem = e_sem.to(device)
        e_asp = e_asp.to(device)
        
        batch_idx = torch.zeros(162, dtype=torch.long).to(device)
        
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Lỗi phân tích VnCoreNLP: {e}")

    extracted_aspects = []
    
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits, attn_weights = model(
                    input_ids, attention_mask, e_syn, e_sem, e_asp, batch_idx
                )
                
                logits = logits[0]
                weights = attn_weights[0]
                
                probs = F.softmax(logits.float(), dim=-1)
                preds = logits.argmax(dim=-1)
                entropies = dist.Categorical(probs=probs).entropy()
                
                for asp_idx in range(34):
                    pred_class = preds[asp_idx].item()
                    asp_entropy = entropies[asp_idx].item()
                    max_prob = probs[asp_idx].max().item()
                    
                    opinion, max_attn = extract_opinion_phrase_pure_attn(
                        row_simulated['text_segmented'], weights, asp_idx, window=3
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Inference Model: {e}")

    if not extracted_aspects:
        message = "Mô hình không tìm thấy khía cạnh nào nổi bật (Do tín hiệu quá nhiễu/Entropy cao)."
    else:
        message = "Dự đoán thành công."

    return PredictResponse(
        original_text=request.text,
        message=message,
        results=extracted_aspects
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8800, reload=True)