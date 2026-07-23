import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from vncorenlp import VnCoreNLP

from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse
)
from ml_service import process_single_prediction, device
from model_network import EndToEndAspectModel
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Đang tải mô hình lên {device}...")
    try:
        model = EndToEndAspectModel()

        state_dict = torch.load("model.pth", map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        ml_models["model"] = model
        print("✅ Tải mô hình AI thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình AI: {e}")

    print("Đang tải VnCoreNLP (Word Segmentation & Dependency Parsing)...")
    try:
        annotator = VnCoreNLP(
            "./libs/VnCoreNLP-1.2.jar",
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

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = ml_models.get("model")
    annotator = ml_models.get("annotator")

    if not model or not annotator:
        raise HTTPException(status_code=503, detail="Server đang khởi động tài nguyên, vui lòng thử lại sau.")

    try:
        extracted_aspects = process_single_prediction(request.text, model, annotator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Inference Model: {e}")

    message = "Dự đoán thành công." if extracted_aspects else "Mô hình không tìm thấy khía cạnh nào nổi bật (Do tín hiệu quá nhiễu/Entropy cao)."

    return PredictResponse(
        original_text=request.text,
        message=message,
        results=extracted_aspects
    )


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    model = ml_models.get("model")
    annotator = ml_models.get("annotator")

    if not model or not annotator:
        raise HTTPException(status_code=503, detail="Server đang khởi động tài nguyên, vui lòng thử lại sau.")

    all_batch_results = []
    success_count = 0

    for item in request.reviews:
        try:
            extracted_aspects = process_single_prediction(item.text, model, annotator)
            msg = "Thành công" if extracted_aspects else "Không tìm thấy khía cạnh"

            all_batch_results.append(PredictResponse(
                id=item.id,
                original_text=item.text,
                message=msg,
                results=extracted_aspects
            ))
            success_count += 1

        except Exception as e:
            all_batch_results.append(PredictResponse(
                id=item.id,
                original_text=item.text,
                message=f"Lỗi: {str(e)}",
                results=[]
            ))

    return BatchPredictResponse(
        message="Hoàn tất xử lý Batch.",
        total_processed=success_count,
        batch_results=all_batch_results
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8800, reload=True)