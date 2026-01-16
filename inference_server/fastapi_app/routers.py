"""
FastAPI Routers
===============
API routers for the FastAPI inference server.
"""

import base64
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile

from .dependencies import (
    get_inference_engine,
    get_optional_inference_engine,
    get_validation_pipeline,
    get_api_key_manager,
    get_phase3_manager,
    verify_api_key,
    check_rate_limit,
    require_admin,
    get_request_context,
    RequestContext,
)
from .schemas import (
    HealthResponse,
    PredictionResponse,
    BatchPredictionResponse,
    ClassesResponse,
    ModelsResponse,
    ErrorResponse,
    RejectionResponse,
    Base64ImageRequest,
    PredictionResult,
    InferenceInfo,
    BatchResultItem,
    BatchSummary,
    ClassInfo,
    ModelInfo,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    FeedbackRecorded,
    # Phase 3 schemas
    Phase3Response,
    Phase3AttentionInfo,
    Phase3RegionInfo,
    Phase3MultiLabelPrediction,
)
from ..feedback import FeedbackManager
from ..feedback.feedback_manager import get_feedback_manager
from ..feedback.user_tracker import get_user_tracker
from ..feedback.data_collector import get_data_collector

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter(prefix="/api/v1", tags=["Inference API"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["Admin API"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    engine=Depends(get_optional_inference_engine),
):
    """
    Health check endpoint.
    
    Returns service health status and model information.
    """
    model_loaded = engine is not None
    model_info = None
    
    if engine:
        try:
            model_info = {
                "format": engine.model_format,
                "device": engine.device,
                "num_classes": engine.num_classes,
            }
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version="1.0.0",
        model=ModelInfo(loaded=model_loaded, info=model_info),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models.
    
    Returns publicly exposed models only.
    """
    try:
        from ..engine.model_registry import get_registry
        
        registry = get_registry()
        exposed_models = registry.get_exposed_models()
        
        return ModelsResponse(
            status="success",
            models=exposed_models,
        )
    except Exception as e:
        logger.exception("Error listing models")
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_LIST_ERROR", "message": str(e)},
        )


@router.get("/classes", response_model=ClassesResponse)
async def list_classes(
    engine=Depends(get_optional_inference_engine),
):
    """
    List pest classes.
    
    Returns all pest classes the model can detect.
    """
    default_classes = [
        "Healthy",
        "army worm",
        "Internode borer",
        "mealy bug",
        "Pink borer",
        "porcupine damage",
        "Rat damage",
        "root borer",
        "Stalk borer",
        "termite",
        "Top borer",
    ]
    
    classes = engine.class_names if engine else default_classes
    
    return ClassesResponse(
        status="success",
        num_classes=len(classes),
        classes=[
            ClassInfo(id=i, name=name)
            for i, name in enumerate(classes)
        ],
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Image file to classify"),
    include_probabilities: bool = Query(False, description="Include all class probabilities"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id", description="User ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email", description="User email"),
    x_device_id: Optional[str] = Header(None, alias="X-Device-Id", description="Device ID"),
    x_latitude: Optional[str] = Header(None, alias="X-Latitude", description="GPS latitude"),
    x_longitude: Optional[str] = Header(None, alias="X-Longitude", description="GPS longitude"),
    x_app_version: Optional[str] = Header(None, alias="X-App-Version", description="App version"),
    api_key: dict = Depends(check_rate_limit),
    engine=Depends(get_inference_engine),
    pipeline=Depends(get_validation_pipeline),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Single image prediction with user tracking.
    
    Upload an image to get pest classification.
    Include user info via headers (X-User-Id, X-User-Email, X-Device-Id, X-Latitude, X-Longitude, X-App-Version).
    Returns a feedback_id that can be used to submit corrections.
    """
    # Parse header values
    user_id = x_user_id
    email = x_user_email
    device_id = x_device_id
    app_version = x_app_version
    latitude = float(x_latitude) if x_latitude else None
    longitude = float(x_longitude) if x_longitude else None
    
    # Read image
    image_bytes = await image.read()
    filename = image.filename or "upload"
    
    # Track user submission
    user_tracker = get_user_tracker()
    is_flagged_user = False
    user_trust_score = None
    
    if user_tracker and user_id:
        import hashlib
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        track_result = user_tracker.record_submission(
            user_id=user_id,
            image_hash=image_hash,
            latitude=latitude,
            longitude=longitude,
            email=email,
            device_id=device_id,
        )
        is_flagged_user = track_result.get("is_flagged", False)
        user_trust_score = track_result.get("trust_score")
    
    # Validate image
    if pipeline:
        validation_result = pipeline.validate_pre_inference(image_bytes, filename)
        
        if not validation_result.valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": validation_result.error_code or "IMAGE_VALIDATION_FAILED",
                    "message": validation_result.error_message or "Validation failed",
                    "failed_layer": validation_result.failed_layer.value if validation_result.failed_layer else None,
                },
            )
    
    # Run inference
    try:
        result = engine.predict(image_bytes)
        inference_time = ctx.elapsed_ms
        
        class_name = result.get("class_name", "")
        class_id = result.get("predicted_class", -1)
        confidence = result.get("confidence", 0.0)
        all_probs = result.get("probabilities", {})
        
        # Check if model is confident enough
        # If confidence is too low, the image might be unclear or not a valid pest image
        MIN_CONFIDENCE_THRESHOLD = 0.35  # Below this, ask for clearer image
        UNCERTAIN_THRESHOLD = 0.50  # Below this, warn about uncertainty
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            # Check entropy of predictions - high entropy means model is very uncertain
            if isinstance(all_probs, dict) and len(all_probs) > 1:
                import math
                probs_list = list(all_probs.values())
                entropy = -sum(p * math.log(p + 1e-10) for p in probs_list if p > 0)
                max_entropy = math.log(len(probs_list))  # Maximum possible entropy
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Very high entropy (> 0.8) means model can't distinguish - likely not a valid image
                if normalized_entropy > 0.8:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "IMAGE_UNCLEAR",
                            "message": "Unable to identify the image clearly. Please capture a clearer, well-lit photo of the affected plant area.",
                            "confidence": round(confidence, 4),
                            "suggestion": "Ensure good lighting and focus on the damaged/affected area of the plant.",
                        },
                    )
            
            # Low confidence but not high entropy - might be edge case
            logger.warning(f"Low confidence prediction: {class_name} at {confidence:.2%}")
        
        # Collect image data silently
        data_collector = get_data_collector()
        image_hash = None
        if data_collector:
            image_hash = data_collector.collect_image(
                image_bytes=image_bytes,
                predicted_class=class_name,
                predicted_class_id=class_id,
                confidence=confidence,
                user_id=user_id,
                email=email,
                device_id=device_id,
                latitude=latitude,
                longitude=longitude,
                request_id=ctx.request_id,
                app_version=app_version,
                original_filename=filename,
                all_probabilities=all_probs if isinstance(all_probs, dict) else None,
                user_trust_score=user_trust_score,
                is_flagged_user=is_flagged_user,
            )
        
        # Register for feedback
        feedback_id = None
        feedback_mgr = get_feedback_manager()
        if feedback_mgr:
            feedback_id = feedback_mgr.register_prediction(
                predicted_class=class_name,
                predicted_class_id=class_id,
                confidence=confidence,
                image_bytes=image_bytes,
                request_id=ctx.request_id,
                image_hash=image_hash,
                user_id=user_id,
            )
        
        # Build response
        prediction = PredictionResult(
            **{
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),
            }
        )
        
        if include_probabilities:
            probs = result.get("probabilities", {})
            if isinstance(probs, dict):
                prediction.all_probabilities = {
                    k: round(v, 4) for k, v in probs.items()
                }
        
        # Run Phase 3 analysis if available
        phase3_data = None
        phase3_manager = get_phase3_manager()
        if phase3_manager and phase3_manager.is_operational():
            try:
                # Get tensors from result (added by pytorch_inference)
                image_tensor = result.get("image_tensor")
                logits_tensor = result.get("logits_tensor", result.get("logits"))
                model = getattr(engine, 'model', None)
                
                if model is not None and image_tensor is not None:
                    p3_result = phase3_manager.run_inference(
                        model, image_tensor, logits_tensor, class_id
                    )
                    
                    if not p3_result.is_empty():
                        phase3_data = {
                            "executed": p3_result.phase3_executed,
                            "processing_time_ms": p3_result.execution_time_ms,
                            "error": p3_result.failure_message if p3_result.had_failure else None,
                        }
                        
                        # Add regions from p3_result.regions (not relevance_scores)
                        if not p3_result.regions.is_empty():
                            regions_list = []
                            for proposal in p3_result.regions.proposals:
                                regions_list.append({
                                    "region_id": proposal.region_id,
                                    "bbox": list(proposal.bbox) if proposal.bbox else None,
                                    "relevance_score": proposal.confidence,
                                    "label": None,
                                })
                            phase3_data["regions"] = regions_list
                            phase3_data["top_region_score"] = max(r["relevance_score"] for r in regions_list) if regions_list else None
                            
                        # Add relevance scores if available
                        if not p3_result.relevance_scores.is_empty():
                            if phase3_data.get("regions"):
                                for r in phase3_data["regions"]:
                                    score = p3_result.relevance_scores.region_scores.get(r["region_id"], r["relevance_score"])
                                    r["relevance_score"] = float(score)
                                phase3_data["top_region_score"] = max(r["relevance_score"] for r in phase3_data["regions"])
                        
                        # Add multi-label predictions
                        if not p3_result.multi_label.is_empty():
                            predictions = []
                            class_names = getattr(engine, 'class_names', None) or []
                            for class_id in p3_result.multi_label.predicted_labels:
                                label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                                conf = p3_result.multi_label.label_confidences.get(class_id, 0.0)
                                predictions.append({
                                    "label": label_name,
                                    "confidence": float(conf),
                                })
                            # Also add from class_probabilities if available
                            if not predictions and p3_result.multi_label.class_probabilities:
                                for cid, prob in sorted(p3_result.multi_label.class_probabilities.items(), 
                                                       key=lambda x: x[1], reverse=True)[:3]:
                                    label_name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                                    predictions.append({
                                        "label": label_name,
                                        "confidence": float(prob),
                                    })
                            phase3_data["multi_label"] = predictions  # Direct list, not wrapped
                        
                        # Add attention map info
                        if not p3_result.attention_map.is_empty():
                            phase3_data["attention_map"] = getattr(p3_result.attention_map, 'attention_map_base64', None)
                            phase3_data["attention_method"] = p3_result.attention_map.generation_method
            except Exception as p3_err:
                logger.debug(f"Phase 3 analysis skipped: {p3_err}")
        
        # Build Phase 3 response if available
        phase3_response = None
        if phase3_data and isinstance(phase3_data, dict):
            try:
                # Build attention info
                attention_info = None
                if phase3_data.get("attention_map"):
                    attention_info = Phase3AttentionInfo(
                        available=True,
                        map_uri=phase3_data.get("attention_map"),
                        method=phase3_data.get("attention_method", "grad_cam"),
                    )
                
                # Build regions info
                regions_list = None
                if phase3_data.get("regions"):
                    regions_list = [
                        Phase3RegionInfo(
                            region_id=i,
                            bbox=r.get("bbox"),
                            relevance_score=r.get("relevance_score", 0.0),
                            label=r.get("label"),
                        )
                        for i, r in enumerate(phase3_data["regions"])
                    ]
                
                # Build multi-label predictions
                multi_label_list = None
                if phase3_data.get("multi_label"):
                    multi_label_list = [
                        Phase3MultiLabelPrediction(
                            label=p.get("label", ""),
                            confidence=p.get("confidence", 0.0),
                        )
                        for p in phase3_data["multi_label"]
                    ]
                
                phase3_response = Phase3Response(
                    is_experimental=True,
                    executed=phase3_data.get("executed", False),
                    attention=attention_info,
                    regions=regions_list,
                    top_region_score=phase3_data.get("top_region_score"),
                    multi_label=multi_label_list,
                    processing_time_ms=phase3_data.get("processing_time_ms"),
                    error=phase3_data.get("error"),
                )
            except Exception as p3_err:
                logger.warning(f"Failed to build Phase 3 response: {p3_err}")
                phase3_response = Phase3Response(
                    is_experimental=True,
                    executed=False,
                    error=str(p3_err),
                )
        
        return PredictionResponse(
            status="success",
            request_id=ctx.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            prediction=prediction,
            inference=InferenceInfo(
                model_format=engine.model_format,
                device=engine.device,
                time_ms=round(inference_time, 2),
            ),
            feedback_id=feedback_id,
            phase3=phase3_response,
        )
    
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "INFERENCE_ERROR", "message": str(e)},
        )


@router.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(
    request_body: Base64ImageRequest,
    api_key: dict = Depends(check_rate_limit),
    engine=Depends(get_inference_engine),
    pipeline=Depends(get_validation_pipeline),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Predict from base64 encoded image.
    
    Alternative to file upload for mobile/JS clients.
    """
    # Decode base64
    try:
        image_bytes = base64.b64decode(request_body.image_data)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_BASE64", "message": "Invalid base64 image data"},
        )
    
    # Validate
    if pipeline:
        validation_result = pipeline.validate(image_bytes, "base64_upload")
        
        if not validation_result.valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "IMAGE_VALIDATION_FAILED",
                    "message": validation_result.message,
                    "failed_layer": validation_result.failed_layer,
                },
            )
    
    # Inference
    try:
        result = engine.predict(image_bytes)
        
        class_name = result.get("class_name", "")
        class_id = result.get("predicted_class", -1)
        confidence = result.get("confidence", 0.0)
        
        prediction = PredictionResult(
            **{
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),
            }
        )
        
        if request_body.include_probabilities:
            probs = result.get("probabilities", {})
            if isinstance(probs, dict):
                prediction.all_probabilities = {
                    k: round(v, 4) for k, v in probs.items()
                }
        
        # Run Phase 3 analysis if available
        phase3_response = None
        phase3_manager = get_phase3_manager()
        if phase3_manager and phase3_manager.is_operational():
            try:
                # Get tensors from result (added by pytorch_inference)
                image_tensor = result.get("image_tensor")
                logits_tensor = result.get("logits_tensor", result.get("logits"))
                model = getattr(engine, 'model', None)
                
                logger.info(f"Phase 3: model={model is not None}, image_tensor={image_tensor is not None}, logits={logits_tensor is not None}")
                
                if model is not None and image_tensor is not None:
                    p3_result = phase3_manager.run_inference(
                        model, image_tensor, logits_tensor, class_id
                    )
                    
                    # Log Phase 3 result details
                    logger.info(f"Phase 3 result: executed={p3_result.phase3_executed}, "
                               f"features_empty={p3_result.features.is_empty()}, "
                               f"regions_empty={p3_result.regions.is_empty()}, "
                               f"relevance_empty={p3_result.relevance_scores.is_empty()}, "
                               f"multi_label_empty={p3_result.multi_label.is_empty()}, "
                               f"attention_empty={p3_result.attention_map.is_empty()}, "
                               f"had_failure={p3_result.had_failure}, "
                               f"failure_msg={p3_result.failure_message}")
                    
                    # Always build phase3_data if Phase 3 executed
                    phase3_data = {
                        "executed": p3_result.phase3_executed,
                        "processing_time_ms": p3_result.execution_time_ms,
                        "error": p3_result.failure_message if p3_result.had_failure else None,
                    }
                    
                    # Add regions from p3_result.regions (not relevance_scores)
                    if not p3_result.regions.is_empty():
                        regions_list = []
                        for proposal in p3_result.regions.proposals:
                            regions_list.append({
                                "region_id": proposal.region_id,
                                "bbox": list(proposal.bbox) if proposal.bbox else None,
                                "relevance_score": proposal.confidence,
                                "label": None,
                            })
                        phase3_data["regions"] = regions_list
                        phase3_data["top_region_score"] = max(r["relevance_score"] for r in regions_list) if regions_list else None
                        
                    # Add relevance scores if available
                    if not p3_result.relevance_scores.is_empty():
                        # Update region relevance scores
                        if phase3_data.get("regions"):
                            for r in phase3_data["regions"]:
                                score = p3_result.relevance_scores.region_scores.get(r["region_id"], r["relevance_score"])
                                r["relevance_score"] = float(score)
                            phase3_data["top_region_score"] = max(r["relevance_score"] for r in phase3_data["regions"])
                    
                    # Add multi-label predictions - convert class IDs to labels
                    if not p3_result.multi_label.is_empty():
                        predictions = []
                        class_names = getattr(engine, 'class_names', None) or []
                        for class_id in p3_result.multi_label.predicted_labels:
                            label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                            conf = p3_result.multi_label.label_confidences.get(class_id, 0.0)
                            predictions.append({
                                "label": label_name,
                                "confidence": float(conf),
                            })
                        # Also add from class_probabilities if available
                        if not predictions and p3_result.multi_label.class_probabilities:
                            for cid, prob in sorted(p3_result.multi_label.class_probabilities.items(), 
                                                   key=lambda x: x[1], reverse=True)[:3]:
                                label_name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                                predictions.append({
                                    "label": label_name,
                                    "confidence": float(prob),
                                })
                        phase3_data["multi_label"] = predictions  # Direct list, not wrapped
                    
                    # Add attention map info
                    if not p3_result.attention_map.is_empty():
                        phase3_data["attention_map"] = getattr(p3_result.attention_map, 'attention_map_base64', None)
                        phase3_data["attention_method"] = p3_result.attention_map.generation_method
                    
                    # Build Phase 3 response
                    attention_info = None
                    if phase3_data.get("attention_map"):
                        attention_info = Phase3AttentionInfo(
                            available=True,
                            map_uri=phase3_data.get("attention_map"),
                            method=phase3_data.get("attention_method", "grad_cam"),
                        )
                    
                    regions_info = None
                    if phase3_data.get("regions"):
                        regions_info = [
                            Phase3RegionInfo(
                                region_id=i,
                                bbox=r.get("bbox"),
                                relevance_score=r.get("relevance_score", 0.0),
                                label=r.get("label"),
                            )
                            for i, r in enumerate(phase3_data["regions"])
                        ]
                    
                    multi_label_list = None
                    if phase3_data.get("multi_label"):
                        multi_label_list = [
                            Phase3MultiLabelPrediction(
                                label=p.get("label", ""),
                                confidence=p.get("confidence", 0.0),
                            )
                            for p in phase3_data["multi_label"]
                        ]
                    
                    phase3_response = Phase3Response(
                        is_experimental=True,
                        executed=phase3_data.get("executed", False),
                        attention=attention_info,
                        regions=regions_info,
                        top_region_score=phase3_data.get("top_region_score"),
                        multi_label=multi_label_list,
                        processing_time_ms=phase3_data.get("processing_time_ms"),
                        error=phase3_data.get("error"),
                    )
            except Exception as p3_err:
                logger.debug(f"Phase 3 analysis skipped: {p3_err}")
        
        return PredictionResponse(
            status="success",
            request_id=ctx.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            prediction=prediction,
            inference=InferenceInfo(
                model_format=engine.model_format,
                device=engine.device,
                time_ms=round(ctx.elapsed_ms, 2),
            ),
            phase3=phase3_response,
        )
    
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "INFERENCE_ERROR", "message": str(e)},
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    images: List[UploadFile] = File(..., description="Multiple images to classify"),
    api_key: dict = Depends(check_rate_limit),
    engine=Depends(get_inference_engine),
    pipeline=Depends(get_validation_pipeline),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Batch image prediction.
    
    Upload multiple images for classification.
    Maximum 10 images per request.
    """
    max_batch = 10
    
    if len(images) > max_batch:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BATCH_SIZE_EXCEEDED",
                "message": f"Maximum batch size is {max_batch} images",
            },
        )
    
    results = []
    successful = 0
    failed = 0
    
    for i, image in enumerate(images):
        image_bytes = await image.read()
        filename = image.filename or f"image_{i}"
        
        # Validate
        if pipeline:
            validation_result = pipeline.validate(image_bytes, filename)
            if not validation_result.valid:
                results.append(BatchResultItem(
                    index=i,
                    filename=filename,
                    status="rejected",
                    error=validation_result.message,
                ))
                failed += 1
                continue
        
        # Predict
        try:
            item_start = time.time()
            result = engine.predict(image_bytes)
            item_time = (time.time() - item_start) * 1000
            
            results.append(BatchResultItem(
                index=i,
                filename=filename,
                status="success",
                prediction=PredictionResult(
                    **{
                        "class": result.get("class_name", ""),
                        "class_id": result.get("predicted_class", -1),
                        "confidence": round(result.get("confidence", 0.0), 4),
                    }
                ),
                inference_time_ms=round(item_time, 2),
            ))
            successful += 1
        
        except Exception as e:
            results.append(BatchResultItem(
                index=i,
                filename=filename,
                status="error",
                error=str(e),
            ))
            failed += 1
    
    total_time = ctx.elapsed_ms
    
    return BatchPredictionResponse(
        status="success" if failed == 0 else "partial" if successful > 0 else "error",
        request_id=ctx.request_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        summary=BatchSummary(
            total=len(images),
            successful=successful,
            failed=failed,
            total_time_ms=round(total_time, 2),
            avg_time_ms=round(total_time / len(images), 2),
        ),
        results=results,
    )


# Admin routes
@admin_router.get("/stats")
async def get_stats(
    api_key: dict = Depends(require_admin),
    key_manager=Depends(get_api_key_manager),
):
    """
    Get API statistics.
    
    Admin only endpoint.
    """
    stats = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    
    if key_manager:
        stats["api_keys"] = key_manager.get_stats()
    
    return {"status": "success", "stats": stats}


@admin_router.post("/keys")
async def create_api_key(
    name: str = Query(..., description="Key name"),
    tier: str = Query("free", description="Access tier"),
    description: str = Query("", description="Key description"),
    api_key: dict = Depends(require_admin),
    key_manager=Depends(get_api_key_manager),
):
    """
    Create new API key.
    
    Admin only endpoint.
    """
    if key_manager is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "KEY_MANAGER_UNAVAILABLE", "message": "API key manager not available"},
        )
    
    new_key = key_manager.generate_key(name=name, tier=tier, description=description)
    
    return {
        "status": "success",
        "key": new_key,
    }


# Feedback routes
@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    api_key: dict = Depends(check_rate_limit),
):
    """
    Submit feedback on a prediction.
    
    Use the feedback_id from a prediction response to confirm if the
    prediction was correct or provide the correct classification.
    This helps improve the model over time.
    """
    feedback_mgr = get_feedback_manager()
    
    if feedback_mgr is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "FEEDBACK_UNAVAILABLE", "message": "Feedback system not available"},
        )
    
    result = feedback_mgr.submit_feedback(
        feedback_id=feedback.feedback_id,
        is_correct=feedback.is_correct,
        correct_class=feedback.correct_class,
        correct_class_id=feedback.correct_class_id,
        user_comment=feedback.user_comment,
        api_key_id=api_key.get("key_id") if api_key else None,
        device_info=feedback.device_info,
        app_version=feedback.app_version,
    )
    
    if result["status"] == "error":
        raise HTTPException(
            status_code=400,
            detail={
                "code": result.get("code", "FEEDBACK_ERROR"),
                "message": result.get("message", "Feedback submission failed"),
                "valid_classes": result.get("valid_classes"),
            },
        )
    
    return FeedbackResponse(
        status="success",
        request_id=None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        message=result["message"],
        feedback_id=result["feedback_id"],
        recorded=FeedbackRecorded(
            is_correct=result["recorded"]["is_correct"],
            original_prediction=result["recorded"]["original_prediction"],
            corrected_to=result["recorded"]["corrected_to"],
        ),
    )


@router.get("/feedback/classes", response_model=ClassesResponse)
async def get_feedback_classes(
    engine=Depends(get_optional_inference_engine),
):
    """
    Get valid class names for feedback corrections.
    
    Returns the list of valid class names that can be used when
    submitting a correction via the feedback endpoint.
    
    In addition to pest classes, special categories are also accepted:
    - "junk" or "unrelated": Image is not related to pest detection
    - "other": Some plant issue not in our classification list
    - "unknown": Cannot identify what the image shows
    """
    default_classes = [
        "Healthy",
        "Internode borer",
        "Pink borer",
        "Rat damage",
        "Stalk borer",
        "Top borer",
        "army worm",
        "mealy bug",
        "porcupine damage",
        "root borer",
        "termite",
    ]
    
    # Special feedback categories (not pest classes)
    special_categories = ["junk", "unrelated", "other", "unknown"]
    
    classes = engine.class_names if engine else default_classes
    
    return ClassesResponse(
        status="success",
        request_id=None,
        timestamp=None,
        num_classes=len(classes),
        classes=[
            ClassInfo(id=i, name=name)
            for i, name in enumerate(classes)
        ],
        special_categories=special_categories,
    )


@admin_router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    api_key: dict = Depends(require_admin),
):
    """
    Get feedback statistics.
    
    Admin only endpoint. Returns statistics about user feedback including
    accuracy from user reports and common misclassifications.
    """
    feedback_mgr = get_feedback_manager()
    
    if feedback_mgr is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "FEEDBACK_UNAVAILABLE", "message": "Feedback system not available"},
        )
    
    stats = feedback_mgr.get_statistics()
    
    return FeedbackStatsResponse(
        status="success",
        request_id=None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_predictions=stats["total_predictions"],
        feedback_received=stats["feedback_received"],
        correct_predictions=stats["correct_predictions"],
        incorrect_predictions=stats["incorrect_predictions"],
        accuracy_from_feedback=stats["accuracy_from_feedback"],
        pending_feedbacks=stats["pending_feedbacks"],
        corrections_by_class=stats["corrections_by_class"],
        junk_reports=stats.get("junk_reports", 0),
        special_categories=stats.get("special_categories", {}),
    )


# User Management Admin Endpoints
@admin_router.get("/users")
async def list_users(
    api_key: dict = Depends(require_admin),
):
    """
    List all users and their statistics.
    
    Admin only endpoint.
    """
    user_tracker = get_user_tracker()
    
    if user_tracker is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "USER_TRACKER_UNAVAILABLE", "message": "User tracker not available"},
        )
    
    users = user_tracker.get_all_users()
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_users": len(users),
        "users": users,
    }


@admin_router.get("/users/flagged")
async def list_flagged_users(
    api_key: dict = Depends(require_admin),
):
    """
    List all flagged (suspicious) users.
    
    Admin only endpoint.
    """
    user_tracker = get_user_tracker()
    
    if user_tracker is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "USER_TRACKER_UNAVAILABLE", "message": "User tracker not available"},
        )
    
    flagged = user_tracker.get_flagged_users()
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "flagged_count": len(flagged),
        "flagged_users": flagged,
    }


@admin_router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    api_key: dict = Depends(require_admin),
):
    """
    Get detailed statistics for a specific user.
    
    Admin only endpoint.
    """
    user_tracker = get_user_tracker()
    
    if user_tracker is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "USER_TRACKER_UNAVAILABLE", "message": "User tracker not available"},
        )
    
    user_stats = user_tracker.get_user_stats(user_id)
    
    if user_stats is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "USER_NOT_FOUND", "message": f"User {user_id} not found"},
        )
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user": user_stats,
    }


@admin_router.post("/users/{user_id}/unflag")
async def unflag_user(
    user_id: str,
    note: str = Query("", description="Admin note for unflagging"),
    api_key: dict = Depends(require_admin),
):
    """
    Unflag a user (remove suspicious status).
    
    Admin only endpoint.
    """
    user_tracker = get_user_tracker()
    
    if user_tracker is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "USER_TRACKER_UNAVAILABLE", "message": "User tracker not available"},
        )
    
    success = user_tracker.unflag_user(user_id, admin_note=note)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail={"code": "USER_NOT_FOUND", "message": f"User {user_id} not found"},
        )
    
    return {
        "status": "success",
        "message": f"User {user_id} has been unflagged",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@admin_router.get("/data/stats")
async def get_data_collection_stats(
    api_key: dict = Depends(require_admin),
):
    """
    Get data collection statistics.
    
    Admin only endpoint. Shows collected images stats.
    """
    data_collector = get_data_collector()
    
    if data_collector is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATA_COLLECTOR_UNAVAILABLE", "message": "Data collector not available"},
        )
    
    stats = data_collector.get_statistics()
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection_stats": stats,
    }


@admin_router.get("/data/export")
async def export_data(
    format: str = Query("csv", description="Export format: csv or excel"),
    api_key: dict = Depends(require_admin),
):
    """
    Export collected data to CSV/Excel.
    
    Admin only endpoint.
    """
    user_tracker = get_user_tracker()
    data_collector = get_data_collector()
    
    exports = {}
    
    if user_tracker:
        try:
            if format == "excel":
                exports["users"] = user_tracker.export_to_excel()
            else:
                exports["users"] = user_tracker.export_to_csv()
        except Exception as e:
            exports["users_error"] = str(e)
    
    if data_collector:
        try:
            exports["images"] = data_collector.export_to_csv()
        except Exception as e:
            exports["images_error"] = str(e)
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exported_files": exports,
    }


@admin_router.get("/data/training")
async def get_training_data(
    include_correct: bool = Query(True, description="Include confirmed correct predictions"),
    include_corrected: bool = Query(True, description="Include user-corrected predictions"),
    only_trusted: bool = Query(True, description="Only include trusted user submissions"),
    api_key: dict = Depends(require_admin),
):
    """
    Get data suitable for model retraining.
    
    Admin only endpoint. Returns image paths and labels.
    """
    data_collector = get_data_collector()
    
    if data_collector is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATA_COLLECTOR_UNAVAILABLE", "message": "Data collector not available"},
        )
    
    training_data = data_collector.get_training_data(
        include_correct=include_correct,
        include_corrected=include_corrected,
        only_trusted=only_trusted,
    )
    
    return {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(training_data),
        "data": training_data,
    }


# ============================================================
# Model Retraining Endpoints
# ============================================================

@admin_router.get("/retrain/status")
async def get_retrain_status(
    api_key: dict = Depends(require_admin),
):
    """
    Get model retraining status.
    
    Shows:
    - Current training status (is_training, progress)
    - Model version and total retrains count
    - Image counts per class
    - Whether thresholds are met for auto-retrain
    - Last training date
    """
    try:
        from ..training.retrain_manager import get_retrain_manager
        
        retrain_manager = get_retrain_manager(
            model_path="D:/KnowledgeDistillation/student_model_rotation_robust.pt",
            feedback_dir="./feedback_data/images",
        )
        
        status = retrain_manager.get_status()
        
        return {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": {
                "version": status.current_version_string,
                "total_fine_tunes": status.total_fine_tunes,
                "total_comprehensive": status.total_comprehensive,
            },
            "retraining": {
                "is_training": status.is_training,
                "progress": round(status.training_progress * 100, 1),
                "current_epoch": status.current_epoch,
                "total_epochs": status.total_epochs,
                "last_trained": status.last_trained,
                "last_backup": status.last_backup_path,
                "error": status.error,
            },
            "feedback_images": {
                "total": status.total_feedback_images,
                "junk_images": status.junk_images,
                "per_class": status.images_per_class,
                "classes_with_images": status.classes_with_images,
            },
            "thresholds": {
                "per_class": status.threshold_per_class,
                "total": status.threshold_total,
                "min_classes": status.threshold_min_classes,
                "ready_to_retrain": status.ready_to_retrain,
                "blocked_reason": status.retrain_blocked_reason,  # Why retraining is blocked
            },
        }
        
    except Exception as e:
        logger.exception("Error getting retrain status")
        raise HTTPException(
            status_code=500,
            detail={"code": "RETRAIN_STATUS_ERROR", "message": str(e)},
        )


@admin_router.get("/retrain/history")
async def get_retrain_history(
    api_key: dict = Depends(require_admin),
):
    """
    Get complete model retraining history.
    
    Returns a list of all past retraining runs with:
    - Version number
    - Timestamp
    - Images used
    - Duration
    - Success/failure status
    """
    try:
        from ..training.retrain_manager import get_retrain_manager
        
        retrain_manager = get_retrain_manager(
            model_path="D:/KnowledgeDistillation/student_model_rotation_robust.pt",
            feedback_dir="./feedback_data/images",
        )
        
        history = retrain_manager.get_training_history()
        
        return {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_entries": len(history),
            "history": history,
        }
        
    except Exception as e:
        logger.exception("Error getting retrain history")
        raise HTTPException(
            status_code=500,
            detail={"code": "RETRAIN_HISTORY_ERROR", "message": str(e)},
        )


@admin_router.post("/retrain/trigger")
async def trigger_retraining(
    force: bool = Query(False, description="Force retraining even if thresholds not met"),
    api_key: dict = Depends(require_admin),
):
    """
    Manually trigger model retraining.
    
    This will:
    1. Backup the current model
    2. Fine-tune with feedback images
    3. Update the deployment model
    
    Use force=true to retrain even if thresholds aren't met.
    """
    try:
        from ..training.retrain_manager import get_retrain_manager
        
        retrain_manager = get_retrain_manager(
            model_path="D:/KnowledgeDistillation/student_model_rotation_robust.pt",
            feedback_dir="./feedback_data/images",
        )
        
        status = retrain_manager.get_status()
        
        if status.is_training:
            return {
                "status": "info",
                "message": "Retraining already in progress",
                "progress": round(status.training_progress * 100, 1),
            }
        
        if not force and not status.ready_to_retrain:
            return {
                "status": "info",
                "message": "Thresholds not met for retraining",
                "feedback_images": status.total_feedback_images,
                "threshold_total": status.threshold_total,
                "hint": "Use force=true to retrain anyway",
            }
        
        # Start retraining
        started = retrain_manager.start_retraining(force=force)
        
        if started:
            return {
                "status": "success",
                "message": "Retraining started in background",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        else:
            return {
                "status": "error",
                "message": "Failed to start retraining",
            }
            
    except Exception as e:
        logger.exception("Error triggering retraining")
        raise HTTPException(
            status_code=500,
            detail={"code": "RETRAIN_TRIGGER_ERROR", "message": str(e)},
        )


@router.get("/retrain/status")
@router.get("/retraining/status")  # Alias for compatibility
async def get_retrain_status_public():
    """
    Get model retraining status (public endpoint).
    
    Shows basic training status without admin details.
    Available at both /retrain/status and /retraining/status.
    """
    try:
        from ..training.retrain_manager import get_retrain_manager
        
        retrain_manager = get_retrain_manager(
            model_path="D:/KnowledgeDistillation/student_model_rotation_robust.pt",
            feedback_dir="./feedback_data/images",
        )
        
        status = retrain_manager.get_status()
        
        return {
            "status": "success",
            "is_training": status.is_training,
            "progress": round(status.training_progress * 100, 1) if status.is_training else None,
            "last_trained": status.last_trained,
            "feedback_images_collected": status.total_feedback_images,
            "ready_for_improvement": status.ready_to_retrain,
        }
        
    except Exception as e:
        return {
            "status": "unavailable",
            "message": "Retraining status not available",
        }

