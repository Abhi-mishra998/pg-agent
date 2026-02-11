#!/usr/bin/env python3
"""
pg-agent Review API

FastAPI-based REST API for the recommendation review system.
Provides endpoints for:
- Review card CRUD operations
- Action approval/rejection workflow
- Audit trail tracking
- UI serving (static files)

Static file serving is optional - directories are mounted only if they exist.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional

from recommendations.review_schema import ReviewCard
from recommendations.approval_store import save_card, load_card


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="pg-agent Recommendation Review API",
        description="REST API for PostgreSQL agent recommendation review and approval workflow"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files with existence checks
    _mount_static_files(app)
    
    return app


def _mount_static_files(app: FastAPI) -> None:
    """
    Mount static files for UI with graceful fallback if directories don't exist.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Mount /static for review UI assets
    static_dir = Path("recommendations/static")
    if static_dir.exists() and static_dir.is_dir():
        try:
            app.mount(
                "/static",
                StaticFiles(directory=str(static_dir), html=True),
                name="static"
            )
            logger.info(f"Static files mounted from {static_dir}")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
    
    # Mount /docs/ui for UI mockup
    ui_dir = Path("docs/ui_mockup")
    if ui_dir.exists() and ui_dir.is_dir():
        try:
            app.mount(
                "/docs/ui",
                StaticFiles(directory=str(ui_dir), html=True),
                name="ui_mockup"
            )
            logger.info(f"UI mockup mounted from {ui_dir}")
        except Exception as e:
            logger.warning(f"Could not mount UI mockup: {e}")
    else:
        logger.warning(f"UI mockup directory not found: {ui_dir}")


# Create the app instance
app = create_app()


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/")
async def root():
    """
    Serve the UI mockup index page.
    
    Falls back to JSON message if index.html doesn't exist.
    """
    index_path = Path("docs/ui_mockup/index.html")
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "pg-agent Review API",
        "status": "running",
        "endpoints": [
            "GET /health - Health check",
            "POST /cards - Create review card",
            "GET /cards/{card_id} - Get review card",
            "POST /cards/{card_id}/actions/{action_id}/approve - Approve action",
            "POST /cards/{card_id}/actions/{action_id}/reject - Reject action",
            "POST /cards/{card_id}/actions/{action_id}/execute - Execute action",
        ]
    }


@app.get("/ui")
async def ui():
    """Alternative UI endpoint."""
    return root()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/cards")
async def create_card(card: Dict[str, Any]):
    """
    Create a new review card.
    
    Args:
        card: Review card data as dictionary
        
    Returns:
        Created card ID
    """
    try:
        rc = ReviewCard.from_dict(card)
        save_card(rc.to_dict())
        return {"status": "created", "card_id": rc.card_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/cards/{card_id}")
async def get_card(card_id: str):
    """
    Get a review card by ID.
    
    Args:
        card_id: The card ID to retrieve
        
    Returns:
        Review card data
    """
    data = load_card(card_id)
    if not data:
        raise HTTPException(status_code=404, detail="card not found")
    return data


@app.post("/cards/{card_id}/actions/{action_id}/approve")
async def approve_action(card_id: str, action_id: str, payload: Dict[str, str]):
    """
    Approve an action on a review card.
    
    Args:
        card_id: The card ID
        action_id: The action ID to approve
        payload: Must include 'approver' and optionally 'comments'
        
    Returns:
        Approval confirmation
    """
    card_data = load_card(card_id)
    if not card_data:
        raise HTTPException(status_code=404, detail="card not found")

    # Load the review card
    card = ReviewCard.from_dict(card_data)
    approver = payload.get("approver")
    comments = payload.get("comments", "")
    
    if not approver:
        raise HTTPException(status_code=400, detail="approver required")

    ok = card.log_approval(action_id, approver, comments)
    if not ok:
        raise HTTPException(status_code=400, detail="approval failed or action not configured for approval")

    # Persist the updated card
    save_card(card.to_dict())
    return {"status": "approved", "card_id": card.card_id, "action_id": action_id}


@app.post("/cards/{card_id}/actions/{action_id}/reject")
async def reject_action(card_id: str, action_id: str, payload: Dict[str, str]):
    """
    Reject an action on a review card.
    
    Args:
        card_id: The card ID
        action_id: The action ID to reject
        payload: Must include 'rejector' and 'reason'
        
    Returns:
        Rejection confirmation
    """
    card_data = load_card(card_id)
    if not card_data:
        raise HTTPException(status_code=404, detail="card not found")

    card = ReviewCard.from_dict(card_data)
    rejector = payload.get("rejector")
    reason = payload.get("reason", "")
    
    if not rejector or not reason:
        raise HTTPException(status_code=400, detail="rejector and reason required")

    ok = card.log_rejection(action_id, rejector, reason)
    if not ok:
        raise HTTPException(status_code=400, detail="rejection failed or action not configured for approval")

    save_card(card.to_dict())
    return {"status": "rejected", "card_id": card.card_id, "action_id": action_id}


@app.post("/cards/{card_id}/actions/{action_id}/execute")
async def execute_action(card_id: str, action_id: str, payload: Dict[str, str]):
    """
    Execute an approved action.
    
    Args:
        card_id: The card ID
        action_id: The action ID to execute
        payload: Must include 'executor'
        
    Returns:
        Execution confirmation
    """
    card_data = load_card(card_id)
    if not card_data:
        raise HTTPException(status_code=404, detail="card not found")

    card = ReviewCard.from_dict(card_data)
    executor = payload.get("executor")
    
    if not executor:
        raise HTTPException(status_code=400, detail="executor required")

    action = card.get_action_by_id(action_id)
    if not action:
        raise HTTPException(status_code=404, detail="action not found")

    # Check safety gates
    if not action.is_safe_to_execute():
        raise HTTPException(
            status_code=403, 
            detail="action not approved or lacks rollback plan"
        )

    ok = card.log_execution(action_id, executor)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to log execution")

    save_card(card.to_dict())
    return {"status": "executed", "card_id": card.card_id, "action_id": action_id}


# =====================================================================
# Run with: uvicorn recommendations.api:app --reload --host 0.0.0.0 --port 8000
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

