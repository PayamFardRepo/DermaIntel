"""
AI Consultation Agent Router - Conversational AI Dermatologist

This router provides an agent-driven conversational experience where the AI:
1. Analyzes uploaded skin images
2. Asks targeted follow-up questions
3. Provides visual explanations with image annotations
4. Generates personalized risk assessments
"""

from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import json
import os
import uuid
import base64
from pathlib import Path

from database import get_db, User, AnalysisHistory
from auth import get_current_active_user

router = APIRouter(tags=["AI Consultation"])

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Consultation session storage (in production, use Redis or database)
consultation_sessions: Dict[str, Dict[str, Any]] = {}

# Upload directory
UPLOAD_DIR = Path("uploads/consultations")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# System prompt for the consultation agent
CONSULTATION_AGENT_PROMPT = """You are an empathetic AI dermatology consultation assistant. You are having a conversation with a patient who has uploaded an image of a skin concern.

YOUR ROLE:
- Guide the patient through a thorough but friendly consultation
- Ask targeted questions based on what you observe in the image
- Be warm, understanding, and reassuring while being medically accurate
- Explain things in plain language that anyone can understand

CONVERSATION FLOW:
1. First, acknowledge the patient's concern and describe what you observe in the image
2. Ask 2-3 relevant follow-up questions (one at a time)
3. After gathering information, provide a comprehensive assessment
4. Suggest appropriate next steps

QUESTION TYPES TO ASK (pick the most relevant):
- Duration: "How long have you had this spot?"
- Changes: "Has it changed in size, color, or shape recently?"
- Symptoms: "Does it itch, hurt, bleed, or cause any discomfort?"
- History: "Have you had similar spots before?"
- Sun exposure: "Do you spend a lot of time in the sun? Have you had sunburns?"
- Family history: "Does anyone in your family have a history of skin cancer?"
- Location sensitivity: "Is this area frequently exposed to sun or friction?"

RESPONSE FORMAT:
- Use a warm, conversational tone
- Keep responses concise (2-4 paragraphs max)
- Use simple language, avoiding complex medical jargon
- When asking questions, ask only ONE question at a time
- Include appropriate reassurance when possible

IMPORTANT GUIDELINES:
- Never provide a definitive diagnosis - always recommend professional consultation for concerning findings
- Acknowledge the patient's emotions and concerns
- Be clear that this is a screening tool, not a replacement for a dermatologist
- For concerning findings, encourage timely medical follow-up without causing panic

ALWAYS END MESSAGES WITH:
- If asking a question: End with the question clearly stated
- If providing final assessment: Include "next steps" recommendations
"""

VISUAL_ANALYSIS_PROMPT = """Analyze this skin image and identify key features that should be highlighted for the patient.

Return your analysis as JSON with this structure:
{
    "description": "Brief description of what you observe",
    "features": [
        {
            "type": "concern|neutral|positive",
            "label": "Brief label (e.g., 'Irregular Border', 'Uniform Color')",
            "description": "One sentence explanation",
            "region": "general location (e.g., 'upper left', 'center', 'throughout')"
        }
    ],
    "initial_impression": "Your initial impression of what this might be (be appropriately cautious)",
    "confidence_level": "low|medium|high",
    "urgency": "routine|moderate|high",
    "suggested_questions": ["Question 1", "Question 2", "Question 3"]
}

Focus on features relevant to the ABCDE criteria for lesions:
- Asymmetry
- Border irregularity
- Color variation
- Diameter
- Evolution/changes

Be medically accurate but not alarmist. Most skin findings are benign."""


def get_openai_client():
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class ConsultationSession:
    """Manages the state of a consultation session."""

    def __init__(self, session_id: str, user_id: int):
        self.session_id = session_id
        self.user_id = user_id
        self.image_path: Optional[str] = None
        self.image_analysis: Optional[Dict] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.patient_responses: Dict[str, str] = {}
        self.questions_asked: List[str] = []
        self.stage: str = "initial"  # initial, gathering_info, assessment
        self.created_at = datetime.utcnow()
        self.clinical_context: Dict[str, Any] = {}

    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "image_path": self.image_path,
            "image_analysis": self.image_analysis,
            "conversation_history": self.conversation_history,
            "patient_responses": self.patient_responses,
            "questions_asked": self.questions_asked,
            "stage": self.stage,
            "created_at": self.created_at.isoformat(),
            "clinical_context": self.clinical_context
        }


@router.get("/ai-consultation/status")
async def get_consultation_status():
    """Check if AI consultation is available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_api_key = bool(api_key)

    return {
        "available": OPENAI_AVAILABLE and has_api_key,
        "openai_installed": OPENAI_AVAILABLE,
        "api_key_configured": has_api_key,
        "features": [
            "conversational_consultation",
            "image_analysis",
            "visual_annotations",
            "personalized_questions",
            "risk_assessment"
        ] if has_api_key else []
    }


@router.post("/ai-consultation/start")
async def start_consultation(
    image: UploadFile = File(...),
    clinical_context: Optional[str] = Form(None),  # JSON string with age, skin_type, etc.
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Start a new AI consultation session with an uploaded image.

    Returns:
    - session_id: Unique ID for this consultation
    - initial_message: AI's first response with observations and first question
    - image_analysis: Visual analysis with features to highlight
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Save uploaded image
    file_extension = Path(image.filename).suffix or ".jpg"
    image_filename = f"{session_id}{file_extension}"
    image_path = UPLOAD_DIR / image_filename

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Create session
    session = ConsultationSession(session_id, current_user.id)
    session.image_path = str(image_path)

    # Parse clinical context if provided
    if clinical_context:
        try:
            session.clinical_context = json.loads(clinical_context)
        except json.JSONDecodeError:
            pass

    # Encode image for vision API
    base64_image = encode_image_to_base64(str(image_path))

    # Step 1: Analyze the image
    try:
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": VISUAL_ANALYSIS_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this skin image and provide your assessment in JSON format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )

        # Parse the analysis JSON
        analysis_text = analysis_response.choices[0].message.content
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in analysis_text:
                json_start = analysis_text.find("```json") + 7
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end].strip()
            elif "```" in analysis_text:
                json_start = analysis_text.find("```") + 3
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end].strip()

            session.image_analysis = json.loads(analysis_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure
            session.image_analysis = {
                "description": analysis_text,
                "features": [],
                "initial_impression": "Unable to parse detailed analysis",
                "confidence_level": "low",
                "urgency": "routine",
                "suggested_questions": [
                    "How long have you had this spot?",
                    "Has it changed recently?",
                    "Does it cause any symptoms?"
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

    # Step 2: Generate initial conversational message
    context_info = ""
    if session.clinical_context:
        if session.clinical_context.get("age"):
            context_info += f"Patient age: {session.clinical_context['age']}\n"
        if session.clinical_context.get("skin_type"):
            context_info += f"Skin type: {session.clinical_context['skin_type']}\n"

    initial_prompt = f"""The patient has uploaded an image of a skin concern. Here is your analysis:

{json.dumps(session.image_analysis, indent=2)}

{context_info}

Please provide a warm, empathetic opening message that:
1. Acknowledges their concern
2. Describes what you observe in simple terms (2-3 sentences)
3. Asks your FIRST follow-up question (choose the most relevant one from your suggested questions)

Remember: Be warm and conversational, not clinical. Ask only ONE question."""

    try:
        conversation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CONSULTATION_AGENT_PROMPT},
                {"role": "user", "content": initial_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        initial_message = conversation_response.choices[0].message.content
        session.add_message("assistant", initial_message)
        session.stage = "gathering_info"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

    # Store session
    consultation_sessions[session_id] = session

    return {
        "session_id": session_id,
        "initial_message": initial_message,
        "image_analysis": session.image_analysis,
        "stage": session.stage
    }


@router.post("/ai-consultation/respond")
async def respond_to_consultation(
    session_id: str = Form(...),
    message: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Continue the consultation conversation.

    The AI will:
    - Process the patient's response
    - Ask follow-up questions or provide assessment
    - Track conversation state
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Get session
    session = consultation_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Consultation session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")

    # Add user message to history
    session.add_message("user", message)

    # Build conversation context
    messages = [{"role": "system", "content": CONSULTATION_AGENT_PROMPT}]

    # Add image analysis context
    messages.append({
        "role": "system",
        "content": f"Image analysis context:\n{json.dumps(session.image_analysis, indent=2)}"
    })

    # Add clinical context if available
    if session.clinical_context:
        context_parts = []
        if session.clinical_context.get("age"):
            context_parts.append(f"Age: {session.clinical_context['age']}")
        if session.clinical_context.get("skin_type"):
            context_parts.append(f"Skin Type: {session.clinical_context['skin_type']}")
        if session.clinical_context.get("family_history_melanoma"):
            context_parts.append("Family history of melanoma: Yes")
        if session.clinical_context.get("family_history_skin_cancer"):
            context_parts.append("Family history of skin cancer: Yes")

        if context_parts:
            messages.append({
                "role": "system",
                "content": f"Patient clinical context:\n" + "\n".join(context_parts)
            })

    # Add conversation history
    for msg in session.conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Determine conversation stage
    questions_asked = len([m for m in session.conversation_history if m["role"] == "assistant"])

    if questions_asked >= 3:
        # Time for final assessment
        messages.append({
            "role": "system",
            "content": """You have gathered enough information. Now provide a comprehensive, personalized assessment that includes:
1. Summary of what you observed and learned
2. Your assessment (be appropriately cautious)
3. Personalized risk factors based on their responses
4. Clear next steps recommendation
5. Any reassurance that's appropriate

Remember to be warm and supportive. End with a clear recommendation for next steps."""
        })
        session.stage = "assessment"
    else:
        # Continue gathering information
        messages.append({
            "role": "system",
            "content": f"You have asked {questions_asked} question(s). Ask one more relevant follow-up question based on their response. Remember to acknowledge what they shared before asking the next question."
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )

        ai_response = response.choices[0].message.content
        session.add_message("assistant", ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

    return {
        "session_id": session_id,
        "response": ai_response,
        "stage": session.stage,
        "questions_asked": questions_asked + 1
    }


@router.post("/ai-consultation/get-visual-explanation")
async def get_visual_explanation(
    session_id: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get visual annotations/explanation for the uploaded image.
    Returns regions of interest with explanations.
    """
    session = consultation_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if not session.image_analysis:
        raise HTTPException(status_code=400, detail="No image analysis available")

    # Return the visual analysis features
    return {
        "session_id": session_id,
        "features": session.image_analysis.get("features", []),
        "description": session.image_analysis.get("description", ""),
        "overall_assessment": {
            "impression": session.image_analysis.get("initial_impression", ""),
            "confidence": session.image_analysis.get("confidence_level", "low"),
            "urgency": session.image_analysis.get("urgency", "routine")
        }
    }


@router.get("/ai-consultation/session/{session_id}")
async def get_consultation_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the current state of a consultation session."""
    session = consultation_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return session.to_dict()


@router.post("/ai-consultation/generate-summary")
async def generate_consultation_summary(
    session_id: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate a summary of the consultation that can be shared with a dermatologist.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    session = consultation_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Build summary prompt
    conversation_text = "\n".join([
        f"{'Patient' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in session.conversation_history
    ])

    prompt = f"""Based on this AI consultation, generate a concise clinical summary suitable for sharing with a dermatologist.

IMAGE ANALYSIS:
{json.dumps(session.image_analysis, indent=2)}

CLINICAL CONTEXT:
{json.dumps(session.clinical_context, indent=2)}

CONVERSATION:
{conversation_text}

Generate a professional summary with:
1. Chief Complaint
2. History of Present Illness (from patient responses)
3. AI Image Analysis Findings
4. Patient-Reported Information
5. AI Assessment & Recommendations
6. Suggested Follow-up

Keep it concise and clinically relevant."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical documentation assistant. Generate clear, professional clinical summaries."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        summary = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

    return {
        "session_id": session_id,
        "summary": summary,
        "generated_at": datetime.utcnow().isoformat(),
        "disclaimer": "This AI-generated summary is for informational purposes only and should be reviewed by a healthcare professional."
    }


@router.delete("/ai-consultation/session/{session_id}")
async def end_consultation_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """End and clean up a consultation session."""
    session = consultation_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Clean up image file
    if session.image_path and os.path.exists(session.image_path):
        try:
            os.remove(session.image_path)
        except:
            pass

    # Remove session
    del consultation_sessions[session_id]

    return {"status": "success", "message": "Consultation session ended"}
