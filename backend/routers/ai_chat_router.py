"""
AI Chat Router - LLM Integration for Natural Language Interaction

Provides conversational AI capabilities for:
- Explaining diagnoses in plain language
- Answering questions about skin conditions
- Patient education
- Treatment information
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
import asyncio

from database import get_db, User, AnalysisHistory
from auth import get_current_active_user

router = APIRouter(tags=["AI Chat"])

# Try to import OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. AI chat will be limited.")

# System prompt for medical context
SYSTEM_PROMPT = """You are a knowledgeable dermatology assistant integrated into a skin analysis application.
Your role is to help patients understand their skin analysis results in plain, accessible language.

IMPORTANT GUIDELINES:
1. Always be empathetic and reassuring while being accurate
2. Explain medical terms in simple language
3. Never provide definitive diagnoses - always recommend professional consultation
4. Be clear that AI analysis is a screening tool, not a replacement for dermatologist evaluation
5. For concerning findings, encourage timely medical follow-up without causing panic
6. Provide actionable advice when appropriate (sun protection, monitoring changes, etc.)
7. Answer questions about skin conditions, treatments, and prevention
8. If asked about something outside dermatology, politely redirect to skin-related topics

DISCLAIMERS TO INCLUDE WHEN RELEVANT:
- "This AI analysis is for informational purposes only and does not constitute medical advice."
- "Please consult a board-certified dermatologist for proper diagnosis and treatment."
"""

def get_openai_client():
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def get_async_openai_client():
    """Get async OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return AsyncOpenAI(api_key=api_key)

def format_analysis_context(analysis: AnalysisHistory) -> str:
    """Format analysis results into context for the LLM."""
    context_parts = []

    context_parts.append(f"Analysis Date: {analysis.created_at.strftime('%Y-%m-%d') if analysis.created_at else 'Unknown'}")

    if analysis.predicted_class:
        context_parts.append(f"Primary Diagnosis: {analysis.predicted_class}")

    if analysis.lesion_confidence:
        context_parts.append(f"Confidence: {analysis.lesion_confidence * 100:.1f}%")

    if analysis.risk_level:
        context_parts.append(f"Risk Level: {analysis.risk_level}")

    if analysis.inflammatory_condition:
        context_parts.append(f"Inflammatory Condition: {analysis.inflammatory_condition}")

    if analysis.infectious_disease:
        context_parts.append(f"Infectious Finding: {analysis.infectious_disease}")

    if analysis.burn_severity:
        context_parts.append(f"Burn Assessment: {analysis.burn_severity}")

    if analysis.body_location:
        context_parts.append(f"Location: {analysis.body_location}")

    if analysis.differential_diagnoses:
        try:
            differentials = analysis.differential_diagnoses if isinstance(analysis.differential_diagnoses, list) else json.loads(analysis.differential_diagnoses)
            if differentials:
                diff_str = ", ".join([d.get('condition', d) if isinstance(d, dict) else str(d) for d in differentials[:3]])
                context_parts.append(f"Differential Diagnoses: {diff_str}")
        except:
            pass

    if analysis.treatment_recommendations:
        try:
            treatments = analysis.treatment_recommendations if isinstance(analysis.treatment_recommendations, list) else json.loads(analysis.treatment_recommendations)
            if treatments:
                context_parts.append(f"Recommended Actions: {', '.join(treatments[:3])}")
        except:
            pass

    if analysis.risk_recommendation:
        context_parts.append(f"Risk Recommendation: {analysis.risk_recommendation}")

    # Multimodal data if available
    if analysis.multimodal_enabled:
        context_parts.append("Note: This analysis used multimodal data (image + patient history/labs)")

    return "\n".join(context_parts)


@router.get("/ai-chat/status")
async def get_ai_chat_status():
    """Check if AI chat is available and configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_api_key = bool(api_key)

    return {
        "available": OPENAI_AVAILABLE and has_api_key,
        "openai_installed": OPENAI_AVAILABLE,
        "api_key_configured": has_api_key,
        "model": "gpt-4o" if has_api_key else None,
        "features": [
            "diagnosis_explanation",
            "condition_questions",
            "treatment_info",
            "patient_education"
        ] if has_api_key else []
    }


@router.post("/ai-chat")
async def chat_with_ai(
    message: str = Form(...),
    analysis_id: Optional[int] = Form(None),
    conversation_history: Optional[str] = Form(None),  # JSON string of previous messages
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Chat with AI about skin analysis results.

    Parameters:
    - message: User's question or message
    - analysis_id: Optional ID of analysis to discuss
    - conversation_history: JSON string of previous messages for context
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI package not installed. Please install with: pip install openai"
        )

    client = get_openai_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )

    # Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add analysis context if provided
    if analysis_id:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if analysis:
            context = format_analysis_context(analysis)
            messages.append({
                "role": "system",
                "content": f"The user is asking about the following skin analysis results:\n\n{context}\n\nPlease help them understand these results."
            })

    # Add conversation history if provided
    if conversation_history:
        try:
            history = json.loads(conversation_history)
            for msg in history[-10:]:  # Keep last 10 messages for context
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        except json.JSONDecodeError:
            pass

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        return {
            "response": assistant_message,
            "model": "gpt-4o",
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI chat error: {str(e)}"
        )


@router.post("/ai-chat/stream")
async def chat_with_ai_stream(
    message: str = Form(...),
    analysis_id: Optional[int] = Form(None),
    conversation_history: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Chat with AI with streaming response.
    Returns Server-Sent Events for real-time response display.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI package not installed"
        )

    client = get_async_openai_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    # Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add analysis context if provided
    if analysis_id:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if analysis:
            context = format_analysis_context(analysis)
            messages.append({
                "role": "system",
                "content": f"The user is asking about the following skin analysis results:\n\n{context}\n\nPlease help them understand these results."
            })

    # Add conversation history
    if conversation_history:
        try:
            history = json.loads(conversation_history)
            for msg in history[-10:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        except json.JSONDecodeError:
            pass

    messages.append({"role": "user", "content": message})

    async def generate():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/ai-chat/explain-condition")
async def explain_condition(
    condition: str = Form(...),
    severity: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a plain-language explanation of a skin condition.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    prompt = f"""Please explain the skin condition "{condition}" in simple, patient-friendly terms.

Include:
1. What it is (brief description)
2. Common causes
3. Typical symptoms
4. General treatment approaches
5. When to see a doctor

{"Severity context: " + severity if severity else ""}

Keep the explanation concise but informative. Use bullet points where helpful."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )

        return {
            "condition": condition,
            "explanation": response.choices[0].message.content,
            "disclaimer": "This information is for educational purposes only. Please consult a healthcare provider for medical advice."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ai-chat/summarize-analysis")
async def summarize_analysis(
    analysis_id: int = Form(...),
    language_level: str = Form("simple"),  # "simple", "detailed", "medical"
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a plain-language summary of an analysis.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    context = format_analysis_context(analysis)

    level_instructions = {
        "simple": "Use very simple language that anyone can understand. Avoid medical jargon.",
        "detailed": "Provide a thorough explanation with some medical context, but keep it accessible.",
        "medical": "Use appropriate medical terminology while still being clear."
    }

    prompt = f"""Please summarize this skin analysis in plain language:

{context}

{level_instructions.get(language_level, level_instructions["simple"])}

Structure your summary as:
1. What was found (main finding)
2. What this means for the patient
3. Recommended next steps
4. Any reassuring points if appropriate"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.5
        )

        return {
            "analysis_id": analysis_id,
            "summary": response.choices[0].message.content,
            "language_level": language_level,
            "disclaimer": "This AI-generated summary is for informational purposes only."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Quick response templates for common questions (no API call needed)
QUICK_RESPONSES = {
    "what is melanoma": {
        "response": """Melanoma is a type of skin cancer that develops from the cells that give skin its color (melanocytes).

**Key points:**
- It's the most serious type of skin cancer
- Often appears as a new spot or a changing mole
- Early detection is crucial - when caught early, it's highly treatable
- Risk factors include sun exposure, fair skin, and family history

**ABCDE warning signs:**
- **A**symmetry - one half doesn't match the other
- **B**order - irregular, ragged edges
- **C**olor - varied shades of brown, black, or other colors
- **D**iameter - larger than 6mm (pencil eraser size)
- **E**volving - changing in size, shape, or color

If you notice any of these signs, please consult a dermatologist promptly.""",
        "source": "cached"
    },
    "what is a mole": {
        "response": """A mole (medical term: nevus) is a common skin growth made up of pigment cells (melanocytes).

**Key facts:**
- Most adults have 10-40 moles
- They usually appear in childhood and may change or fade over time
- Most moles are harmless (benign)
- They can be flat or raised, smooth or rough

**When to be concerned:**
- A new mole appearing after age 30
- A mole that's changing in size, shape, or color
- A mole that itches, bleeds, or doesn't heal
- A mole that looks different from your other moles

Regular self-examinations and annual skin checks with a dermatologist are recommended, especially if you have many moles or a family history of skin cancer.""",
        "source": "cached"
    }
}


@router.post("/ai-chat/quick")
async def quick_chat(
    message: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Quick responses for common questions (uses cached responses when available).
    Falls back to AI for other questions.
    """
    # Check for cached response
    message_lower = message.lower().strip()
    for key, response_data in QUICK_RESPONSES.items():
        if key in message_lower:
            return {
                "response": response_data["response"],
                "source": "cached",
                "disclaimer": "This information is for educational purposes only."
            }

    # Fall back to AI
    if not OPENAI_AVAILABLE:
        return {
            "response": "I can help with common skin questions. Try asking about specific conditions like 'What is melanoma?' or 'What is a mole?'",
            "source": "fallback"
        }

    client = get_openai_client()
    if not client:
        return {
            "response": "AI chat is not configured. Please ask your administrator to set up the OpenAI API key.",
            "source": "fallback"
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )

        return {
            "response": response.choices[0].message.content,
            "source": "ai"
        }

    except Exception as e:
        return {
            "response": f"Sorry, I encountered an error. Please try again or ask a different question.",
            "source": "error",
            "error": str(e)
        }


# Differential Diagnosis Reasoning System Prompt
DIFFERENTIAL_REASONING_PROMPT = """You are an expert dermatologist AI assistant explaining your diagnostic reasoning process.

Your task is to provide a clear, step-by-step chain-of-thought explanation for how a differential diagnosis was reached.

IMPORTANT GUIDELINES:
1. Use a logical, systematic approach (like a dermatologist would)
2. Explain what visual or clinical features support each consideration
3. Explain why certain conditions were ruled out or ranked lower
4. Be educational but accessible to patients
5. Always emphasize that this is AI-assisted analysis, not a definitive diagnosis
6. Use clear headings and bullet points for readability

FORMAT YOUR RESPONSE AS:
1. **Initial Assessment**: What key features were observed
2. **Primary Diagnosis Reasoning**: Why the top diagnosis was selected
3. **Differential Considerations**: Why other conditions were considered
4. **Key Distinguishing Features**: What separates the primary diagnosis from alternatives
5. **Recommended Next Steps**: What would confirm or rule out the diagnosis

Keep the explanation thorough but concise (around 400-600 words)."""


@router.post("/ai-chat/differential-reasoning")
async def get_differential_reasoning(
    primary_diagnosis: str = Form(...),
    confidence: Optional[float] = Form(None),
    differential_diagnoses: Optional[str] = Form(None),  # JSON string
    clinical_context: Optional[str] = Form(None),  # JSON string with patient info
    risk_level: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate chain-of-thought reasoning for differential diagnosis.
    Explains why the AI considered certain conditions and how it reached its conclusion.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Build the context for reasoning
    context_parts = [f"Primary Diagnosis: {primary_diagnosis}"]

    if confidence:
        context_parts.append(f"Confidence Level: {confidence * 100:.1f}%")

    if risk_level:
        context_parts.append(f"Risk Assessment: {risk_level}")

    # Parse differential diagnoses
    differentials = []
    if differential_diagnoses:
        try:
            differentials = json.loads(differential_diagnoses)
            if differentials:
                diff_text = "Differential Diagnoses Considered:\n"
                for i, diff in enumerate(differentials[:5], 1):
                    if isinstance(diff, dict):
                        condition = diff.get('condition', diff.get('name', str(diff)))
                        prob = diff.get('probability', diff.get('confidence', 0))
                        diff_text += f"  {i}. {condition} ({prob * 100:.1f}% probability)\n"
                    else:
                        diff_text += f"  {i}. {diff}\n"
                context_parts.append(diff_text)
        except json.JSONDecodeError:
            pass

    # Parse clinical context
    if clinical_context:
        try:
            context_data = json.loads(clinical_context)
            clinical_text = "Clinical Context:\n"
            if context_data.get('age'):
                clinical_text += f"  - Age: {context_data['age']}\n"
            if context_data.get('skin_type'):
                clinical_text += f"  - Skin Type: {context_data['skin_type']}\n"
            if context_data.get('body_location'):
                clinical_text += f"  - Location: {context_data['body_location']}\n"
            if context_data.get('duration'):
                clinical_text += f"  - Duration: {context_data['duration']}\n"
            if context_data.get('symptoms'):
                clinical_text += f"  - Symptoms: {context_data['symptoms']}\n"
            if context_data.get('family_history'):
                clinical_text += f"  - Family History: {context_data['family_history']}\n"
            context_parts.append(clinical_text)
        except json.JSONDecodeError:
            pass

    analysis_context = "\n".join(context_parts)

    prompt = f"""Please provide a chain-of-thought explanation for the following skin analysis:

{analysis_context}

Explain the diagnostic reasoning process as if you were a dermatologist explaining to a patient why this diagnosis was reached and why other conditions were considered."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": DIFFERENTIAL_REASONING_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.5
        )

        reasoning = response.choices[0].message.content

        return {
            "primary_diagnosis": primary_diagnosis,
            "reasoning": reasoning,
            "differentials_analyzed": len(differentials),
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            "disclaimer": "This AI-generated reasoning is for educational purposes only. It does not constitute medical advice. Please consult a board-certified dermatologist for proper diagnosis."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating reasoning: {str(e)}")
