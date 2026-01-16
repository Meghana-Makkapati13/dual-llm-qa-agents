from pydantic import BaseModel, Field, field_validator
from typing import List


class SessionRequest(BaseModel):
    """Request model for Q&A session"""
    subject: str = Field(..., min_length=1, description="The subject for Q&A pairs")
    num_pairs: int = Field(default=10, ge=1, le=50, description="Number of Q&A pairs to generate")
    
    @field_validator('subject')
    @classmethod
    def subject_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Subject cannot be empty or whitespace only')
        return v.strip()


class QAPair(BaseModel):
    """Model for a single question-answer pair"""
    id: int = Field(..., ge=1, description="1-based index of the Q&A pair")
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Generated answer")


class SessionResponse(BaseModel):
    """Response model for Q&A session"""
    subject: str = Field(..., description="The subject that was used")
    num_pairs: int = Field(..., description="Number of Q&A pairs generated")
    pairs: List[QAPair] = Field(..., description="List of question-answer pairs")