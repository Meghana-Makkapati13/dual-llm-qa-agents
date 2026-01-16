from openai import OpenAI
import os
from typing import List
import logging

logger = logging.getLogger(__name__)


class QuestionAgent:
    """Agent responsible for generating questions using OpenAI API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # You can change to "gpt-4o", "gpt-4-turbo", etc.
        self.difficulty_levels = ["easy", "medium", "hard"]
        self.question_history: List[str] = []
    
    def generate_question(self, subject: str, iteration: int, total_iterations: int) -> str:
        """
        Generate a question based on subject and difficulty progression
        
        Args:
            subject: The topic for the question
            iteration: Current iteration number (0-indexed)
            total_iterations: Total number of iterations
            
        Returns:
            Generated question string
        """
        # Determine difficulty based on progression
        progress = iteration / max(total_iterations - 1, 1)
        if progress < 0.33:
            difficulty = "easy"
        elif progress < 0.66:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        # Build prompt with history awareness
        history_context = ""
        if self.question_history:
            history_context = f"\n\nPreviously asked questions:\n" + "\n".join(
                f"- {q}" for q in self.question_history[-3:]
            )
        
        prompt = f"""You are an expert educator creating {difficulty} level questions about {subject}.

Generate ONE clear, unambiguous question about {subject} at {difficulty} difficulty level.

Requirements:
- The question must be specific and answerable
- Make it different from previous questions in style and focus
- For easy: focus on definitions and basic concepts
- For medium: focus on application and understanding
- For hard: focus on analysis, synthesis, or complex scenarios
- Do not repeat similar questions
{history_context}

Generate only the question, no explanations or preamble."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates educational questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            question = response.choices[0].message.content.strip()
            
            # Store in history
            self.question_history.append(question)
            
            logger.info(f"Generated {difficulty} question: {question}")
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            raise RuntimeError(f"Failed to generate question: {str(e)}")


class AnswerAgent:
    """Agent responsible for answering questions using OpenAI API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # You can change to "gpt-4o", "gpt-4-turbo", etc.
    
    def generate_answer(self, question: str) -> str:
        """
        Generate an answer to the given question
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer string
        """
        prompt = f"""You are a knowledgeable expert. Answer the following question concisely and accurately.

Question: {question}

Provide a clear, well-structured answer. Be informative but concise. Use examples if they help clarify the concept.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable expert who provides clear and accurate answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")


def run_qa_session(subject: str, num_pairs: int, api_key: str) -> List[dict]:
    """
    Run a Q&A session with the specified number of pairs
    
    Args:
        subject: The topic for questions
        num_pairs: Number of Q&A pairs to generate
        api_key: OpenAI API key
        
    Returns:
        List of dictionaries containing id, question, and answer
    """
    question_agent = QuestionAgent(api_key)
    answer_agent = AnswerAgent(api_key)
    
    pairs = []
    
    logger.info(f"Starting Q&A session on '{subject}' with {num_pairs} pairs")
    
    for i in range(num_pairs):
        try:
            # Generate question
            question = question_agent.generate_question(subject, i, num_pairs)
            
            # Generate answer
            answer = answer_agent.generate_answer(question)
            
            # Store pair
            pair = {
                "id": i + 1,
                "question": question,
                "answer": answer
            }
            pairs.append(pair)
            
            logger.info(f"Completed pair {i + 1}/{num_pairs}")
            
        except Exception as e:
            logger.error(f"Error in iteration {i + 1}: {e}")
            raise
    
    logger.info(f"Q&A session completed successfully with {len(pairs)} pairs")
    return pairs