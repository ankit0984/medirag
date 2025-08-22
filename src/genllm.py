import re
import config
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.autorag import auto_retriever


# ---------------- Utility: Strip Markdown ---------------- #
def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from a string."""
    if not text:
        return text
    # Remove bold/italic markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Remove headers like ###, ##
    text = re.sub(r"#+\s*", "", text)
    # Remove backticks
    text = text.replace("`", "")
    return text.strip()


# ---------------- Structured Schema (Pydantic v2) ---------------- #
class MedicalResponse(BaseModel):
    definition: str = Field(..., description="Brief definition of the condition.")
    causes_risk_factors: List[str] = Field(..., description="Causes and risk factors as a list.")
    symptoms: List[str] = Field(..., description="List of symptoms.")
    diagnosis: List[str] = Field(..., description="Diagnostic methods.")
    treatment_cure: List[str] = Field(..., description="Treatment or cure methods.")
    prognosis_complications: List[str] = Field(..., description="Prognosis and possible complications.")
    prevention_lifestyle: List[str] = Field(..., description="Prevention and lifestyle advice.")
    additional_notes: List[str] = Field(..., description="Additional notes.")

    def strip_all_markdown(self) -> "MedicalResponse":
        """Return a copy of this object with Markdown removed from all fields."""
        return MedicalResponse(
            definition=strip_markdown(self.definition),
            causes_risk_factors=[strip_markdown(x) for x in self.causes_risk_factors],
            symptoms=[strip_markdown(x) for x in self.symptoms],
            diagnosis=[strip_markdown(x) for x in self.diagnosis],
            treatment_cure=[strip_markdown(x) for x in self.treatment_cure],
            prognosis_complications=[strip_markdown(x) for x in self.prognosis_complications],
            prevention_lifestyle=[strip_markdown(x) for x in self.prevention_lifestyle],
            additional_notes=[strip_markdown(x) for x in self.additional_notes],
        )


# ---------------- LLM + Prompt ---------------- #
llm = ChatGoogleGenerativeAI(
    api_key=config.gemini_key,
    model=config.gemini_model,
    max_retries=2
)

structured_llm = llm.with_structured_output(MedicalResponse)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a highly knowledgeable medical AI assistant.
Always output in **plain text only** (NO markdown, no asterisks, no bold, no headings).
Use simple numbered or bulleted lists when needed.

Required fields (always output all):
1. Definition
2. Causes & Risk Factors
3. Symptoms
4. Diagnosis
5. Treatment / Cure
6. Prognosis / Complications
7. Prevention & Lifestyle Advice
8. Additional Notes

---
Context:
{context}

Question:
{input}

Structured Medical Response:
"""),
    ("human", "Answer the question above in the exact plain-text structured format.")
])


# ---------------- Retrieval Chain ---------------- #
class RetrievalChain:
    def __init__(self, retriever=auto_retriever):
        self.retriever = retriever

    def invoke(self, question: str) -> MedicalResponse:
        try:
            docs = self.retriever.invoke(question)   # ✅ new API
            context_text = "\n\n".join([d.page_content for d in docs])

            formatted_prompt = prompt_template.format(
                context=context_text,
                input=question
            )

            response = structured_llm.invoke(formatted_prompt)

            # ✅ Ensure no Markdown in final output
            return response.strip_all_markdown()

        except Exception as e:
            print(f"⚠️ Error in structured output: {e}")
            return MedicalResponse(
                definition="Not available in the provided context.",
                causes_risk_factors=[],
                symptoms=[],
                diagnosis=[],
                treatment_cure=[],
                prognosis_complications=[],
                prevention_lifestyle=[],
                additional_notes=[]
            )


# ---------------- Main Entry Function ---------------- #
def genllm(question: str = "What is acne?") -> MedicalResponse:
    """
    Always returns a validated MedicalResponse object with NO Markdown.
    """
    chain = RetrievalChain()
    return chain.invoke(question)

