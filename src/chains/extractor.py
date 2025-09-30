import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() 

SYSTEM = (
    "You are a clinical information extractor. From a patient–doctor dialogue, "
    "produce strict JSON with exactly two top-level keys: \"conditions\" and \"medications\".\n"
    "\n"
    "Schema\n"
    "- \"conditions\": array of objects. Each object may have the keys:\n"
    "  • \"name\" (required, short clinical term)\n"
    "  • \"status\" (optional; one of: active, suspected, possible, resolved)\n"
    "  • \"onset\" (optional; short free-text like a date or phrase)\n"
    "  • \"negated\" (optional; boolean true/false)\n"
    "  • \"evidence\" (optional; brief quote or snippet from the dialogue)\n"
    "- \"medications\": array of objects. Each object may have the keys:\n"
    "  • \"name\" (required; generic drug or class)\n"
    "  • \"dose\" (optional; e.g., 200 mg)\n"
    "  • \"route\" (optional; e.g., PO, IV)\n"
    "  • \"freq\"  (optional; e.g., BID, q6h)\n"
    "  • \"negated\" (optional; boolean true/false)\n"
    "  • \"recommendation\" (optional; boolean true/false; true if suggested/recommended rather than confirmed taken)\n"
    "  • \"evidence\" (optional; brief quote or snippet)\n"
    "\n"
    "Rules\n"
    "1) Return ONLY JSON. No prose, no markdown, no extra text.\n"
    "2) Use only the keys listed above; do not add other keys. Omit unknown fields rather than inventing values.\n"
    "3) \"negated\" must be a boolean (true/false), never a string.\n"
    "4) Extract only entities explicitly about THIS patient. Do not add umbrella categories (e.g., \"chronic disease\") "
    "   unless the term itself is explicitly mentioned. Do not list general advice as a patient condition.\n"
    "5) Prefer concise names (e.g., sore throat, dry cough). If something is explicitly ruled out, include it in "
    "   \"conditions\" with \"negated\": true.\n"
    "6) If nothing fits a section, return an empty array for that section.\n"
    "7) Be faithful to the dialogue; do not hallucinate.\n"
    "\n"
    "{format_instructions}"
)

HUMAN = (
    "Dialogue:\n{dialogue}\n\n"
    "Return ONLY JSON. No extra text."
)

def build_extractor(model_name="gpt-4o-mini", temperature=0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n" + "{format_instructions}"),
        ("human", HUMAN),
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser  # prompt -> LLM -> JSON 
    return chain

def run_extractor_on_text(chain, dialogue_text):
    try:
        return chain.invoke({"dialogue": dialogue_text})
    except Exception as e:
        return {"conditions": [], "medications": [], "_error": str(e)}
