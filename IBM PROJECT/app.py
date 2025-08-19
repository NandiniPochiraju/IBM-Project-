import os, json, numpy as np # type: ignore
from typing import List, Dict
from flask import Flask, request, jsonify # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore

# --- Optional: IBM watsonx.ai (Granite) ---
USE_IBM = all(k in os.environ for k in ["WX_API_KEY", "WX_PROJECT_ID", "WX_REGION"])
if USE_IBM:
    from ibm_watsonx_ai import Credentials, WatsonxLLM # type: ignore

MODEL_ID = "all-MiniLM-L6-v2"  # tiny, fast sentence embedding model
GEN_MODEL = "granite-13b-instruct"  # watsonx.ai text generation model name

app = Flask(__name__)

# -----------------------------
# Load a tiny demo recipe set
# -----------------------------
RECIPES: List[Dict] = [
    {
        "title": "Tomato Basil Pasta",
        "ingredients": ["pasta", "tomato", "garlic", "olive oil", "basil", "salt"],
        "steps": [
            "Boil pasta until al dente.",
            "Sauté garlic in olive oil, add chopped tomatoes, simmer 6–8 min.",
            "Toss pasta with sauce, finish with basil and salt."
        ],
        "diet": ["vegetarian"]
    },
    {
        "title": "Veggie Fried Rice",
        "ingredients": ["rice", "egg", "carrot", "peas", "soy sauce", "spring onion", "oil"],
        "steps": [
            "Scramble egg, set aside.",
            "Stir-fry veggies, add rice and soy sauce.",
            "Fold in egg and spring onion; serve."
        ],
        "diet": ["contains-egg"]
    },
    {
        "title": "Chickpea Salad",
        "ingredients": ["chickpeas", "cucumber", "tomato", "onion", "lemon", "olive oil", "salt", "pepper"],
        "steps": [
            "Combine chopped veggies with chickpeas.",
            "Dress with lemon, olive oil, salt and pepper."
        ],
        "diet": ["vegan", "gluten-free"]
    },
]

# --------------------------------
# Build embeddings + FAISS index
# --------------------------------
embedder = SentenceTransformer(MODEL_ID)

def make_doc(recipe: Dict) -> str:
    return f"{recipe['title']} | ingredients: {', '.join(recipe['ingredients'])} | steps: {' '.join(recipe['steps'])}"

DOCS = [make_doc(r) for r in RECIPES]
EMB = np.array(embedder.encode(DOCS, normalize_embeddings=True), dtype="float32")
index = faiss.IndexFlatIP(EMB.shape[1])     # inner product == cosine because normalized
index.add(EMB)

# --------------------------------
# Optional: IBM watsonx.ai client
# --------------------------------
if USE_IBM:
    creds = Credentials(
        api_key=os.environ["WX_API_KEY"],
        url=f"https://{os.environ['WX_REGION']}.ml.cloud.ibm.com"
    )
    llm = WatsonxLLM(
        model_id=GEN_MODEL,
        params={"decoding_method": "greedy", "max_new_tokens": 350, "temperature": 0.2},
        project_id=os.environ["WX_PROJECT_ID"],
        credentials=creds,
    )

def retrieve(ingredients: List[str], top_k: int = 3) -> List[Dict]:
    q = "ingredients: " + ", ".join(ingredients)
    q_emb = np.array([embedder.encode(q, normalize_embeddings=True)], dtype="float32")
    scores, idx = index.search(q_emb, top_k)
    return [RECIPES[i] for i in idx[0]]

def build_prompt(user_ing: List[str], prefs: Dict, cands: List[Dict]) -> str:
    pref_str = ", ".join([f"{k}={v}" for k, v in prefs.items() if v])
    context = "\n\n".join(
        [f"Title: {r['title']}\nIngredients: {', '.join(r['ingredients'])}\nSteps: {' '.join(r['steps'])}"
         for r in cands]
    )
    return f"""
You are a culinary assistant. The user has only these ingredients: {', '.join(user_ing)}.
Preferences/constraints: {pref_str or 'none'}.

From the CONTEXT recipes below, pick the best fit, adapt to available ingredients,
propose safe substitutions if needed, and output:

- Title
- Why this fits the ingredients
- Ingredients list (quantities, realistic)
- Step-by-step instructions (numbered)
- Substitutions (bullet points)
- Dietary notes

CONTEXT:
{context}
"""

def generate_recipe(prompt: str) -> str:
    if USE_IBM:
        return llm.generate(prompt)
    # Fallback local template if IBM creds not set
    return (
        "Title: Pantry-Friendly Dish\n"
        "Why it fits: uses mostly the provided ingredients.\n"
        "Ingredients: (quantities depend on servings)\n"
        "1) Combine what you have; 2) Cook; 3) Season and serve.\n"
        "Substitutions: olive oil→any neutral oil; basil→coriander; lemon→vinegar.\n"
        "Dietary notes: adjust for vegan/vegetarian as needed."
    )

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json(force=True)
    ingredients = [i.strip().lower() for i in data.get("ingredients", []) if i.strip()]
    prefs = {
        "diet": data.get("diet", ""),
        "servings": data.get("servings", ""),
        "avoid": ", ".join(data.get("avoid", [])) if data.get("avoid") else ""
    }
    if not ingredients:
        return jsonify({"error": "Provide 'ingredients': [..]"}), 400

    candidates = retrieve(ingredients, top_k=3)
    prompt = build_prompt(ingredients, prefs, candidates)
    recipe_text = generate_recipe(prompt)
    return jsonify({
        "ingredients_provided": ingredients,
        "retrieved_candidates": [c["title"] for c in candidates],
        "generated_recipe": recipe_text
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "message": "RAG Recipe Agent up. POST /suggest with JSON."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
