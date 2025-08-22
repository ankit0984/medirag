from src.genllm import genllm
from src.store import ragcollection

if __name__ == "__main__":
    ragcollection()
    print("\n=== Medical AI Assistant ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_question = input("Enter your medical question: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Exiting Medical Assistant. Stay healthy!")
            break

        print("\n--- Answer ---\n")
        try:
            response = genllm(user_question)

            # Clean JSON with no markdown
            print(response.model_dump_json(indent=2))

            # Or access fields directly
            print("\nDefinition:", response.definition)
            print("Symptoms:", ", ".join(response.symptoms))

            print("\n==============================\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")
