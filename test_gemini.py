import google.generativeai as genai

key = "YOUR_API_KEY_HERE"
genai.configure(api_key=key)

print("[Testing New Gemini Key...]")
for m in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
    try:
        print(f"Testing model: {m}...")
        model = genai.GenerativeModel(m)
        resp = model.generate_content("Hello! Explain solid state physics in 1 short sentence.")
        print(f"Success with {m}:\n{resp.text}\n")
        break
    except Exception as e:
        print(f"Failed {m}: {e}\n")
