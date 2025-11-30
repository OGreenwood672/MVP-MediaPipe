import huggingface_hub

print("Logging into Hugging Face...")
print("If you don't have a token, create one here: https://huggingface.co/settings/tokens (Type: Read)")

huggingface_hub.login()

print("Login successful! You can now run the heatmap script.")