# Using Hugging Face (FREE!) 🎉

## Why Hugging Face?
- ✅ **Completely FREE** - No credit card required!
- ✅ **No API costs** - Unlimited usage with free tier
- ✅ **Good quality** - Uses open-source models like Mistral-7B
- ✅ **Simple setup** - Just get a free token

---

## Step 1: Get Your FREE Hugging Face Token

1. Go to https://huggingface.co/join
2. Sign up for a FREE account (use email or Google/GitHub)
3. Go to https://huggingface.co/settings/tokens
4. Click **"New token"**
5. Give it a name (e.g., "MAI Mashup")
6. Select **"Read"** permission
7. Click **"Generate"**
8. Copy your token (starts with `hf_...`)

---

## Step 2: Set Your Token

### Option A: Export in Terminal (temporary)
```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

### Option B: Add to Profile (permanent)
```bash
# Add to ~/.bash_profile or ~/.zshrc
echo 'export HUGGINGFACE_TOKEN="hf_your_token_here"' >> ~/.bash_profile
source ~/.bash_profile
```

### Option C: Set via Python
```python
import os
os.environ['HUGGINGFACE_TOKEN'] = 'hf_your_token_here'
```

---

## Step 3: Test Your Token

```bash
python test_huggingface_token.py
```

---

## Step 4: Use in MAI

1. Start the server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Open http://localhost:8000

3. Select **"🆓 Hugging Face (FREE!)"** as the AI Provider

4. Upload your songs and create a mashup!

---

## Available FREE Models

The system uses **Mistral-7B-Instruct-v0.2** by default, which is:
- Fast and efficient
- Good at following instructions
- Completely free to use

Other free models you can try:
- `meta-llama/Llama-2-7b-chat-hf` - Facebook's Llama
- `HuggingFaceH4/zephyr-7b-beta` - Great for instructions
- `tiiuae/falcon-7b-instruct` - Fast and efficient

To change model, set environment variable:
```bash
export HUGGINGFACE_MODEL="meta-llama/Llama-2-7b-chat-hf"
```

---

## Troubleshooting

### "Model is loading..."
- Free tier models sometimes need to "wake up"
- The system will automatically wait and retry
- Usually takes 10-30 seconds

### "API token not provided"
- Make sure you exported the token: `echo $HUGGINGFACE_TOKEN`
- Token should start with `hf_`
- Re-run the export command

### Rate Limiting
- Free tier has generous limits
- If you hit limits, wait a few minutes and try again
- Or create multiple free accounts with different tokens

---

## Comparison: Free vs Paid

| Feature | Hugging Face (FREE) | OpenAI (Paid) |
|---------|---------------------|---------------|
| Cost | $0 | ~$0.01-0.03 per mashup |
| Setup | Free token | Credit card required |
| Quality | Good (7B models) | Excellent (GPT-4) |
| Speed | Medium (10-30s) | Fast (5-10s) |
| Rate Limits | Generous | Pay per use |

**Recommendation:** Start with Hugging Face FREE, upgrade to OpenAI later if needed!

---

## Support

If you have issues:
1. Check token is set: `echo $HUGGINGFACE_TOKEN`
2. Verify token at https://huggingface.co/settings/tokens
3. Try the test script: `python test_huggingface_token.py`
4. Check server logs for detailed error messages

Enjoy creating FREE AI mashups! 🎵✨
