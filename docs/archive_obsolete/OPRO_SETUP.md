# OPRO Setup Guide

This guide covers setup for Sprint 9: OPRO prompt optimization.

---

## 1. Install Dependencies

The OPRO optimizer requires additional API client libraries:

```bash
pip install anthropic openai
```

Or install from updated requirements:

```bash
pip install -r requirements.txt
```

---

## 2. Get an API Key

You need an API key from either:

### Option A: Anthropic Claude (Recommended)

1. Go to: https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-ant-...`)

**Pricing** (as of Oct 2024):
- Claude 3.5 Sonnet: $3/M input tokens, $15/M output tokens
- Estimated cost for full OPRO run (50 iterations): **<$1**

### Option B: OpenAI GPT-4

1. Go to: https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `sk-...`)

**Pricing**:
- GPT-4: $30/M input tokens, $60/M output tokens
- Estimated cost for full OPRO run (50 iterations): **~$5-10**

---

## 3. Set Environment Variable

### Linux/Mac/WSL:

```bash
# For Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# OR for OpenAI GPT-4
export OPENAI_API_KEY="sk-your-key-here"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Windows (PowerShell):

```powershell
# For Anthropic Claude (recommended)
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"

# OR for OpenAI GPT-4
$env:OPENAI_API_KEY="sk-your-key-here"

# Make permanent
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-your-key-here', 'User')
```

### Windows (Command Prompt):

```cmd
# For Anthropic Claude (recommended)
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# OR for OpenAI GPT-4
set OPENAI_API_KEY=sk-your-key-here

# Make permanent
setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
```

---

## 4. Verify Setup

```bash
# Check API key is set
python -c "import os; print('API key set:', bool(os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY')))"

# Test imports
python -c "from scripts.opro_optimizer import OPROOptimizer; print('OPRO ready!')"
```

Expected output:
```
API key set: True
OPRO ready!
```

---

## 5. Alternative: Pass API Key Directly

If you don't want to set environment variables, you can pass the API key directly:

```bash
python scripts/run_opro.py \
    --api_key "sk-ant-your-key-here" \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_test
```

**⚠️ Security Warning**: Passing keys via command line can expose them in shell history. Environment variables are preferred.

---

## 6. Choose Optimizer LLM

By default, OPRO uses Claude 3.5 Sonnet. You can change the LLM:

```bash
# Claude 3.5 Sonnet (default, recommended)
python scripts/run_opro.py --optimizer_llm claude-3-5-sonnet-20241022 ...

# GPT-4 (OpenAI)
python scripts/run_opro.py --optimizer_llm gpt-4o ...

# GPT-4 Turbo
python scripts/run_opro.py --optimizer_llm gpt-4-turbo ...
```

---

## Troubleshooting

### Error: "No module named 'anthropic'"

**Solution**:
```bash
pip install anthropic openai
```

### Error: "API key not found"

**Solution**:
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY  # Linux/Mac/WSL
echo %ANTHROPIC_API_KEY%  # Windows CMD
echo $env:ANTHROPIC_API_KEY  # Windows PowerShell

# If not set, export it again
export ANTHROPIC_API_KEY="your-key-here"
```

### Error: "Authentication failed"

**Possible causes**:
1. API key is incorrect or expired
2. API key doesn't have sufficient permissions
3. Account has no credits (check your billing)

**Solution**:
- Verify key on API console
- Check account billing/credits
- Generate a new API key

### Error: "Rate limit exceeded"

**Solution**:
- Wait a few minutes and retry
- Reduce `--candidates_per_iter` (e.g., from 5 to 3)
- Add delay between iterations (modify `opro_optimizer.py`)

---

## Next Steps

Once setup is complete, proceed to:

1. **[OPRO Quick Start Guide](OPRO_QUICKSTART.md)** - Run your first optimization
2. **[Sprint 9 Specification](sprints/SPRINT9_OPRO_SPECIFICATION.md)** - Full technical details

---

## Security Best Practices

1. **Never commit API keys to git**
   - Add `.env` to `.gitignore` if using dotenv
   - Use environment variables or secret management

2. **Rotate keys periodically**
   - Generate new keys every few months
   - Delete old keys from API console

3. **Monitor usage**
   - Check API console for unexpected usage
   - Set up billing alerts

4. **Limit key permissions**
   - Use project-specific keys if available
   - Set rate limits if supported

---

## Cost Monitoring

### Anthropic Console
- Dashboard: https://console.anthropic.com/
- Usage tracking available in real-time
- Set up billing alerts

### OpenAI Dashboard
- Dashboard: https://platform.openai.com/usage
- Monthly usage tracking
- Set up billing limits

### Estimated Costs

**Typical OPRO run** (50 iterations, 3 candidates/iter, with early stopping):

| LLM | Input Tokens | Output Tokens | Total Cost |
|-----|--------------|---------------|------------|
| **Claude 3.5 Sonnet** | ~50k | ~25k | **~$0.53** |
| GPT-4o | ~50k | ~25k | ~$2.25 |
| GPT-4 Turbo | ~50k | ~25k | ~$3.00 |

**Note**: Qwen2-Audio evaluation runs locally on your GPU (free).

---

**Ready to start?** → See [OPRO Quick Start Guide](OPRO_QUICKSTART.md)
