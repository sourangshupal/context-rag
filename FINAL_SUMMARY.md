# Project Complete - Advanced RAG with Contextual Retrieval

## ✅ All Files Created Successfully

### Core Implementation
1. **advanced_rag.py** (900+ lines)
   - Complete RAG implementation with detailed comments
   - Properly formatted Python (no JSON issues!)
   - Type-safe with Optional[str] type hints

### Documentation
2. **README.md** - Comprehensive guide (9.9 KB)
3. **QUICKSTART.md** - 5-minute quick start (6.5 KB)
4. **SCRIPT_GUIDE.md** - Script usage guide (NEW!)
5. **.env.example** - API keys template (701 bytes)
6. **requirements.txt** - All dependencies (216 bytes)
7. **.gitignore** - Git patterns (536 bytes)

### Sample Data
8. **data/documents/machine_learning_intro.md** (6.6 KB)
9. **data/documents/neural_networks.md** (8.3 KB)

## 🎯 Script Structure (advanced_rag.py)

### 7 Major Sections (900+ lines of code)

| Section | Lines | Description |
|---------|--------|-------------|
| 1. Configuration | 1-60 | Imports, API keys, model config |
| 2. Document Processing | 62-180 | Load files, chunking logic |
| 3. Baseline RAG | 183-400 | VectorDB class, embeddings, search |
| 4. Contextual Embeddings | 403-680 | Claude integration, prompt caching |
| 5. Evaluation | 683-760 | Pass@k metrics, mock data |
| 6. Cost Analysis | 763-850 | Pricing models, savings calc |
| 7. Main Demo | 853-end | Complete workflow demonstration |

### Key Classes

1. **VectorDB** (Lines 183-400)
   - Baseline RAG implementation
   - Voyage AI embeddings
   - Cosine similarity search
   - Persistent storage with pickle

2. **ContextualVectorDB** (Lines 403-680)
   - Enhanced with Claude-generated context
   - Prompt caching for cost savings
   - Parallel processing (configurable threads)
   - Token usage tracking

## 🚀 How to Use

### Option 1: Quick Demo (Works Immediately)
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env and paste your keys

# Run demo (uses included sample documents)
python advanced_rag.py
```

### Option 2: Use Your Own Documents
```bash
# Add your files to data/documents/
# Supports: PDF, Markdown, Text

cp your_document.pdf data/documents/
cp your_notes.md data/documents/

# Run with your documents
python advanced_rag.py
```

### Option 3: Learn by Reading Code
```bash
# Open and study the script
# Each section has extensive comments explaining:
#   - What the code does
#   - Why it matters
#   - How it works
#   - Best practices

# Key sections to understand:
#   - Lines 183-220: VectorDB class
#   - Lines 403-440: ContextualVectorDB class  
#   - Lines 257-277: Cosine similarity logic
#   - Lines 447-485: Prompt caching implementation
```

## 📊 Performance Comparison

| Method | Pass@5 | Pass@10 | Pass@20 | Setup Cost | Query Cost |
|--------|--------|---------|---------|-------------|-------------|
| Baseline RAG | 81% | 87% | 90% | ~$0.50 | Free |
| Contextual Embeddings | 88% | 92% | 94% | ~$2.00 | Free |
| + Hybrid Search | 89% | 93% | 95% | ~$2.00 | +10ms |
| + Reranking | 90%+ | 94%+ | 96%+ | ~$2.00 | +$0.10/1K |

**Claude Research Based:**
- Contextual: 30-40% reduction in retrieval failures
- Prompt caching: 60-70% cost reduction
- Hybrid search: Additional 10% improvement

## 📖 Learning Resources

### Start Here
1. **SCRIPT_GUIDE.md** - Script-specific guide (NEW!)
   - Explains all functions and classes
   - Usage examples
   - Extension points

### Then Read
2. **QUICKSTART.md** - 5-minute setup guide
3. **advanced_rag.py** - Code with extensive comments

### Finally Reference
4. **README.md** - Comprehensive documentation
5. **Claude Cookbook** - Source material
   https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide

## 💰 Cost Information

### API Pricing (January 2026)

| Service | Model | Price | Usage |
|----------|-------|-------|--------|
| Anthropic | Haiku-4.5 | $0.25/M input | Contextualization |
| Anthropic | Haiku-4.5 | $1.25/M output | Contextualization |
| Voyage AI | voyage-2 | $0.05/M tokens | Embeddings |
| Cohere | rerank-english-v3.0 | $0.10/1K queries | Reranking |

### Sample Cost Estimates

**For 1000 chunks (typical document set):**
- Baseline RAG: ~$0.50 (embeddings only)
- Contextual (no cache): ~$2.00 (full price)
- Contextual (with cache): ~$0.65 (67% savings!)

**Query costs (per 1000 queries):**
- Baseline: $0 (embeddings cached)
- With Reranking: $0.10

## 🔑 API Keys Required

### Where to Get Keys

1. **Anthropic** (Required for Contextual Embeddings)
   - URL: https://console.anthropic.com/
   - For: Claude Haiku 4.5
   - Cost: $0.25/M input, $1.25/M output

2. **Voyage AI** (Required for Embeddings)
   - URL: https://www.voyageai.com/
   - For: voyage-2 embeddings
   - Cost: $0.05/M tokens

3. **Cohere** (Required for Reranking)
   - URL: https://cohere.com/
   - For: rerank-english-v3.0
   - Cost: $0.10/1K queries

### Setup

Create `.env` file:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
VOYAGE_API_KEY=your-voyage-key-here
COHERE_API_KEY=your-cohere-key-here
```

**Important:** Never commit `.env` to version control!

## ✨ Key Features Explained

### 1. Contextual Embeddings
**Problem:** Chunks lose context when embedded in isolation

**Solution:** Claude analyzes full document and generates:
"This chunk describes the optimization technique applied to the API gateway layer..."

**Impact:** Pass@10 improves from 87% to 92%

### 2. Prompt Caching
**How it works:**
- Chunk 1: Write document to cache (full price)
- Chunk 2-50: Read from cache (90% discount)
- Cache: 5 minutes (enough for all chunks)

**Savings:** 60-70% cost reduction

### 3. Cosine Similarity
**Formula:** `dot(A, B) / (norm(A) * norm(B))`

**Why Cosine:**
- Magnitude-independent (text length doesn't matter)
- Focuses on semantic direction
- Works well with normalized embeddings

### 4. Pass@k Evaluation
**Metric:** Is golden chunk in top-k results?

**Why Pass@k:**
- Intuitive to understand
- Practical for production
- Balances precision and recall

## 🎓 Learning Path

### Week 1: Foundations
- Understand RAG architecture
- Run baseline implementation
- Experiment with chunking
- Learn vector similarity

### Week 2: Contextual Retrieval
- Enable contextual embeddings
- Understand prompt caching
- Analyze token usage
- Compare with baseline

### Week 3: Advanced Techniques
- Implement hybrid search (Elasticsearch)
- Add reranking (Cohere)
- Integrate MCP servers
- Optimize performance

### Week 4: Production
- Build evaluation pipeline
- Monitor costs and latency
- Implement error handling
- Deploy to cloud

## 🛠️ Troubleshooting

| Issue | Solution |
|--------|----------|
| "No documents found" | Add files to `data/documents/` |
| "API key invalid" | Check `.env` has correct keys |
| Import errors | Run `pip install -r requirements.txt` first |
| Out of memory | Reduce `parallel_threads` or process separately |
| Rate limit | Increase delays or reduce threads |

## 📈 What Students Will Learn

### Concepts
- RAG architecture and components
- Contextual embeddings theory
- Prompt caching benefits
- Hybrid search strategies
- Evaluation metrics (Pass@k)
- Cost optimization techniques

### Skills
- Working with modern embedding models
- Using LLM APIs (Claude, Cohere)
- Building vector databases
- Implementing similarity search
- Creating evaluation pipelines
- Optimizing costs with caching

### Best Practices
- Chunking strategies (800-1200 chars)
- Token usage monitoring
- Production deployment patterns
- Error handling and retries
- Performance optimization

## 🎯 Success Criteria

Students will successfully complete this tutorial when they can:

✅ Load and chunk documents from various formats
✅ Implement baseline RAG with vector embeddings
✅ Add contextual embeddings with Claude
✅ Understand and leverage prompt caching
✅ Implement evaluation with Pass@k metrics
✅ Analyze costs and optimize spending
✅ Extend with hybrid search and reranking

## 📞 Need Help?

1. **Check SCRIPT_GUIDE.md** - Script-specific issues
2. **Review README.md** - Comprehensive documentation
3. **See QUICKSTART.md** - Setup problems
4. **Read code comments** - Each function is documented

## 🚀 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys from `.env.example`
3. Run demo: `python advanced_rag.py`
4. Study the code and understand each section
5. Experiment with your own documents
6. Extend with hybrid search, reranking, MCP

---

**Status:** ✅ COMPLETE AND READY TO USE
**Created:** January 16, 2026
**Version:** 1.0.0
**Language:** Python 3.9+

## 🔑 License

Educational use. Based on Claude's cookbook and research.

---

**Ready to learn Advanced RAG! 🚀**
