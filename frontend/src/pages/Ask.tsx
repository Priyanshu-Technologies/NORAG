import { useState, type FormEvent } from 'react';
import {
  ArrowUp,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  FileText,
  LoaderCircle,
  ShieldCheck,
  Sparkles,
  XCircle,
} from 'lucide-react';
import { noragApi, type QueryResult, type QuerySource } from '../api/client';
import './Ask.css';
import ReactMarkdown from 'react-markdown';

function displayNumber(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function SourceCard({ source }: { source: QuerySource }) {
  const [expanded, setExpanded] = useState(false);
  const scores = source.retrieval_scores;

  return (
    <article className="ask-source">
      <button
        type="button"
        className="ask-source-toggle"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <div className="ask-source-heading">
          <span className="ask-source-number">{source.source_id}</span>
          <div>
            <strong>{source.section || `Document passage ${source.chunk_id.slice(0, 8)}`}</strong>
            <span>
              {source.page_numbers.length
                ? `Page ${source.page_numbers.join(', ')}`
                : 'Page not specified'}
              {' · '}
              {source.token_count} tokens
            </span>
          </div>
        </div>
        {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>

      {expanded && (
        <div className="ask-source-content">
          <p>{source.text}</p>
          <div className="ask-source-scores">
            <span>Hybrid: {displayNumber(scores.hybrid_score)}</span>
            <span>Dense: {displayNumber(scores.dense_score)}</span>
            <span>Sparse: {displayNumber(scores.sparse_score)}</span>
            <span>Rerank: {displayNumber(scores.rerank_score)}</span>
          </div>
          <small>Chunk ID: {source.chunk_id}</small>
        </div>
      )}
    </article>
  );
}

function Diagnostics({ result }: { result: QueryResult }) {
  const citation = result.citation_validation;
  const claims = result.claim_verification;
  const retrieval = result.retrieval;

  return (
    <section className="ask-diagnostics">
      <div className="ask-section-heading">
        <div>
          <span className="eyebrow">PIPELINE INSPECTION</span>
          <h2>Retrieval intelligence</h2>
        </div>
        <span className="ask-live-indicator">REAL RESULT</span>
      </div>

      <div className="ask-diagnostic-grid">
        <div className="ask-diagnostic-card">
          <Database size={17} />
          <span>Query intent</span>
          <strong>{retrieval.intent || '—'}</strong>
          <small>Route: {retrieval.route || '—'}</small>
        </div>

        <div className="ask-diagnostic-card">
          <Sparkles size={17} />
          <span>Hybrid weights</span>
          <strong>
            {displayNumber(retrieval.dense_weight, 2)} / {displayNumber(retrieval.sparse_weight, 2)}
          </strong>
          <small>Dense / sparse</small>
        </div>

        <div className="ask-diagnostic-card">
          <BookOpen size={17} />
          <span>Context usage</span>
          <strong>{result.context.token_count.toLocaleString()}</strong>
          <small>
            {result.context.included_count} included · {result.context.excluded_count} excluded
          </small>
        </div>

        <div className="ask-diagnostic-card">
          <Cpu size={17} />
          <span>Generation model</span>
          <strong>{result.generation.model || '—'}</strong>
          <small>{result.generation.provider || 'Provider unavailable'}</small>
        </div>
      </div>

      <div className="ask-verification-grid">
        <div className="ask-verification-card">
          <div className="ask-verification-title">
            <ShieldCheck size={18} />
            <strong>Citation validation</strong>
          </div>
          <div className="ask-verification-value">
            {(citation.citation_coverage * 100).toFixed(1)}%
          </div>
          <div className="ask-progress-track">
            <div
              className="ask-progress-fill"
              style={{
                width: `${Math.min(100, Math.max(0, citation.citation_coverage * 100))}%`,
              }}
            />
          </div>
          <p>
            {citation.cited_sentences} cited · {citation.uncited_sentences} uncited ·{' '}
            {citation.invalid_citation_count} invalid
          </p>
        </div>

        <div className="ask-verification-card">
          <div className="ask-verification-title">
            <CheckCircle2 size={18} />
            <strong>Claim verification</strong>
          </div>
          <div className="ask-verification-value">
            {claims.total_claims === 0 ? '—' : `${(claims.support_rate * 100).toFixed(1)}%`}
          </div>
          <p>
            {claims.supported_claims} supported · {claims.contradicted_claims} contradicted ·{' '}
            {claims.insufficient_claims} insufficient
          </p>
        </div>
      </div>

      <div className="ask-generation-meta">
        <span>Input: {result.generation.input_tokens ?? '—'} tokens</span>
        <span>Output: {result.generation.output_tokens ?? '—'} tokens</span>
        <span>Candidates: {retrieval.returned_results}</span>
        <span>Context budget: {result.context.available_tokens.toLocaleString()}</span>
      </div>
    </section>
  );
}

export default function Ask() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await noragApi.ask(trimmedQuery, topK);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="ask-page">
      <header className="ask-intro">
        <span className="eyebrow">
          <span className="eyebrow-line" />
          KNOWLEDGE INTERFACE
        </span>
        <h1>
          Ask your <span>knowledge.</span>
        </h1>
        <p>
          Ask questions across your indexed documents. Every response is generated using retrieved
          evidence from your knowledge base.
        </p>
      </header>

      <form className="ask-composer" onSubmit={handleSubmit}>
        <label htmlFor="ask-query">YOUR QUESTION</label>
        <textarea
          id="ask-query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask something about your documents..."
          rows={3}
          disabled={loading}
        />

        <div className="ask-composer-footer">
          <label className="ask-topk-label" htmlFor="ask-top-k">
            RETRIEVAL DEPTH
            <select
              id="ask-top-k"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              disabled={loading}
            >
              <option value={3}>3 sources</option>
              <option value={5}>5 sources</option>
              <option value={10}>10 sources</option>
            </select>
          </label>

          <button className="ask-submit" type="submit" disabled={!query.trim() || loading}>
            {loading ? (
              <>
                <LoaderCircle className="ask-spinner" size={17} />
                Processing
              </>
            ) : (
              <>
                Ask NORAG
                <ArrowUp size={17} />
              </>
            )}
          </button>
        </div>
      </form>

      {loading && (
        <div className="ask-loading">
          <LoaderCircle className="ask-spinner" size={22} />
          <div>
            <strong>Working through your question</strong>
            <p>Understanding query · Retrieving evidence · Reranking · Generating answer</p>
          </div>
        </div>
      )}

      {error && (
        <div className="ask-error">
          <XCircle size={19} />
          <div>
            <strong>Request failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {result && (
        <div className="ask-results">
          <section className="ask-answer-panel">
            <div className="ask-section-heading">
              <div>
                <span className="eyebrow">GENERATED RESPONSE</span>
                <h2>Answer</h2>
              </div>
              <span className="ask-answer-status">
                <span />
                COMPLETE
              </span>
            </div>

            <div className="ask-answer-query">
              <span>QUERY</span>
              <p>{result.query}</p>
            </div>

            <div className="ask-answer-text">
              <ReactMarkdown>{result.answer.replace(/\\([*_`#[\]])/g, '$1')}</ReactMarkdown>
            </div>

            {result.generation.finish_reason === 'length' && (
              <div className="ask-generation-warning">
                <strong>Response may be incomplete.</strong>
                <p>
                  The model reached its output-token limit before completing generation. Consider
                  asking a narrower question.
                </p>
              </div>
            )}

            <div className="ask-answer-footer">
              <span>{result.sources.length} evidence sources</span>
              <span>{result.generation.model || 'Model information unavailable'}</span>
            </div>
          </section>

          <section className="ask-sources-section">
            <div className="ask-section-heading">
              <div>
                <span className="eyebrow">RETRIEVED EVIDENCE</span>
                <h2>Source passages</h2>
              </div>
              <span className="ask-source-count">
                {result.sources.length.toString().padStart(2, '0')} SOURCES
              </span>
            </div>

            {result.sources.length ? (
              <div className="ask-source-list">
                {result.sources.map((source) => (
                  <SourceCard key={source.chunk_id} source={source} />
                ))}
              </div>
            ) : (
              <p className="ask-no-sources">No source passages were returned for this answer.</p>
            )}
          </section>

          <Diagnostics result={result} />
        </div>
      )}

      {!result && !loading && !error && (
        <div className="ask-empty-state">
          <div className="ask-empty-icon">
            <FileText size={23} />
          </div>
          <strong>Your knowledge base is ready for questions.</strong>
          <p>Enter a question above to start retrieval and evidence-grounded generation.</p>
        </div>
      )}
    </main>
  );
}
