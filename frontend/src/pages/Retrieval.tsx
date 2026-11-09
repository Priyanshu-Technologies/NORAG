import { Play, TriangleAlert } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { noragApi, type QueryResult, type QuerySource } from '../api/client';
import './Retrieval.css';

const systems = [
  { value: 'hybrid', label: 'Hybrid (dense + sparse)' },
  { value: 'hybrid_rerank', label: 'Hybrid + reranking' },
  { value: 'dense', label: 'Dense only' },
  { value: 'sparse', label: 'Sparse only (BM25)' },
] as const;

function displayNumber(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function SourceRow({ source }: { source: QuerySource }) {
  const scores = source.retrieval_scores;

  return (
    <div className="retrieval-row">
      <span className="retrieval-source-id">{source.source_id}</span>
      <div className="retrieval-source-copy">
        <strong>{source.section || `Passage ${source.chunk_id.slice(0, 8)}`}</strong>
        <span>{source.text}</span>
      </div>
      <span>{source.page_numbers.length ? source.page_numbers.join(', ') : '—'}</span>
      <span>{source.token_count}</span>
      <span>{displayNumber(scores.hybrid_score)}</span>
    </div>
  );
}

export default function Retrieval() {
  const [system, setSystem] = useState<string>('hybrid');
  const [k, setK] = useState(5);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');

    if (!query.trim()) {
      setError('Enter a query to run through the retrieval pipeline.');
      return;
    }

    setRunning(true);

    try {
      const response = await noragApi.ask(query.trim(), k, system);
      setResult(response);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Retrieval failed.');
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="retrieval-page">
      <div className="page-header">
        <span className="eyebrow">RETRIEVAL LAB</span>
        <h1>Experiment with retrieval configurations</h1>
        <p className="page-description">
          Probe the retrieval stage directly: pick a strategy, set the depth, and inspect which
          passages were selected before generation runs.
        </p>
      </div>

      <form className="retrieval-form" onSubmit={handleSubmit}>
        <div className="retrieval-controls">
          <label className="retrieval-field">
            RETRIEVAL SYSTEM
            <select value={system} onChange={(event) => setSystem(event.target.value)}>
              {systems.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="retrieval-field">
            K
            <input
              type="number"
              min={1}
              max={50}
              value={k}
              onChange={(event) => setK(Number(event.target.value))}
            />
          </label>
        </div>

        <label className="retrieval-field">
          QUERY
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            spellCheck={false}
          />
        </label>

        <div className="retrieval-actions">
          <button type="submit" className="primary-button" disabled={running}>
            <Play size={15} />
            {running ? 'Retrieving…' : 'Run retrieval'}
          </button>
        </div>

        {error && (
          <div className="retrieval-error" role="alert">
            <TriangleAlert size={15} />
            {error}
          </div>
        )}
      </form>

      {result && (
        <section className="retrieval-results">
          <div className="section-heading">
            <div>
              <span className="eyebrow">RESULTS</span>
              <h2>
                {result.retrieval.returned_results}{' '}
                {result.retrieval.returned_results === 1 ? 'passage' : 'passages'} at k={k}
              </h2>
            </div>
          </div>

          <div className="retrieval-metrics">
            <div className="retrieval-metric">
              <span>STRATEGY</span>
              <strong className="is-text">{system}</strong>
            </div>
            <div className="retrieval-metric">
              <span>QUERY ROUTE</span>
              <strong className="is-text">{result.retrieval.route || '—'}</strong>
            </div>
            <div className="retrieval-metric">
              <span>WEIGHTS D / S</span>
              <strong className="is-text">
                {displayNumber(result.retrieval.dense_weight, 2)} /{' '}
                {displayNumber(result.retrieval.sparse_weight, 2)}
              </strong>
            </div>
            <div className="retrieval-metric">
              <span>CONTEXT TOKENS</span>
              <strong>{result.context.token_count.toLocaleString()}</strong>
            </div>
          </div>

          <div className="retrieval-answer-card">
            <span className="retrieval-answer-label">GROUNDED ANSWER</span>
            <div className="retrieval-answer">
              <ReactMarkdown>{result.answer || 'No answer was produced.'}</ReactMarkdown>
            </div>
          </div>

          <div className="retrieval-table">
            <div className="retrieval-row head">
              <span>#</span>
              <span>PASSAGE</span>
              <span>PAGE</span>
              <span>TOKENS</span>
              <span>HYBRID</span>
            </div>

            {result.sources.map((source) => (
              <SourceRow key={source.chunk_id} source={source} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
