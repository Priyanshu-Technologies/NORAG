import { Play, TriangleAlert } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { noragApi, type EvaluationReport, type EvaluationSystem } from '../api/client';
import './Evaluation.css';

const systems: { value: EvaluationSystem; label: string }[] = [
  { value: 'hybrid', label: 'Hybrid (dense + sparse)' },
  { value: 'hybrid_rerank', label: 'Hybrid + reranking' },
  { value: 'dense', label: 'Dense only' },
  { value: 'sparse', label: 'Sparse only (BM25)' },
];

const sampleCases = JSON.stringify(
  [
    {
      query: 'How does hybrid retrieval combine dense and sparse signals?',
      relevant_chunk_ids: ['<chunk-id-from-the-documents-page>'],
    },
  ],
  null,
  2,
);

const metricLabels = [
  ['recall_at_k', 'Recall'],
  ['precision_at_k', 'Precision'],
  ['mrr_at_k', 'MRR'],
  ['ndcg_at_k', 'nDCG'],
] as const;

function asPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function Evaluation() {
  const [system, setSystem] = useState<EvaluationSystem>('hybrid');
  const [k, setK] = useState(5);
  const [casesText, setCasesText] = useState(sampleCases);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');

    let parsed: unknown;

    try {
      parsed = JSON.parse(casesText);
    } catch {
      setError('Cases must be valid JSON.');
      return;
    }

    if (!Array.isArray(parsed) || parsed.length === 0) {
      setError('Provide a JSON array with at least one evaluation case.');
      return;
    }

    const cases = parsed.map((item) => {
      const record = (item ?? {}) as Record<string, unknown>;
      const query = typeof record.query === 'string' ? record.query.trim() : '';
      const ids = Array.isArray(record.relevant_chunk_ids)
        ? record.relevant_chunk_ids.filter(
            (id): id is string => typeof id === 'string' && id.length > 0,
          )
        : [];

      return { query, relevant_chunk_ids: ids };
    });

    if (cases.some((item) => !item.query || item.relevant_chunk_ids.length === 0)) {
      setError('Every case needs a query and at least one relevant chunk ID.');
      return;
    }

    setRunning(true);

    try {
      const response = await noragApi.evaluate({ system, k, cases });
      setReport(response.report);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Evaluation failed.');
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="evaluation-page">
      <div className="page-header">
        <span className="eyebrow">EVALUATION</span>
        <h1>Measure retrieval quality</h1>
        <p className="page-description">
          Score recall, precision, MRR and nDCG for a labelled set of queries against the indexed
          knowledge base, per retrieval strategy.
        </p>
      </div>

      <form className="evaluation-form" onSubmit={handleSubmit}>
        <div className="evaluation-controls">
          <label className="evaluation-field">
            RETRIEVAL SYSTEM
            <select
              value={system}
              onChange={(event) => setSystem(event.target.value as EvaluationSystem)}
            >
              {systems.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="evaluation-field">
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

        <label className="evaluation-field">
          EVALUATION CASES (JSON)
          <textarea
            value={casesText}
            onChange={(event) => setCasesText(event.target.value)}
            rows={10}
            spellCheck={false}
          />
        </label>

        <div className="evaluation-actions">
          <button type="submit" className="primary-button" disabled={running}>
            <Play size={15} />
            {running ? 'Running evaluation…' : 'Run evaluation'}
          </button>
          <p className="evaluation-note">
            Relevant chunk IDs are the <code>chunk_id</code> values returned by the query endpoint.
          </p>
        </div>

        {error && (
          <div className="evaluation-error" role="alert">
            <TriangleAlert size={15} />
            {error}
          </div>
        )}
      </form>

      {report && (
        <section className="evaluation-results">
          <div className="section-heading">
            <div>
              <span className="eyebrow">RESULTS</span>
              <h2>
                {report.query_count} {report.query_count === 1 ? 'query' : 'queries'} at k=
                {report.k}
              </h2>
            </div>
          </div>

          <div className="evaluation-metrics">
            {metricLabels.map(([key, label]) => (
              <div className="evaluation-metric" key={key}>
                <span>{label}</span>
                <strong>{asPercent(report.average_metrics[key])}</strong>
              </div>
            ))}
          </div>

          <div className="evaluation-table">
            <div className="evaluation-row head">
              <span>QUERY</span>
              <span>RELEVANT</span>
              <span>RETRIEVED</span>
              <span>RECALL</span>
            </div>

            {report.per_query.map((row) => (
              <div className="evaluation-row" key={row.query}>
                <span className="evaluation-query">{row.query}</span>
                <span>{row.relevant_count}</span>
                <span>{row.retrieved_count}</span>
                <span>{asPercent(row.metrics.recall_at_k)}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
