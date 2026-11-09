const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api';
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

function withAuth(headers: HeadersInit = {}): HeadersInit {
  return API_KEY ? { ...headers, 'X-API-Key': API_KEY } : headers;
}

export interface NoragStats {
  documents: number;
  chunks: number;
  dense_chunks: number;
  sparse_chunks: number;
  queries: number;
  uploads: number;
  deletions: number;
  errors: number;
  environment: string;
  ollama_model: string;
}

export type EvaluationSystem = 'dense' | 'sparse' | 'hybrid' | 'hybrid_rerank';

export interface RankingMetrics {
  recall_at_k: number;
  precision_at_k: number;
  mrr_at_k: number;
  ndcg_at_k: number;
}

export interface QueryEvaluation {
  query: string;
  relevant_count: number;
  retrieved_count: number;
  metrics: RankingMetrics;
}

export interface EvaluationReport {
  k: number;
  query_count: number;
  average_metrics: RankingMetrics;
  per_query: QueryEvaluation[];
}

export interface EvaluationCaseInput {
  query: string;
  relevant_chunk_ids: string[];
}

export interface DocumentSummary {
  document_id: string;
  file_name: string | null;
  title: string | null;
  content_hash: string;
  created_at: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  file_name: string;
  content_hash: string;
  chunk_count: number;
}

interface DeleteDocumentResponse {
  document_id: string;
  deleted: boolean;
}

export interface QuerySource {
  source_id: string;
  document_id: string;
  chunk_id: string;
  text: string;
  section: string | null;
  page_numbers: number[];
  token_count: number;
  retrieval_scores: {
    hybrid_score: number | null;
    dense_score: number | null;
    sparse_score: number | null;
    rerank_score: number | null;
  };
}

export interface QueryResult {
  query: string;
  answer: string;
  sources: QuerySource[];
  retrieval: {
    intent: string;
    route: string;
    rule: string;
    dense_weight: number | null;
    sparse_weight: number | null;
    candidate_k: number | null;
    top_k: number | null;
    returned_results: number;
  };
  context: {
    token_count: number;
    available_tokens: number;
    included_count: number;
    excluded_count: number;
  };
  generation: {
    provider: string;
    model: string;
    input_tokens: number | null;
    output_tokens: number | null;
    finish_reason: string | null;
  };
  citation_validation: {
    total_sentences: number;
    cited_sentences: number;
    uncited_sentences: number;
    invalid_citation_count: number;
    citation_coverage: number;
  };
  claim_verification: {
    total_claims: number;
    supported_claims: number;
    contradicted_claims: number;
    insufficient_claims: number;
    support_rate: number;
  };
}

type AnyRecord = Record<string, unknown>;

function asRecord(value: unknown): AnyRecord {
  return value !== null && typeof value === 'object' ? (value as AnyRecord) : {};
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function normalizeQueryResponse(payload: unknown): QueryResult {
  const outer = asRecord(payload);
  const data = asRecord(outer.result ?? outer);

  // Support the curated public response format.
  if (Array.isArray(data.sources)) {
    const retrieval = asRecord(data.retrieval);
    const context = asRecord(data.context);
    const generation = asRecord(data.generation);
    const citations = asRecord(data.citation_validation);
    const claims = asRecord(data.claim_verification);

    return {
      query: asString(data.query),
      answer: asString(data.answer),
      sources: data.sources.map((item) => {
        const source = asRecord(item);
        const scores = asRecord(source.retrieval_scores);

        return {
          source_id: asString(source.source_id),
          document_id: asString(source.document_id),
          chunk_id: asString(source.chunk_id),
          text: asString(source.text),
          section: typeof source.section === 'string' ? source.section : null,
          page_numbers: Array.isArray(source.page_numbers)
            ? source.page_numbers.filter((page): page is number => typeof page === 'number')
            : [],
          token_count: asNumber(source.token_count),
          retrieval_scores: {
            hybrid_score: asNumber(scores.hybrid_score, NaN),
            dense_score: asNumber(scores.dense_score, NaN),
            sparse_score: asNumber(scores.sparse_score, NaN),
            rerank_score: asNumber(scores.rerank_score, NaN),
          },
        };
      }),
      retrieval: {
        intent: asString(retrieval.intent),
        route: asString(retrieval.route),
        rule: asString(retrieval.classification_rule),
        dense_weight: asNumber(retrieval.planned_dense_weight, NaN),
        sparse_weight: asNumber(retrieval.planned_sparse_weight, NaN),
        candidate_k: asNumber(retrieval.candidate_k, NaN),
        top_k: asNumber(retrieval.requested_top_k, NaN),
        returned_results: asNumber(retrieval.returned_results),
      },
      context: {
        token_count: asNumber(context.token_count),
        available_tokens: asNumber(context.available_tokens),
        included_count: asNumber(context.included_count),
        excluded_count: asNumber(context.excluded_count),
      },
      generation: {
        provider: asString(generation.provider),
        model: asString(generation.model),
        input_tokens: asNumber(generation.input_tokens, NaN),
        output_tokens: asNumber(generation.output_tokens, NaN),
        finish_reason:
          typeof generation.finish_reason === 'string' ? generation.finish_reason : null,
      },
      citation_validation: {
        total_sentences: asNumber(citations.total_sentences),
        cited_sentences: asNumber(citations.cited_sentences),
        uncited_sentences: asNumber(citations.uncited_sentences),
        invalid_citation_count: asNumber(citations.invalid_citation_count),
        citation_coverage: asNumber(citations.citation_coverage),
      },
      claim_verification: {
        total_claims: asNumber(claims.total_claims),
        supported_claims: asNumber(claims.supported_claims),
        contradicted_claims: asNumber(claims.contradicted_claims),
        insufficient_claims: asNumber(claims.insufficient_claims),
        support_rate: asNumber(claims.support_rate),
      },
    };
  }

  // Support the existing internal AnswerResponse format.
  const retrieval = asRecord(data.retrieval);
  const analysis = asRecord(retrieval.analysis);
  const plan = asRecord(retrieval.plan);
  const context = asRecord(data.context);
  const generation = asRecord(data.generation);
  const citations = asRecord(data.citation_validation);
  const claims = asRecord(data.claim_verification);
  const prompt = asRecord(data.prompt);

  const promptSources = Array.isArray(prompt.sources) ? prompt.sources.map(asRecord) : [];

  const sources: QuerySource[] = [];

  const contextItems = Array.isArray(context.items) ? context.items : [];

  for (const rawItem of contextItems) {
    const item = asRecord(rawItem);
    const rankedResult = asRecord(item.result);
    const hybridResult = asRecord(rankedResult.result);
    const chunk = asRecord(hybridResult.chunk);

    const chunkId = asString(chunk.id ?? chunk.chunk_id);
    const promptSource = promptSources.find((source) => asString(source.chunk_id) === chunkId);

    if (!promptSource || !chunkId) continue;

    const locations = Array.isArray(chunk.source_locations)
      ? chunk.source_locations.map(asRecord)
      : [];

    const pageNumbers = Array.isArray(promptSource.page_numbers)
      ? promptSource.page_numbers.filter((page): page is number => typeof page === 'number')
      : locations
          .map((location) => location.page_number)
          .filter((page): page is number => typeof page === 'number');

    sources.push({
      source_id: asString(promptSource.source_id),
      document_id: asString(chunk.document_id),
      chunk_id: chunkId,
      text: asString(chunk.text),
      section: typeof promptSource.section === 'string' ? promptSource.section : null,
      page_numbers: pageNumbers,
      token_count: asNumber(item.token_count, asNumber(chunk.token_count)),
      retrieval_scores: {
        hybrid_score: asNumber(hybridResult.score, NaN),
        dense_score: asNumber(hybridResult.dense_score, NaN),
        sparse_score: asNumber(hybridResult.sparse_score, NaN),
        rerank_score: asNumber(rankedResult.rerank_score, NaN),
      },
    });
  }

  const results = Array.isArray(retrieval.results) ? retrieval.results : [];

  return {
    query: asString(data.query),
    answer: asString(data.answer),
    sources,
    retrieval: {
      intent: asString(analysis.intent),
      route: asString(analysis.route),
      rule: asString(analysis.rule),
      dense_weight: asNumber(plan.dense_weight, NaN),
      sparse_weight: asNumber(plan.sparse_weight, NaN),
      candidate_k: asNumber(plan.candidate_k, NaN),
      top_k: asNumber(plan.final_top_k, NaN),
      returned_results: results.length,
    },
    context: {
      token_count: asNumber(context.token_count),
      available_tokens: asNumber(context.available_tokens),
      included_count: asNumber(context.included_count),
      excluded_count: asNumber(context.excluded_count),
    },
    generation: {
      provider: asString(generation.provider_name),
      model: asString(generation.model_name),
      input_tokens: asNumber(generation.input_tokens, NaN),
      output_tokens: asNumber(generation.output_tokens, NaN),
      finish_reason: typeof generation.finish_reason === 'string' ? generation.finish_reason : null,
    },
    citation_validation: {
      total_sentences: asNumber(citations.total_sentences),
      cited_sentences: asNumber(citations.cited_sentences),
      uncited_sentences: asNumber(citations.uncited_sentences),
      invalid_citation_count: asNumber(citations.invalid_citation_count),
      citation_coverage: asNumber(citations.citation_coverage),
    },
    claim_verification: {
      total_claims: asNumber(claims.total_claims),
      supported_claims: asNumber(claims.supported_claims),
      contradicted_claims: asNumber(claims.contradicted_claims),
      insufficient_claims: asNumber(claims.insufficient_claims),
      support_rate: asNumber(claims.support_rate),
    },
  };
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: withAuth(options.headers),
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const body = await response.json();
      message = body.detail ?? message;

      if (Array.isArray(message)) {
        message = message.map((item: { msg?: string }) => item.msg).join(', ');
      }
    } catch {
      // Keep the fallback message.
    }

    throw new Error(String(message));
  }

  return response.json() as Promise<T>;
}

export const noragApi = {
  listDocuments: () => request<DocumentSummary[]>('/documents'),

  uploadDocument: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    return request<DocumentUploadResponse>('/documents', {
      method: 'POST',
      body: formData,
    });
  },

  deleteDocument: (documentId: string) =>
    request<DeleteDocumentResponse>(`/documents/${documentId}`, {
      method: 'DELETE',
    }),

  ask: async (query: string, topK = 5, system?: string): Promise<QueryResult> => {
    const payload = await request<unknown>('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK, ...(system ? { system } : {}) }),
    });

    return normalizeQueryResponse(payload);
  },

  downloadDocument: async (documentId: string, fileName: string) => {
    const response = await fetch(`${API_BASE}/documents/${documentId}/file`, {
      headers: withAuth(),
    });

    if (!response.ok) {
      throw new Error(`Download failed (${response.status})`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');

    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    URL.revokeObjectURL(url);
  },

  getStats: () => request<NoragStats>('/stats'),

  evaluate: (payload: { system: EvaluationSystem; k: number; cases: EvaluationCaseInput[] }) =>
    request<{ report: EvaluationReport }>('/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};
