import {
  ArrowDownRight,
  ArrowUpRight,
  Boxes,
  BrainCircuit,
  Database,
  FileText,
  GitBranch,
  Plus,
  Workflow,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { noragApi, type NoragStats } from '../api/client';

const pipeline = [
  {
    number: '01',
    name: 'Ingestion',
    description: 'Structure-aware document processing',
    icon: FileText,
  },
  {
    number: '02',
    name: 'Representation',
    description: 'Dense and sparse representations',
    icon: Boxes,
  },
  {
    number: '03',
    name: 'Retrieval',
    description: 'Hybrid search and reranking',
    icon: GitBranch,
  },
  {
    number: '04',
    name: 'Generation',
    description: 'Evidence-grounded responses',
    icon: BrainCircuit,
  },
];

export default function Overview() {
  const [stats, setStats] = useState<NoragStats | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;

    noragApi
      .getStats()
      .then((payload) => {
        if (!cancelled) setStats(payload);
      })
      .catch(() => {
        if (!cancelled) setUnavailable(true);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const format = (value: number | undefined) =>
    typeof value === 'number' ? value.toLocaleString() : '—';

  const snapshotLabel = stats
    ? `LIVE · ${stats.environment.toUpperCase()}`
    : unavailable
      ? 'SERVICE UNAVAILABLE'
      : 'LOADING…';

  return (
    <div className="overview">
      <section className="welcome-section">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-line" />
            RETRIEVAL INTELLIGENCE PLATFORM
          </div>

          <h1>
            Your knowledge.
            <br />
            <span>Connected.</span>
          </h1>

          <p className="welcome-description">
            A modular retrieval intelligence system built to turn unstructured information into
            grounded, traceable answers.
          </p>

          <div className="welcome-actions">
            <Link to="/documents" className="primary-button">
              <Plus size={17} />
              Add documents
              <ArrowUpRight size={15} />
            </Link>

            <Link to="/retrieval" className="secondary-button">
              Explore retrieval
              <ArrowUpRight size={15} />
            </Link>
          </div>
        </div>

        <div className="hero-visual">
          <div className="visual-orbit orbit-one" />
          <div className="visual-orbit orbit-two" />
          <div className="visual-orbit orbit-three" />

          <div className="visual-center">
            <div className="visual-center-inner">
              <Workflow size={35} strokeWidth={1.3} />
            </div>
          </div>

          <div className="visual-node node-top">
            <FileText size={15} />
          </div>
          <div className="visual-node node-right">
            <Database size={15} />
          </div>
          <div className="visual-node node-bottom">
            <BrainCircuit size={15} />
          </div>
          <div className="visual-node node-left">
            <GitBranch size={15} />
          </div>

          <span className="visual-label label-top">KNOWLEDGE</span>
          <span className="visual-label label-bottom">INTELLIGENCE</span>
        </div>
      </section>

      <section className="metrics-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">SYSTEM SNAPSHOT</span>
            <h2>Your knowledge base</h2>
          </div>
          <span className="snapshot-label">{snapshotLabel}</span>
        </div>

        <div className="metrics-grid">
          <MetricCard
            label="Documents"
            value={format(stats?.documents)}
            description="Total indexed documents"
            icon={FileText}
          />
          <MetricCard
            label="Indexed chunks"
            value={format(stats?.chunks)}
            description="Retrievable knowledge units"
            icon={Boxes}
          />
          <MetricCard
            label="Queries processed"
            value={format(stats?.queries)}
            description="Knowledge retrieval requests"
            icon={GitBranch}
          />
          <MetricCard
            label="Uploads"
            value={format(stats?.uploads)}
            description="Documents ingested this run"
            icon={Database}
          />
        </div>
      </section>

      <section className="architecture-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">UNDER THE HOOD</span>
            <h2>Retrieval architecture</h2>
          </div>
          <Link to="/retrieval" className="text-link">
            Explore system <ArrowUpRight size={15} />
          </Link>
        </div>

        <div className="pipeline">
          {pipeline.map(({ number, name, description, icon: Icon }, index) => (
            <div className="pipeline-stage" key={number}>
              <div className="pipeline-number">{number}</div>

              <div className="pipeline-icon">
                <Icon size={20} strokeWidth={1.6} />
              </div>

              <div className="pipeline-copy">
                <strong>{name}</strong>
                <span>{description}</span>
              </div>

              {index !== pipeline.length - 1 && (
                <div className="pipeline-connector">
                  <ArrowUpRight size={14} />
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="bottom-grid">
        <div className="panel getting-started">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">GET STARTED</span>
              <h3>Build your knowledge base</h3>
            </div>
            <span className="panel-icon">
              <ArrowDownRight size={18} />
            </span>
          </div>

          <p>
            Start by adding documents. NORAG will process their structure, create representations
            and make their content retrievable.
          </p>

          <Link to="/documents" className="text-link">
            Open document manager <ArrowUpRight size={15} />
          </Link>
        </div>

        <div className="panel system-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">SYSTEM DESIGN</span>
              <h3>Built for traceability</h3>
            </div>
            <span className="panel-icon">
              <GitBranch size={18} />
            </span>
          </div>

          <div className="principle-list">
            <Principle title="Hybrid retrieval" detail="Dense + sparse search" />
            <Principle title="Adaptive ranking" detail="Context-aware reranking" />
            <Principle title="Grounded generation" detail="Evidence-backed answers" />
            <Principle title="Verification" detail="Citation and claim checks" />
          </div>
        </div>
      </section>
    </div>
  );
}

function MetricCard({
  label,
  value,
  description,
  icon: Icon,
}: {
  label: string;
  value: string;
  description: string;
  icon: typeof FileText;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <Icon size={17} strokeWidth={1.6} />
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-description">{description}</span>
    </div>
  );
}

function Principle({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="principle">
      <div className="principle-indicator" />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}
