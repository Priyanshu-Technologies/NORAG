import { ArrowLeft, Construction } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const descriptions: Record<string, { title: string; description: string }> = {
  '/documents': {
    title: 'Document intelligence',
    description:
      'Manage your knowledge sources, inspect documents and control their indexing lifecycle.',
  },
  '/retrieval': {
    title: 'Retrieval laboratory',
    description: 'Explore query understanding, hybrid retrieval, ranking and evidence selection.',
  },
  '/ask': {
    title: 'Ask your knowledge',
    description: 'Interact with your documents through evidence-grounded question answering.',
  },
  '/evaluation': {
    title: 'Measure intelligence',
    description: 'Inspect retrieval quality, ranking performance and answer verification.',
  },
};

export default function WorkspacePlaceholder() {
  const location = useLocation();
  const page = descriptions[location.pathname];

  return (
    <div className="placeholder-page">
      <div className="placeholder-icon">
        <Construction size={25} />
      </div>

      <span className="eyebrow">WORKSPACE MODULE</span>
      <h1>{page?.title ?? 'Workspace'}</h1>
      <p>{page?.description}</p>

      <div className="placeholder-status">
        <span className="status-dot" />
        MODULE UNDER DEVELOPMENT
      </div>

      <Link to="/" className="secondary-button">
        <ArrowLeft size={15} />
        Return to overview
      </Link>
    </div>
  );
}
