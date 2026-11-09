import { ArrowLeft, Compass } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="placeholder-page">
      <div className="placeholder-icon">
        <Compass size={25} strokeWidth={1.6} />
      </div>

      <span className="eyebrow">404</span>
      <h1>Page not found</h1>
      <p>The page you requested does not exist in this workspace.</p>

      <Link to="/" className="secondary-button">
        <ArrowLeft size={15} />
        Return to overview
      </Link>
    </div>
  );
}
