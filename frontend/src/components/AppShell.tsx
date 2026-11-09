import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Activity,
  ArrowUpRight,
  FileStack,
  FlaskConical,
  Layers3,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

const navigation = [
  { label: 'Overview', path: '/', icon: Layers3 },
  { label: 'Documents', path: '/documents', icon: FileStack },
  { label: 'Retrieval Lab', path: '/retrieval', icon: FlaskConical },
  { label: 'Ask NORAG', path: '/ask', icon: Sparkles },
  { label: 'Evaluation', path: '/evaluation', icon: Activity },
];

const pageTitles: Record<string, string> = {
  '/': 'Overview',
  '/documents': 'Documents',
  '/retrieval': 'Retrieval Lab',
  '/ask': 'Ask NORAG',
  '/evaluation': 'Evaluation',
};

export default function AppShell() {
  const location = useLocation();
  const pageTitle = pageTitles[location.pathname] ?? 'Not found';

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Layers3 size={19} strokeWidth={2.2} />
          </div>
          <div className="brand-copy">
            <span className="brand-name">NORAG</span>
            <span className="brand-version">KNOWLEDGE SYSTEMS</span>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="section-label">WORKSPACE</span>

          <nav className="navigation">
            {navigation.map(({ label, path, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === '/'}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Icon size={17} strokeWidth={1.8} />
                <span>{label}</span>
                {label === 'Ask NORAG' && <span className="nav-shortcut">↵</span>}
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <div className="note-icon">
              <ShieldCheck size={17} />
            </div>
            <div>
              <strong>Evidence first.</strong>
              <p>Answers grounded in your knowledge.</p>
            </div>
          </div>

          <a
            className="sidebar-footer profile-link"
            href="https://www.linkedin.com/in/priyanshu-sharma-646233394/"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open Priyanshu Sharma's LinkedIn profile"
          >
            <div className="avatar">P</div>
            <div className="profile-copy">
              <strong>Developer</strong>
              <span>LinkedIn profile</span>
            </div>
            <ArrowUpRight size={15} className="muted-icon" />
          </a>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{pageTitle}</strong>
          </div>

          <div className="topbar-actions">
            <div className="environment-indicator">
              <span className="status-dot" />
            </div>
          </div>
        </header>

        <div className="page-container">
          <Outlet />
        </div>

        <footer className="app-footer">
          <span>NORAG</span>
          <span>BEYOND ORDINARY RETRIEVAL.</span>
          <span>V1.0.0</span>
        </footer>
      </main>
    </div>
  );
}
