import { BrowserRouter, Route, Routes } from 'react-router-dom';
import AppShell from './components/AppShell';
import Ask from './pages/Ask';
import Documents from './pages/Documents';
import Evaluation from './pages/Evaluation';
import NotFound from './pages/NotFound';
import Overview from './pages/Overview';
import Retrieval from './pages/Retrieval';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Overview />} />
          <Route path="documents" element={<Documents />} />
          <Route path="retrieval" element={<Retrieval />} />
          <Route path="ask" element={<Ask />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
