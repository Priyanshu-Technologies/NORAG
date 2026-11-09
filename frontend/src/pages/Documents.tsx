import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  File,
  FileText,
  FileUp,
  LoaderCircle,
  RefreshCw,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { noragApi, type DocumentSummary } from '../api/client';

const ACCEPTED_FILES = '.pdf,.txt,.md,.html';

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

function getFileExtension(fileName: string) {
  return fileName.split('.').pop()?.toUpperCase() ?? 'FILE';
}

export default function Documents() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busyDocument, setBusyDocument] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await noragApi.listDocuments();
      setDocuments(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load documents.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const fetchDocuments = async () => {
      try {
        const result = await noragApi.listDocuments();

        if (!cancelled) {
          setDocuments(result);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unable to load documents.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void fetchDocuments();

    return () => {
      cancelled = true;
    };
  }, []);

  const uploadFile = async (file?: File) => {
    if (!file || uploading) return;

    setUploading(true);
    setError(null);
    setNotice(null);

    try {
      const result = await noragApi.uploadDocument(file);
      setNotice(
        `Uploaded ${result.file_name} successfully · ${result.chunk_count} chunks created.`,
      );
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setUploading(false);

      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  };

  const handleDelete = async (document: DocumentSummary) => {
    const name = document.file_name ?? document.title ?? 'this document';

    if (!window.confirm(`Delete "${name}" from NORAG? This cannot be undone.`)) {
      return;
    }

    setBusyDocument(document.document_id);
    setError(null);
    setNotice(null);

    try {
      await noragApi.deleteDocument(document.document_id);
      setDocuments((current) =>
        current.filter((item) => item.document_id !== document.document_id),
      );
      setNotice(`${name} was deleted.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete document.');
    } finally {
      setBusyDocument(null);
    }
  };

  const handleDownload = async (document: DocumentSummary) => {
    const name = document.file_name ?? 'document';

    setBusyDocument(document.document_id);
    setError(null);

    try {
      await noragApi.downloadDocument(document.document_id, name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to download document.');
    } finally {
      setBusyDocument(null);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void uploadFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="documents-page">
      <div className="documents-heading">
        <div>
          <span className="eyebrow">KNOWLEDGE SOURCES</span>
          <h1>Documents</h1>
          <p>Manage the documents that power your retrieval and answer generation.</p>
        </div>

        <button
          className="documents-refresh"
          type="button"
          onClick={() => void loadDocuments()}
          disabled={loading}
          title="Refresh documents"
        >
          <RefreshCw size={16} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      <div
        className={`documents-dropzone${dragging ? ' is-dragging' : ''}${uploading ? ' is-uploading' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setDragging(false);
          }
        }}
        onDrop={handleDrop}
      >
        <div className="documents-upload-icon">
          {uploading ? <LoaderCircle className="spin" size={25} /> : <UploadCloud size={25} />}
        </div>

        <div className="documents-upload-copy">
          <h2>{uploading ? 'Processing document…' : 'Add documents to your knowledge base'}</h2>
          <p>
            {uploading
              ? 'NORAG is ingesting, chunking and indexing your file.'
              : 'Drag and drop a file here, or choose one from your computer.'}
          </p>
          <span>PDF, TXT, Markdown and HTML · Maximum file size 25 MB</span>
        </div>

        <button
          className="documents-upload-button"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          <FileUp size={16} />
          Choose file
        </button>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_FILES}
          hidden
          onChange={(event) => void uploadFile(event.target.files?.[0])}
        />
      </div>

      {error && (
        <div className="documents-message documents-error" role="alert">
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      {notice && (
        <div className="documents-message documents-success" role="status">
          <CheckCircle2 size={17} />
          <span>{notice}</span>
        </div>
      )}

      <div className="documents-list-heading">
        <div>
          <span className="eyebrow">YOUR LIBRARY</span>
          <h2>Indexed documents</h2>
        </div>
        <span className="documents-count">
          {documents.length} {documents.length === 1 ? 'DOCUMENT' : 'DOCUMENTS'}
        </span>
      </div>

      <div className="documents-table">
        <div className="documents-table-header">
          <span>DOCUMENT</span>
          <span>ADDED</span>
          <span>FORMAT</span>
          <span>ACTIONS</span>
        </div>

        {loading ? (
          <div className="documents-empty">
            <LoaderCircle className="spin" size={22} />
            <span>Loading your document library…</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="documents-empty">
            <FileText size={25} />
            <strong>Your knowledge base is empty</strong>
            <span>Upload your first document to get started.</span>
          </div>
        ) : (
          documents.map((document) => {
            const fileName = document.file_name ?? document.title ?? 'Untitled document';
            const busy = busyDocument === document.document_id;

            return (
              <div className="documents-table-row" key={document.document_id}>
                <div className="documents-file-cell">
                  <div className="documents-file-icon">
                    <File size={19} />
                  </div>
                  <div className="documents-file-info">
                    <strong title={fileName}>{fileName}</strong>
                    <span>{document.title ?? 'Untitled document'}</span>
                  </div>
                </div>

                <span className="documents-date">{formatDate(document.created_at)}</span>

                <span className="documents-format">{getFileExtension(fileName)}</span>

                <div className="documents-actions">
                  <button
                    type="button"
                    title="Download original"
                    aria-label={`Download ${fileName}`}
                    onClick={() => void handleDownload(document)}
                    disabled={busy}
                  >
                    <Download size={16} />
                  </button>
                  <button
                    type="button"
                    title="Delete document"
                    aria-label={`Delete ${fileName}`}
                    className="documents-delete"
                    onClick={() => void handleDelete(document)}
                    disabled={busy}
                  >
                    {busy ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
