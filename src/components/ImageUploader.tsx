import React, { useRef, useState } from 'react';
import { FileImage, ImagePlus, Trash2, UploadCloud } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File | null) => void;
  selectedFile: File | null;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, selectedFile }) => {
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const validateAndSelectFile = (file: File) => {
    const allowedTypes = ['image/jpeg', 'image/png'];
    const maxSize = 20 * 1024 * 1024;
    if (!allowedTypes.includes(file.type)) {
      setError('Choose a JPEG or PNG image. Other formats are not accepted.');
      return;
    }
    if (file.size > maxSize) {
      setError('Choose an image smaller than 20 MB.');
      return;
    }
    setError(null);
    onImageSelected(file);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) validateAndSelectFile(file);
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) validateAndSelectFile(file);
  };

  const clearFile = () => {
    onImageSelected(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div id="image-uploader-wrapper">
      <input ref={fileInputRef} id="input-file-upload" type="file" className="sr-only" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={handleChange} />

      {!previewUrl ? (
        <div
          id="uploader-dropzone"
          onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }}
          onDrop={handleDrop}
          className={`rounded-xl border-2 border-dashed px-6 py-9 text-center transition ${dragActive ? 'border-brand-500 bg-brand-50' : 'border-slate-300 bg-slate-50/70'}`}
        >
          <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-white text-brand-700 shadow-sm ring-1 ring-slate-200">
            <UploadCloud className="h-5 w-5" aria-hidden="true" />
          </span>
          <h3 className="mt-4 text-sm font-semibold text-slate-900">Add one ultrasound frame</h3>
          <p className="mx-auto mt-1 max-w-md text-sm leading-6 text-slate-500">Use a de-identified liver ultrasound screenshot in JPEG or PNG format, up to 20 MB.</p>
          <button type="button" onClick={() => fileInputRef.current?.click()} className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 shadow-sm transition hover:border-brand-300 hover:text-brand-700">
            <ImagePlus className="h-4 w-4" aria-hidden="true" />Choose image
          </button>
          <p className="mt-3 text-xs text-slate-400">or drag and drop it here</p>
        </div>
      ) : (
        <div id="uploader-preview-container" className="overflow-hidden rounded-xl border border-slate-300 bg-slate-950">
          <div className="flex items-center justify-between gap-3 border-b border-white/10 bg-slate-900 px-3 py-2.5 text-white">
            <div className="flex min-w-0 items-center gap-2"><FileImage className="h-4 w-4 shrink-0 text-cyan-300" /><span className="truncate text-xs font-medium">{selectedFile?.name}</span><span className="shrink-0 text-xs text-slate-400">{selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : ''}</span></div>
            <button id="btn-remove-image" type="button" onClick={clearFile} className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-slate-300 transition hover:bg-white/10 hover:text-white" aria-label="Remove selected image" title="Remove image"><Trash2 className="h-4 w-4" /></button>
          </div>
          <img src={previewUrl} alt="Selected ultrasound preview" className="mx-auto max-h-[380px] w-full object-contain" />
        </div>
      )}

      {error && <p className="mt-2 text-sm font-medium text-rose-700" role="alert">{error}</p>}
    </div>
  );
};
