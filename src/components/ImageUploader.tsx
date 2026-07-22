import React, { useState, useRef } from 'react';
import { Upload, X, FileImage, ShieldCheck } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File | null) => void;
  selectedFile: File | null;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, selectedFile }) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate preview when file is selected
  React.useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndSelectFile = (file: File) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!allowedTypes.includes(file.type)) {
      alert('Unsupported file format. Please upload JPG, PNG, or BMP.');
      return;
    }

    if (file.size > maxSize) {
      alert('File exceeds the 50MB limit. Please upload a smaller image.');
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelected(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full" id="image-uploader-wrapper">
      <input
        ref={fileInputRef}
        type="file"
        id="input-file-upload"
        className="hidden"
        accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
        onChange={handleChange}
      />

      {!previewUrl ? (
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={onButtonClick}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all bg-slate-50/50 min-h-[260px] flex flex-col items-center justify-center ${
            dragActive
              ? 'border-blue-500 bg-blue-50/70 scale-[1.01]'
              : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50'
          }`}
          id="uploader-dropzone"
        >
          <div className="flex flex-col items-center justify-center">
            <div className="w-14 h-14 bg-blue-50 rounded-full flex items-center justify-center mb-4 border border-blue-100/50 shadow-sm text-blue-500">
              <Upload className="h-6 w-6 stroke-[2.2]" />
            </div>
            
            <h3 className="text-slate-800 font-bold text-sm mb-1">New Ultrasound Analysis</h3>
            <p className="text-slate-500 text-xs max-w-xs mb-4 leading-relaxed">
              Drag and drop your liver ultrasound image here or click to browse. (DICOM, JPG, PNG, or BMP supported up to 50MB).
            </p>

            <div className="flex items-center space-x-1.5 text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-100/60 px-2.5 py-1 rounded-full font-semibold">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
              <span>Medical-Grade Diagnostic Input</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative border border-slate-200 rounded-xl overflow-hidden bg-slate-900 shadow-inner group" id="uploader-preview-container">
          <img
            src={previewUrl}
            alt="Ultrasound Preview"
            className="w-full max-h-[350px] object-contain mx-auto"
            referrerPolicy="no-referrer"
          />
          
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
            <div className="flex items-center space-x-2 text-white">
              <FileImage className="h-4 w-4 text-blue-400" />
              <span className="text-xs font-mono font-medium truncate max-w-xs">{selectedFile?.name}</span>
              <span className="text-[10px] bg-white/20 px-1.5 py-0.5 rounded">
                {selectedFile ? (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB' : ''}
              </span>
            </div>
          </div>

          <button
            onClick={handleClear}
            className="absolute top-3 right-3 p-2 bg-black/60 hover:bg-red-600 text-white rounded-full transition-colors cursor-pointer"
            title="Remove Image"
            id="btn-remove-image"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
};
