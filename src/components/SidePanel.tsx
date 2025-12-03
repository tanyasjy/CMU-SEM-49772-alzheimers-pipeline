import React from 'react';
import { X, Settings } from 'lucide-react';
import { NotebookSelector } from './NotebookSelector';
import { WorkspaceFiles } from './WorkspaceFiles';
import { SidebarSection } from './Sidebar';

interface Notebook {
  filename: string;
  size: number;
  modified_at: string;
  is_current: boolean;
}

interface SidePanelProps {
  activeSection: SidebarSection;
  onClose: () => void;
  // Notebook props
  notebooks?: Notebook[];
  currentNotebook?: string;
  onUploadNotebook?: (file: File) => Promise<void>;
  onSelectNotebook?: (filename: string) => Promise<void>;
  onDeleteNotebook?: (filename: string) => Promise<void>;
  onRefreshNotebooks?: () => Promise<void>;
}

export const SidePanel: React.FC<SidePanelProps> = ({
  activeSection,
  onClose,
  notebooks,
  currentNotebook,
  onUploadNotebook,
  onSelectNotebook,
  onDeleteNotebook,
  onRefreshNotebooks,
}) => {
  if (!activeSection) return null;

  const titles: Record<Exclude<SidebarSection, null>, string> = {
    notebooks: 'Notebooks',
    files: 'Workspace Files',
    settings: 'Settings',
  };

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h2 className="text-sm font-semibold text-gray-800">
          {titles[activeSection]}
        </h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-200 rounded transition-colors"
          title="Close panel"
          aria-label="Close"
        >
          <X className="w-4 h-4 text-gray-600" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeSection === 'notebooks' && (
          <NotebookSelector
            notebooks={notebooks || []}
            currentNotebook={currentNotebook || ''}
            onUpload={onUploadNotebook || (() => Promise.resolve())}
            onSelect={onSelectNotebook || (() => Promise.resolve())}
            onDelete={onDeleteNotebook || (() => Promise.resolve())}
            onRefresh={onRefreshNotebooks || (() => Promise.resolve())}
          />
        )}

        {activeSection === 'files' && (
          <WorkspaceFiles />
        )}

        {activeSection === 'settings' && (
          <div className="p-4">
            <div className="text-center py-8">
              <Settings className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Settings</h3>
              <p className="text-sm text-gray-600">Settings panel coming soon</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
