import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/router';

export default function Upload() {
  const router = useRouter();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [selectedTextColumns, setSelectedTextColumns] = useState([]);
  const [showColumnSelector, setShowColumnSelector] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    setSelectedFile(file);
    setError('');
    setUploadResult(null);
    
    // Parse CSV to extract headers
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n');
      if (lines.length > 0) {
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        setCsvHeaders(headers);
        setShowColumnSelector(true);
        // Auto-select text-like columns
        const textColumns = headers.filter(h => 
          h.toLowerCase().includes('text') || 
          h.toLowerCase().includes('review') || 
          h.toLowerCase().includes('comment') ||
          h.toLowerCase().includes('feedback') ||
          h.toLowerCase().includes('like') ||
          h.toLowerCase().includes('dislike') ||
          h.toLowerCase().includes('opinion')
        );
        setSelectedTextColumns(textColumns.length > 0 ? textColumns : [headers[0]]);
      }
    };
    reader.readAsText(file);
  };

  const toggleColumn = (column) => {
    setSelectedTextColumns(prev => {
      if (prev.includes(column)) {
        return prev.filter(c => c !== column);
      } else {
        return [...prev, column];
      }
    });
  };

  const uploadBatch = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    if (!selectedTextColumns || selectedTextColumns.length === 0) {
      setError('Please select at least one text column to analyze');
      return;
    }

    setUploading(true);
    setError('');

    try {
      // Read file content
      const reader = new FileReader();
      reader.onload = async (e) => {
        const csvData = e.target.result;

        // Send to API
        const apiUrl = process.env.NODE_ENV === 'production'
          ? '/api/batch-upload'
          : 'http://localhost:8000/api/batch-upload';
        
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            csv_data: csvData,
            filename: selectedFile.name,
            mode: 'business-insights',
            text_columns: selectedTextColumns
          }),
        });

        const result = await response.json();

        if (response.ok) {
          setUploadResult(result);
          setSelectedFile(null);
          setShowColumnSelector(false);
          setCsvHeaders([]);
          setSelectedTextColumns([]);
        } else {
          setError(result.error || 'Upload failed');
        }
        setUploading(false);
      };

      reader.onerror = () => {
        setError('Failed to read file');
        setUploading(false);
      };

      reader.readAsText(selectedFile);
    } catch (err) {
      setError('Upload failed: ' + err.message);
      setUploading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #FF512F 0%, #DD2476 100%)',
      padding: '40px 20px'
    }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto'
      }}>
        {/* Header */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '30px',
          marginBottom: '30px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h1 style={{ margin: 0, fontSize: '28px', color: '#333' }}>
              📊 Batch Upload
            </h1>
            <button
              onClick={() => router.push('/')}
              style={{
                padding: '10px 20px',
                background: '#FF512F',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '600'
              }}
            >
              ← Back
            </button>
          </div>
          <p style={{ color: '#666', marginTop: '10px', marginBottom: 0 }}>
            Upload a CSV file with customer reviews for batch sentiment analysis
          </p>
        </div>

        {/* Upload Area */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '40px',
          marginBottom: '30px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ marginTop: 0, fontSize: '20px', color: '#333' }}>Upload CSV File</h2>
          
          {/* Drag and Drop Zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            style={{
              border: dragActive ? '3px dashed #FF512F' : '3px dashed #ddd',
              borderRadius: '12px',
              padding: '60px 20px',
              textAlign: 'center',
              background: dragActive ? '#FFF5F5' : '#FAFAFA',
              transition: 'all 0.3s ease',
              marginBottom: '20px',
              cursor: 'pointer'
            }}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>
              {selectedFile ? '✓' : '📁'}
            </div>
            {selectedFile ? (
              <div>
                <p style={{ fontSize: '18px', color: '#333', fontWeight: '600', marginBottom: '10px' }}>
                  {selectedFile.name}
                </p>
                <p style={{ color: '#666', fontSize: '14px' }}>
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            ) : (
              <div>
                <p style={{ fontSize: '18px', color: '#333', fontWeight: '600', marginBottom: '10px' }}>
                  Drop CSV file here or click to browse
                </p>
                <p style={{ color: '#666', fontSize: '14px' }}>
                  Supported format: CSV with columns (date, location, product, review_text, rating)
                </p>
              </div>
            )}
            <input
              id="fileInput"
              type="file"
              accept=".csv"
              onChange={handleChange}
              style={{ display: 'none' }}
            />
          </div>

          {/* Column Selector - Show after file is selected */}
          {showColumnSelector && csvHeaders.length > 0 && (
            <div style={{
              padding: '20px',
              background: '#F0FDF4',
              borderRadius: '8px',
              border: '1px solid #86EFAC',
              marginBottom: '20px'
            }}>
              <label style={{ 
                display: 'block',
                fontSize: '14px', 
                fontWeight: '600',
                color: '#166534',
                marginBottom: '12px'
              }}>
                📝 Select column(s) containing review text to analyze:
              </label>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                gap: '10px',
                marginBottom: '12px'
              }}>
                {csvHeaders.map((header, idx) => (
                  <div
                    key={idx}
                    onClick={() => toggleColumn(header)}
                    style={{
                      padding: '12px',
                      background: selectedTextColumns.includes(header) ? '#86EFAC' : 'white',
                      border: selectedTextColumns.includes(header) ? '2px solid #166534' : '1px solid #D1D5DB',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={selectedTextColumns.includes(header)}
                      onChange={() => {}}
                      style={{
                        width: '18px',
                        height: '18px',
                        cursor: 'pointer'
                      }}
                    />
                    <span style={{
                      fontSize: '13px',
                      fontWeight: selectedTextColumns.includes(header) ? '600' : '400',
                      color: selectedTextColumns.includes(header) ? '#166534' : '#374151',
                      wordBreak: 'break-word'
                    }}>
                      {header}
                    </span>
                  </div>
                ))}
              </div>
              <p style={{ fontSize: '12px', color: '#166534', marginTop: '8px', marginBottom: 0 }}>
                ✓ {selectedTextColumns.length} column{selectedTextColumns.length !== 1 ? 's' : ''} selected • Text from selected columns will be combined for analysis
              </p>
            </div>
          )}

          {/* Info Notice */}
          <div style={{
            padding: '12px 16px',
            background: '#EFF6FF',
            borderRadius: '8px',
            border: '1px solid #BFDBFE',
            fontSize: '13px',
            color: '#1E40AF',
            marginBottom: '20px'
          }}>
            ℹ️ <strong>Note:</strong> Previous data will be automatically cleared before processing new uploads
          </div>

          {/* Upload Button */}
          <button
            onClick={uploadBatch}
            disabled={!selectedFile || uploading}
            style={{
              width: '100%',
              padding: '16px',
              background: selectedFile && !uploading 
                ? 'linear-gradient(135deg, #FF512F 0%, #DD2476 100%)' 
                : '#CCC',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: selectedFile && !uploading ? 'pointer' : 'not-allowed',
              transition: 'all 0.3s ease'
            }}
          >
            {uploading ? '⏳ Processing...' : '🚀 Upload and Analyze'}
          </button>

          {/* Error Message */}
          {error && (
            <div style={{
              marginTop: '20px',
              padding: '15px',
              background: '#FEE',
              border: '1px solid #FCC',
              borderRadius: '8px',
              color: '#C33'
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Upload Result */}
        {uploadResult && (
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '40px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ marginTop: 0, fontSize: '20px', color: '#333' }}>✓ Upload Complete</h2>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '20px',
              marginBottom: '30px'
            }}>
              <div style={{
                padding: '20px',
                background: '#F0F9FF',
                borderRadius: '8px',
                border: '1px solid #BFDBFE'
              }}>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#1E40AF' }}>
                  {uploadResult.total_rows}
                </div>
                <div style={{ color: '#666', marginTop: '5px' }}>Total Reviews</div>
              </div>
              
              <div style={{
                padding: '20px',
                background: '#F0FDF4',
                borderRadius: '8px',
                border: '1px solid #BBF7D0'
              }}>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#15803D' }}>
                  {uploadResult.processed_rows}
                </div>
                <div style={{ color: '#666', marginTop: '5px' }}>Processed</div>
              </div>
            </div>

            {/* Sample Results */}
            {uploadResult.results && uploadResult.results.length > 0 && (
              <div>
                <h3 style={{ fontSize: '16px', color: '#333', marginBottom: '15px' }}>
                  Sample Results (First 10)
                </h3>
                <div style={{
                  maxHeight: '300px',
                  overflowY: 'auto',
                  border: '1px solid #E5E7EB',
                  borderRadius: '8px'
                }}>
                  {uploadResult.results.map((result, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '15px',
                        borderBottom: idx < uploadResult.results.length - 1 ? '1px solid #E5E7EB' : 'none',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}
                    >
                      <div style={{ color: '#666', fontSize: '14px' }}>
                        Review #{result.review_id}
                      </div>
                      <div style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '600',
                        background: result.sentiment === 'positive' ? '#DCFCE7' : 
                                   result.sentiment === 'negative' ? '#FEE2E2' : '#FEF3C7',
                        color: result.sentiment === 'positive' ? '#15803D' : 
                               result.sentiment === 'negative' ? '#991B1B' : '#92400E'
                      }}>
                        {result.sentiment}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* View Dashboard Button */}
            <button
              onClick={() => router.push('/dashboard')}
              style={{
                width: '100%',
                marginTop: '30px',
                padding: '16px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              📊 View Dashboard →
            </button>
          </div>
        )}

        {/* CSV Format Guide */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '30px',
          marginTop: '30px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
        }}>
          <h3 style={{ marginTop: 0, fontSize: '18px', color: '#333' }}>📋 Flexible CSV Format</h3>
          <p style={{ color: '#666', marginBottom: '20px' }}>
            ✨ <strong>Smart Column Detection!</strong> Upload any sentiment dataset - the system automatically detects column types.
          </p>

          {/* Supported Formats */}
          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ fontSize: '15px', color: '#333', marginBottom: '10px' }}>✅ Supported Formats</h4>
            <div style={{ fontSize: '13px', color: '#666', lineHeight: '1.8' }}>
              <div>• <strong>IMDB Dataset</strong>: review, sentiment</div>
              <div>• <strong>Twitter Sentiment140</strong>: text, target/sentiment</div>
              <div>• <strong>Amazon Reviews</strong>: reviewText, overall/rating</div>
              <div>• <strong>Yelp Reviews</strong>: text, stars</div>
              <div>• <strong>Custom Datasets</strong>: Any CSV with text content</div>
            </div>
          </div>

          {/* Example 1: Recommended Format */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '14px', color: '#333', marginBottom: '8px' }}>
              📌 Recommended Format (Full Features)
            </h4>
            <div style={{
              background: '#F0F9FF',
              border: '1px solid #BFDBFE',
              borderRadius: '8px',
              padding: '12px',
              fontFamily: 'monospace',
              fontSize: '12px',
              overflowX: 'auto'
            }}>
              <div>date,location,product,review_text,rating</div>
              <div style={{ color: '#666', marginTop: '4px' }}>
                2024-12-01,Seattle,Burger,"Great burger!",5
              </div>
            </div>
          </div>

          {/* Example 2: Minimal Format */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '14px', color: '#333', marginBottom: '8px' }}>
              📌 Minimal Format (Text Only)
            </h4>
            <div style={{
              background: '#F0FDF4',
              border: '1px solid #BBF7D0',
              borderRadius: '8px',
              padding: '12px',
              fontFamily: 'monospace',
              fontSize: '12px',
              overflowX: 'auto'
            }}>
              <div>text</div>
              <div style={{ color: '#666', marginTop: '4px' }}>
                "This product is amazing!"
              </div>
              <div style={{ color: '#666' }}>
                "Worst purchase ever"
              </div>
            </div>
          </div>

          {/* Example 3: Common Datasets */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '14px', color: '#333', marginBottom: '8px' }}>
              📌 Common Dataset Formats
            </h4>
            <div style={{
              background: '#FEF3C7',
              border: '1px solid #FDE68A',
              borderRadius: '8px',
              padding: '12px',
              fontFamily: 'monospace',
              fontSize: '12px',
              overflowX: 'auto'
            }}>
              <div style={{ marginBottom: '8px' }}>
                <strong style={{ color: '#92400E' }}>IMDB:</strong> review,sentiment
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong style={{ color: '#92400E' }}>Twitter:</strong> tweet,sentiment
              </div>
              <div>
                <strong style={{ color: '#92400E' }}>Amazon:</strong> reviewText,overall
              </div>
            </div>
          </div>

          {/* Auto-Detection Info */}
          <div style={{
            background: '#F3F4F6',
            border: '1px solid #D1D5DB',
            borderRadius: '8px',
            padding: '15px',
            marginTop: '20px'
          }}>
            <div style={{ fontSize: '13px', color: '#374151', lineHeight: '1.6' }}>
              <strong>🤖 Auto-Detection:</strong> The system recognizes these column variations:
              <ul style={{ marginTop: '8px', marginBottom: 0, paddingLeft: '20px' }}>
                <li><strong>Text:</strong> review_text, text, review, comment, feedback, tweet, content</li>
                <li><strong>Date:</strong> date, timestamp, created_at, review_date</li>
                <li><strong>Rating:</strong> rating, stars, score, overall</li>
                <li><strong>Location:</strong> location, city, branch, store</li>
                <li><strong>Product:</strong> product, item, category, service</li>
              </ul>
            </div>
          </div>

          {/* Tips */}
          <div style={{ marginTop: '20px', fontSize: '13px', color: '#666' }}>
            <strong style={{ color: '#333' }}>💡 Tips:</strong>
            <ul style={{ marginTop: '8px', marginBottom: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
              <li>Only the text/review column is required - all others are optional</li>
              <li>Use any column names - the system will detect them automatically</li>
              <li>Works with datasets from Kaggle, UCI, Hugging Face, or custom sources</li>
              <li>No preprocessing needed - just upload your raw CSV!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
