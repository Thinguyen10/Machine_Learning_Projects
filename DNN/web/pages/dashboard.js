import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

export default function Dashboard() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({
    location: null,
    days: 30
  });

  useEffect(() => {
    fetchDashboardData();
  }, [filters]);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError('');

    try {
      const apiUrl = process.env.NODE_ENV === 'production'
        ? '/api/dashboard'
        : 'http://localhost:8000/api/dashboard';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filters),
      });

      const data = await response.json();

      if (response.ok) {
        setDashboardData(data);
      } else {
        setError(data.error || 'Failed to load dashboard');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    const colors = {
      'positive': '#10B981',
      'negative': '#EF4444',
      'neutral': '#F59E0B'
    };
    return colors[sentiment?.toLowerCase()] || '#6B7280';
  };

  const getSentimentBg = (sentiment) => {
    const colors = {
      'positive': '#D1FAE5',
      'negative': '#FEE2E2',
      'neutral': '#FEF3C7'
    };
    return colors[sentiment?.toLowerCase()] || '#F3F4F6';
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '40px 20px'
    }}>
      <div style={{
        maxWidth: '1200px',
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
            <div>
              <h1 style={{ margin: 0, fontSize: '28px', color: '#333' }}>
                📈 Analytics Dashboard
              </h1>
              <p style={{ color: '#666', marginTop: '10px', marginBottom: 0 }}>
                Real-time sentiment analysis insights and trends
              </p>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={() => router.push('/')}
                style={{
                  padding: '10px 20px',
                  background: '#667eea',
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
              <button
                onClick={fetchDashboardData}
                style={{
                  padding: '10px 20px',
                  background: '#10B981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600'
                }}
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '20px 30px',
          marginBottom: '30px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ flex: '1', minWidth: '200px' }}>
              <label style={{ display: 'block', color: '#666', fontSize: '14px', marginBottom: '8px' }}>
                Time Period
              </label>
              <select
                value={filters.days}
                onChange={(e) => setFilters({ ...filters, days: parseInt(e.target.value) })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #E5E7EB',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              >
                <option value={7}>Last 7 days</option>
                <option value={30}>Last 30 days</option>
                <option value={90}>Last 90 days</option>
              </select>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '60px',
            textAlign: 'center',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
            <p style={{ color: '#666', fontSize: '18px' }}>Loading dashboard data...</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '40px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
          }}>
            <div style={{
              padding: '20px',
              background: '#FEE2E2',
              border: '1px solid #FCA5A5',
              borderRadius: '8px',
              color: '#991B1B'
            }}>
              ⚠️ {error}
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {!loading && !error && dashboardData && (
          <>
            {/* Statistics Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '30px'
            }}>
              {/* Total Reviews */}
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
                  Total Reviews
                </div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#333' }}>
                  {dashboardData.statistics?.total_reviews || 0}
                </div>
              </div>

              {/* Positive */}
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
                  Positive
                </div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#10B981' }}>
                  {dashboardData.statistics?.positive_count || 0}
                </div>
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
                  {dashboardData.statistics?.positive_percentage || 0}%
                </div>
              </div>

              {/* Negative */}
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
                  Negative
                </div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#EF4444' }}>
                  {dashboardData.statistics?.negative_count || 0}
                </div>
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
                  {dashboardData.statistics?.negative_percentage || 0}%
                </div>
              </div>

              {/* Neutral */}
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
                  Neutral
                </div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#F59E0B' }}>
                  {dashboardData.statistics?.neutral_count || 0}
                </div>
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
                  {dashboardData.statistics?.neutral_percentage || 0}%
                </div>
              </div>
            </div>

            {/* Sentiment Trends Time Series Chart */}
            {dashboardData.trends && dashboardData.trends.length > 0 && (() => {
              const maxPct = 100; // Percentage goes from 0 to 100
              const chartHeight = 300;
              const chartWidth = 800;
              const padding = { top: 20, right: 60, bottom: 60, left: 50 };
              const plotWidth = chartWidth - padding.left - padding.right;
              const plotHeight = chartHeight - padding.top - padding.bottom;
              
              // Create points for each sentiment line using percentages
              const createPoints = (dataKey) => {
                return dashboardData.trends.map((trend, idx) => {
                  const x = padding.left + (idx / (dashboardData.trends.length - 1)) * plotWidth;
                  const y = padding.top + plotHeight - (trend[dataKey] / maxPct) * plotHeight;
                  return { x, y, value: trend[dataKey], date: trend.date };
                });
              };
              
              const positivePoints = createPoints('positive_pct');
              const negativePoints = createPoints('negative_pct');
              const neutralPoints = createPoints('neutral_pct');
              
              // Create SVG path from points
              const createPath = (points) => {
                if (points.length === 0) return '';
                let path = `M ${points[0].x} ${points[0].y}`;
                for (let i = 1; i < points.length; i++) {
                  path += ` L ${points[i].x} ${points[i].y}`;
                }
                return path;
              };
              
              return (
                <div style={{
                  background: 'white',
                  borderRadius: '16px',
                  padding: '30px',
                  marginBottom: '30px',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
                }}>
                  <h2 style={{ marginTop: 0, fontSize: '20px', color: '#333', marginBottom: '10px' }}>
                    📈 Sentiment Trends Over Time
                  </h2>
                  <p style={{ color: '#666', fontSize: '14px', marginBottom: '20px' }}>
                    Percentage distribution of sentiment over time
                  </p>
                  
                  {/* Legend */}
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '20px', justifyContent: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ width: '30px', height: '3px', background: '#10B981', borderRadius: '2px' }} />
                      <span style={{ fontSize: '14px', color: '#666' }}>Positive</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ width: '30px', height: '3px', background: '#F59E0B', borderRadius: '2px' }} />
                      <span style={{ fontSize: '14px', color: '#666' }}>Neutral</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ width: '30px', height: '3px', background: '#EF4444', borderRadius: '2px' }} />
                      <span style={{ fontSize: '14px', color: '#666' }}>Negative</span>
                    </div>
                  </div>
                  
                  {/* SVG Chart */}
                  <div style={{ overflowX: 'auto' }}>
                    <svg width={chartWidth} height={chartHeight} style={{ display: 'block', margin: '0 auto' }}>
                      {/* Y-axis gridlines */}
                      {[0, 25, 50, 75, 100].map((pct, idx) => {
                        const y = padding.top + plotHeight - (pct / 100) * plotHeight;
                        return (
                          <g key={idx}>
                            <line
                              x1={padding.left}
                              y1={y}
                              x2={padding.left + plotWidth}
                              y2={y}
                              stroke="#E5E7EB"
                              strokeWidth="1"
                            />
                            <text
                              x={padding.left - 10}
                              y={y + 4}
                              textAnchor="end"
                              fontSize="11"
                              fill="#9CA3AF"
                            >
                              {pct}%
                            </text>
                          </g>
                        );
                      })}
                      
                      {/* X-axis */}
                      <line
                        x1={padding.left}
                        y1={padding.top + plotHeight}
                        x2={padding.left + plotWidth}
                        y2={padding.top + plotHeight}
                        stroke="#9CA3AF"
                        strokeWidth="2"
                      />
                      
                      {/* Y-axis */}
                      <line
                        x1={padding.left}
                        y1={padding.top}
                        x2={padding.left}
                        y2={padding.top + plotHeight}
                        stroke="#9CA3AF"
                        strokeWidth="2"
                      />
                      
                      {/* X-axis labels (dates) */}
                      {dashboardData.trends.map((trend, idx) => {
                        if (dashboardData.trends.length > 10 && idx % Math.ceil(dashboardData.trends.length / 8) !== 0) return null;
                        const x = padding.left + (idx / (dashboardData.trends.length - 1)) * plotWidth;
                        return (
                          <text
                            key={idx}
                            x={x}
                            y={padding.top + plotHeight + 20}
                            textAnchor="middle"
                            fontSize="10"
                            fill="#6B7280"
                            transform={`rotate(-45, ${x}, ${padding.top + plotHeight + 20})`}
                          >
                            {new Date(trend.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                          </text>
                        );
                      })}
                      
                      {/* Trend lines */}
                      <path
                        d={createPath(positivePoints)}
                        fill="none"
                        stroke="#10B981"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d={createPath(neutralPoints)}
                        fill="none"
                        stroke="#F59E0B"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d={createPath(negativePoints)}
                        fill="none"
                        stroke="#EF4444"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      
                      {/* Data points */}
                      {positivePoints.map((point, idx) => (
                        <circle key={`pos-${idx}`} cx={point.x} cy={point.y} r="4" fill="#10B981">
                          <title>{`${point.date}: ${point.value}% positive`}</title>
                        </circle>
                      ))}
                      {neutralPoints.map((point, idx) => (
                        <circle key={`neu-${idx}`} cx={point.x} cy={point.y} r="4" fill="#F59E0B">
                          <title>{`${point.date}: ${point.value}% neutral`}</title>
                        </circle>
                      ))}
                      {negativePoints.map((point, idx) => (
                        <circle key={`neg-${idx}`} cx={point.x} cy={point.y} r="4" fill="#EF4444">
                          <title>{`${point.date}: ${point.value}% negative`}</title>
                        </circle>
                      ))}
                      
                      {/* Y-axis label */}
                      <text
                        x={15}
                        y={padding.top + plotHeight / 2}
                        textAnchor="middle"
                        fontSize="12"
                        fill="#6B7280"
                        transform={`rotate(-90, 15, ${padding.top + plotHeight / 2})`}
                      >
                        Percentage (%)
                      </text>
                    </svg>
                  </div>
                  
                  {/* Data table below chart */}
                  <div style={{ marginTop: '30px', maxHeight: '200px', overflowY: 'auto' }}>
                    <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
                      <thead style={{ position: 'sticky', top: 0, background: '#F9FAFB' }}>
                        <tr>
                          <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #E5E7EB', color: '#6B7280' }}>Date</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #E5E7EB', color: '#10B981' }}>Positive</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #E5E7EB', color: '#F59E0B' }}>Neutral</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #E5E7EB', color: '#EF4444' }}>Negative</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #E5E7EB', color: '#6B7280' }}>Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dashboardData.trends.map((trend, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #F3F4F6' }}>
                            <td style={{ padding: '10px', color: '#374151' }}>
                              {new Date(trend.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                            </td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#10B981', fontWeight: '600' }}>
                              {trend.positive_pct}%
                            </td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#F59E0B', fontWeight: '600' }}>
                              {trend.neutral_pct}%
                            </td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#EF4444', fontWeight: '600' }}>
                              {trend.negative_pct}%
                            </td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#6B7280' }}>
                              {trend.total_count}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()}

            {/* Aspect Breakdown */}
            {dashboardData.aspects && dashboardData.aspects.length > 0 && (
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                marginBottom: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <h2 style={{ marginTop: 0, fontSize: '20px', color: '#333', marginBottom: '10px' }}>
                  🎯 Top Aspects
                </h2>
                <p style={{ color: '#666', fontSize: '14px', marginBottom: '20px' }}>
                  Positive and negative mentions for each key aspect
                </p>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '20px'
                }}>
                  {dashboardData.aspects.slice(0, 6).map((aspect, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '20px',
                        background: '#F9FAFB',
                        borderRadius: '12px',
                        border: '1px solid #E5E7EB'
                      }}
                    >
                      {/* Aspect Name */}
                      <div style={{
                        fontSize: '18px',
                        fontWeight: '700',
                        color: '#111827',
                        marginBottom: '15px',
                        textTransform: 'capitalize'
                      }}>
                        {aspect.aspect_name}
                      </div>
                      
                      {/* Positive Mentions */}
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '10px 12px',
                        background: '#D1FAE5',
                        borderRadius: '8px',
                        marginBottom: '8px'
                      }}>
                        <span style={{
                          fontSize: '14px',
                          fontWeight: '600',
                          color: '#065F46'
                        }}>
                          👍 Positive
                        </span>
                        <span style={{
                          fontSize: '16px',
                          fontWeight: '700',
                          color: '#10B981'
                        }}>
                          {aspect.positive_mentions || 0}
                        </span>
                      </div>
                      
                      {/* Negative Mentions */}
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '10px 12px',
                        background: '#FEE2E2',
                        borderRadius: '8px',
                        marginBottom: '8px'
                      }}>
                        <span style={{
                          fontSize: '14px',
                          fontWeight: '600',
                          color: '#991B1B'
                        }}>
                          👎 Negative
                        </span>
                        <span style={{
                          fontSize: '16px',
                          fontWeight: '700',
                          color: '#EF4444'
                        }}>
                          {aspect.negative_mentions || 0}
                        </span>
                      </div>
                      
                      {/* Neutral Mentions */}
                      {aspect.neutral_mentions > 0 && (
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '10px 12px',
                          background: '#FEF3C7',
                          borderRadius: '8px',
                          marginBottom: '8px'
                        }}>
                          <span style={{
                            fontSize: '14px',
                            fontWeight: '600',
                            color: '#92400E'
                          }}>
                            ⚖️ Neutral
                          </span>
                          <span style={{
                            fontSize: '16px',
                            fontWeight: '700',
                            color: '#F59E0B'
                          }}>
                            {aspect.neutral_mentions}
                          </span>
                        </div>
                      )}
                      
                      {/* Total */}
                      <div style={{
                        marginTop: '12px',
                        paddingTop: '12px',
                        borderTop: '1px solid #E5E7EB',
                        fontSize: '13px',
                        color: '#6B7280',
                        fontWeight: '600',
                        textAlign: 'center'
                      }}>
                        Total: {aspect.total_mentions || 0} mentions
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recent Batches */}
            {dashboardData.recent_batches && dashboardData.recent_batches.length > 0 && (
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '30px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <h2 style={{ marginTop: 0, fontSize: '20px', color: '#333', marginBottom: '20px' }}>
                  Recent Uploads
                </h2>
                <div style={{
                  border: '1px solid #E5E7EB',
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}>
                  {dashboardData.recent_batches.map((batch, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '15px 20px',
                        borderBottom: idx < dashboardData.recent_batches.length - 1 ? '1px solid #E5E7EB' : 'none',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '10px'
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: '600', color: '#333', marginBottom: '5px' }}>
                          {batch.filename}
                        </div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {new Date(batch.upload_timestamp).toLocaleString()}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
                        <div style={{ fontSize: '14px', color: '#666' }}>
                          {batch.processed_rows} / {batch.total_rows} processed
                        </div>
                        <div style={{
                          padding: '4px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: '600',
                          background: batch.status === 'completed' ? '#D1FAE5' : 
                                     batch.status === 'processing' ? '#FEF3C7' : '#FEE2E2',
                          color: batch.status === 'completed' ? '#15803D' : 
                                 batch.status === 'processing' ? '#92400E' : '#991B1B'
                        }}>
                          {batch.status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {dashboardData.statistics?.total_reviews === 0 && (
              <div style={{
                background: 'white',
                borderRadius: '16px',
                padding: '60px',
                textAlign: 'center',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '64px', marginBottom: '20px' }}>📊</div>
                <h3 style={{ fontSize: '24px', color: '#333', marginBottom: '10px' }}>
                  No Data Yet
                </h3>
                <p style={{ color: '#666', marginBottom: '30px' }}>
                  Upload your first batch of reviews to see analytics here
                </p>
                <button
                  onClick={() => router.push('/upload')}
                  style={{
                    padding: '14px 28px',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  📊 Upload Reviews
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
